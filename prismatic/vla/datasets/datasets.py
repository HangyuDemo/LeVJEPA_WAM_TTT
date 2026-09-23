"""Fixed LIBERO RLDS dataset adapter for JEPA-WAM."""

import hashlib
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.constants import (
    ACTION_TOKEN_BEGIN_IDX,
    NUM_ACTIONS_CHUNK,
    NUM_TOKENS,
)
from prismatic.vla.datasets.rlds import make_interleaved_dataset
from prismatic.vla.datasets.rlds.dataset import make_validation_dataset
from prismatic.vla.datasets.validation_split import episode_split_spec
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights


def _stack_pixel_values(values):
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values)
    if isinstance(values[0], dict):
        return {key: _stack_pixel_values([value[key] for value in values]) for key in values[0]}
    raise ValueError(f"Unsupported pixel value type `{type(values[0])}`")


def _frame_to_pil(frame: Any) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if isinstance(frame, (bytes, bytearray, np.bytes_)):
        return Image.open(BytesIO(frame)).convert("RGB")
    if isinstance(frame, np.ndarray):
        # RLDS chunking adds singleton window axes before the image channels.
        # Remove those axes so PIL receives the expected [H, W, C] image.
        array = frame
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            raise ValueError(f"Expected image with shape [H, W, C], got {array.shape}")
        return Image.fromarray(array).convert("RGB")
    raise ValueError(f"Unsupported frame type `{type(frame)}`")


def _transform_sequence(image_transform: ImageTransform, frames) -> Any:
    return _stack_pixel_values([image_transform(_frame_to_pil(frame)) for frame in frames])


@dataclass
class VLABatchTransform:
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    flow_gr00t_placeholder_tokens: int = NUM_TOKENS
    visual_token_pair_offset: int = 31
    temporal_context_length: int = 1

    @staticmethod
    def _wrist_keys(observation: Dict[str, Any], paired: bool = False) -> list[str]:
        prefix = "pair_image_" if paired else "image_"
        return sorted(key for key in observation if key.startswith(prefix) and "wrist" in key)

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        observation = rlds_batch["observation"]
        context = self.temporal_context_length
        if context < 1:
            raise ValueError("temporal_context_length must be positive.")
        if observation["image_primary"].shape[0] != context:
            raise ValueError(
                f"Expected {context} observation frames, got {observation['image_primary'].shape[0]}."
            )
        instruction = rlds_batch["task"]["language_instruction"].decode().lower()

        prompt_builder = self.prompt_builder_fn("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {instruction}?")
        prompt_builder.add_turn("gpt", "")
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        input_ids.extend([ACTION_TOKEN_BEGIN_IDX] * self.flow_gr00t_placeholder_tokens)

        primary_frames = [
            self.image_transform(_frame_to_pil(frame)) for frame in observation["image_primary"]
        ]
        output = {
            "pixel_values": primary_frames[0] if context == 1 else _stack_pixel_values(primary_frames),
            "input_ids": torch.tensor(input_ids),
            "dataset_name": rlds_batch["dataset_name"],
        }

        actions = rlds_batch["action"]
        expected_action_length = context + NUM_ACTIONS_CHUNK - 1
        if actions.shape[0] != expected_action_length:
            raise ValueError(
                f"Expected {expected_action_length} action entries for context={context}, got {actions.shape[0]}."
            )
        if context == 1:
            output["actions"] = actions
        else:
            output["actions"] = np.stack(
                [actions[index : index + NUM_ACTIONS_CHUNK] for index in range(context)], axis=0
            )

        wrist_keys = self._wrist_keys(observation)
        if not wrist_keys:
            raise ValueError("The public recipe requires a wrist-camera observation.")
        if context == 1:
            output["pixel_values_wrist"] = _stack_pixel_values(
                [self.image_transform(_frame_to_pil(observation[key][0])) for key in wrist_keys]
            )
        else:
            output["pixel_values_wrist"] = _stack_pixel_values(
                [
                    _stack_pixel_values(
                        [self.image_transform(_frame_to_pil(frame)) for frame in observation[key]]
                    )
                    for key in wrist_keys
                ]
            ).transpose(0, 1)

        if "proprio" in observation:
            output["proprio"] = observation["proprio"] if context > 1 else observation["proprio"]
        if "action_valid_mask" in rlds_batch:
            action_valid_mask = rlds_batch["action_valid_mask"]
            output["action_valid_mask"] = (
                action_valid_mask
                if context == 1
                else np.stack(
                    [action_valid_mask[index : index + NUM_ACTIONS_CHUNK] for index in range(context)], axis=0
                )
            )
        if context > 1:
            output["time_valid_mask"] = observation["pad_mask"].astype(np.bool_)

        if self.visual_token_pair_offset > 0:
            if "pair_image_primary" not in observation:
                raise ValueError("Paired primary frames are required for visual-token cosine supervision.")
            if context == 1:
                output["pair_pixel_values"] = _transform_sequence(
                    self.image_transform, observation["pair_image_primary"][0]
                )
            else:
                output["pair_pixel_values"] = _stack_pixel_values(
                    [_transform_sequence(self.image_transform, frames) for frames in observation["pair_image_primary"]]
                )

            pair_wrist_keys = self._wrist_keys(observation, paired=True)
            if not pair_wrist_keys:
                raise ValueError("Paired wrist frames are required for visual-token cosine supervision.")
            if context == 1:
                output["pair_pixel_values_wrist"] = _stack_pixel_values(
                    [_transform_sequence(self.image_transform, observation[key][0]) for key in pair_wrist_keys]
                )
            else:
                output["pair_pixel_values_wrist"] = _stack_pixel_values(
                    [
                        _stack_pixel_values(
                            [_transform_sequence(self.image_transform, frames) for frames in observation[key]]
                        )
                        for key in pair_wrist_keys
                    ]
                ).transpose(0, 1)

        return output


class ValidationDataset(IterableDataset):
    def __init__(self, dataset, batch_transform):
        self.dataset, self.batch_transform = dataset, batch_transform

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError("ValidationDataset requires num_workers=0 for deterministic, nonduplicated reads.")
        for batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(batch)


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: VLABatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 20_000,
        visual_token_pair_offset: int = 31,
        temporal_context_length: int = 1,
        require_full_context: bool = False,
        validation_percent: int = 0,
        validation_sequences_per_suite: int = 16,
    ) -> None:
        if data_mix != "libero_4_task_suites_no_noops":
            raise ValueError("The public recipe only supports `libero_4_task_suites_no_noops`.")

        self.batch_transform = batch_transform
        if temporal_context_length < 1:
            raise ValueError("temporal_context_length must be positive.")
        frame_transform_threads = int(os.getenv("VLA_RLDS_FRAME_TRANSFORM_THREADS", "16"))
        if frame_transform_threads <= 0:
            raise ValueError("VLA_RLDS_FRAME_TRANSFORM_THREADS must be positive.")

        mixture_spec = OXE_NAMED_MIXTURES[data_mix]
        dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            data_root_dir,
            mixture_spec,
        )
        self.split_manifest = {}
        validation_kwargs = []
        for kwargs in dataset_kwargs:
            builder = tfds.builder(kwargs["name"], data_dir=kwargs["data_dir"])
            spec = episode_split_spec(int(builder.info.splits["train"].num_examples), validation_percent)
            if validation_percent:
                metadata = Path(str(builder.data_dir)) / "dataset_info.json"
                self.split_manifest[kwargs["name"]] = {
                    **spec, "builder": builder.info.full_name,
                    "dataset_info_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                    "validation_window": "middle complete context per episode",
                    "validation_max_sequences": validation_sequences_per_suite,
                    "normalization": "existing full-dataset statistics retained for pretrained action compatibility",
                }
                validation_kwargs.append({**kwargs, "split": spec["validation_split"]})
                kwargs["split"] = spec["train_split"]
        frame_transform_kwargs = {
            "resize_size": resize_resolution,
            "num_parallel_calls": frame_transform_threads,
        }
        trajectory_kwargs = {
            "window_size": temporal_context_length,
            "future_action_window_size": NUM_ACTIONS_CHUNK - 1,
            "pair_target_offset": visual_token_pair_offset,
            "skip_unlabeled": True,
            "goal_relabeling_strategy": "uniform",
        }
        self.dataset, self.global_dataset_length, self.dataset_statistics = make_interleaved_dataset(
            traj_transform_kwargs=trajectory_kwargs,
            frame_transform_kwargs=frame_transform_kwargs,
            dataset_kwargs_list=dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            require_full_context=require_full_context,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
        )
        self.dataset_length = self.global_dataset_length
        self.validation_datasets = {
            kwargs["name"]: ValidationDataset(
                make_validation_dataset(
                    kwargs, self.dataset_statistics[kwargs["name"]], trajectory_kwargs,
                    frame_transform_kwargs, validation_sequences_per_suite,
                ), batch_transform,
            )
            for kwargs in validation_kwargs
        }

    def __iter__(self):
        datasets = self.dataset if isinstance(self.dataset, list) else [self.dataset]
        for dataset in datasets:
            for batch in dataset.as_numpy_iterator():
                yield self.batch_transform(batch)

    def __len__(self) -> int:
        return self.dataset_length
