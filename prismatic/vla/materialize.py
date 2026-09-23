"""Build the fixed LIBERO RLDS dataset and collator."""

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.constants import NUM_TOKENS
from prismatic.vla.datasets import RLDSDataset, VLABatchTransform


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    shuffle_buffer_size: int = 20_000,
    visual_token_pair_offset: int = 31,
    target_action_dim: int = 7,
    target_proprio_dim: int = 8,
    temporal_context_length: int = 1,
    flow_gr00t_placeholder_tokens: int = NUM_TOKENS,
    require_full_context: bool = False,
    validation_percent: int = 0,
    validation_sequences_per_suite: int = 16,
) -> Tuple[Dataset, PaddedCollatorForActionPrediction]:
    if flow_gr00t_placeholder_tokens < 1:
        raise ValueError("flow_gr00t_placeholder_tokens must be positive.")
    batch_transform = VLABatchTransform(
        base_tokenizer=tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=prompt_builder_fn,
        flow_gr00t_placeholder_tokens=flow_gr00t_placeholder_tokens,
        visual_token_pair_offset=visual_token_pair_offset,
        temporal_context_length=temporal_context_length,
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        padding_side="right",
        target_action_dim=target_action_dim,
        target_proprio_dim=target_proprio_dim,
        action_placeholder_tokens=flow_gr00t_placeholder_tokens,
        temporal_context_length=temporal_context_length,
    )
    dataset = RLDSDataset(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        require_full_context=require_full_context,
        shuffle_buffer_size=shuffle_buffer_size,
        visual_token_pair_offset=visual_token_pair_offset,
        temporal_context_length=temporal_context_length,
        validation_percent=validation_percent,
        validation_sequences_per_suite=validation_sequences_per_suite,
    )
    return dataset, collator
