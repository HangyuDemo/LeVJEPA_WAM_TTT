"""Local JEPA-WAM checkpoint loading and inference helpers."""

import os
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image

from prismatic.models import load_vla

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _is_native_prismatic_checkpoint_path(model_path: Union[str, Path]) -> bool:
    path = Path(os.path.expanduser(str(model_path)))
    return path.is_file() and path.suffix == ".pt" and path.parent.name == "checkpoints"


def get_vla(cfg: Any) -> torch.nn.Module:
    checkpoint = Path(os.path.expanduser(str(cfg.pretrained_checkpoint)))
    if not _is_native_prismatic_checkpoint_path(checkpoint):
        raise ValueError("Evaluation requires a local `runs/.../checkpoints/*.pt` JEPA-WAM checkpoint.")
    model = load_vla(
        checkpoint,
        load_for_training=False,
        base_vlm=cfg.base_vlm,
        llm_checkpoint_path=cfg.llm_checkpoint_path,
        vjepa_checkpoint_path=cfg.vjepa_checkpoint_path,
        levjepa_checkpoint_path=getattr(cfg, "levjepa_checkpoint_path", None),
        # TTT memory is built from the WAM prediction head's visual
        # representation, so this head must be restored for deployment.
        load_visual_token_cosine_head=True,
    )
    model.validate_ttt_rollout_stride(cfg.num_open_loop_steps)
    fm_steps_override = os.environ.get("FM_NUM_INFERENCE_TIMESTEPS")
    if fm_steps_override is not None:
        fm_steps = int(fm_steps_override)
        if fm_steps < 1:
            raise ValueError("FM_NUM_INFERENCE_TIMESTEPS must be positive.")
        model.action_head.num_inference_timesteps = fm_steps
    updates = getattr(cfg, "ttt_inference_updates", 1)
    if not 1 <= updates <= model.action_head.num_inference_timesteps:
        raise ValueError(
            "ttt_inference_updates must be between 1 and "
            f"num_inference_timesteps={model.action_head.num_inference_timesteps}, got {updates}."
        )
    model.action_head.ttt_inference_updates = updates
    fixed_position = os.environ.get("TTT_INFERENCE_FIXED_OBSERVATION_POSITION", "0")
    if fixed_position not in {"0", "1"}:
        raise ValueError("TTT_INFERENCE_FIXED_OBSERVATION_POSITION must be 0 or 1.")
    model.action_head.ttt_fixed_observation_position = fixed_position == "1"
    slow_only = os.environ.get("TTT_INFERENCE_SLOW_ONLY", "0")
    if slow_only not in {"0", "1"}:
        raise ValueError("TTT_INFERENCE_SLOW_ONLY must be 0 or 1.")
    model.action_head.ttt_inference_slow_only = slow_only == "1"
    return model.eval().to(DEVICE)


def get_vla_action(cfg: Any, vla: torch.nn.Module, obs: Dict[str, Any], task_label: str) -> List[np.ndarray]:
    images = [Image.fromarray(obs["full_image"]), Image.fromarray(obs["wrist_image"])]
    action_chunk = vla.predict_action(
        image=images,
        instruction=task_label,
        unnorm_key=cfg.unnorm_key,
        proprio=obs["state"],
    )
    if action_chunk.ndim == 1:
        action_chunk = action_chunk[None]
    return [action_chunk[index] for index in range(min(len(action_chunk), cfg.num_open_loop_steps))]
