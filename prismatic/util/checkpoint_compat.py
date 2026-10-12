"""Compatibility fixes for checkpoints written before TTT's FSDP-safe LR shape."""

import torch


def upgrade_ttt_lr_model_state(action_head_state: dict) -> int:
    """Turn legacy scalar TTT learning rates into one-element parameters in memory."""
    upgraded = 0
    for name, value in action_head_state.items():
        if name.rsplit(".", 1)[-1] == "learnable_lr" and isinstance(value, torch.Tensor) and value.ndim == 0:
            action_head_state[name] = value.reshape(1)
            upgraded += 1
    return upgraded


def upgrade_ttt_lr_optimizer_state(optimizer_state: dict) -> int:
    """Keep AdamW moments aligned with the upgraded one-element TTT parameters."""
    upgraded = 0
    for name, state in optimizer_state.get("state", {}).items():
        if not isinstance(name, str) or name.rsplit(".", 1)[-1] != "learnable_lr":
            continue
        for moment in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(moment)
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                state[moment] = value.reshape(1)
                upgraded += 1
    return upgraded


def align_singleton_adamw_moments(optimizer: torch.optim.Optimizer) -> int:
    """Repair a scalar moment if FSDP restores its local parameter as shape [1]."""
    upgraded = 0
    for parameter, state in optimizer.state.items():
        if tuple(parameter.shape) != (1,):
            continue
        for moment in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(moment)
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                state[moment] = value.reshape_as(parameter)
                upgraded += 1
    return upgraded
