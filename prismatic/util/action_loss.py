"""Action-token validity shared by the loss and segmented loss weighting."""

import torch


def valid_action_mask(actions, action_valid_mask=None, time_valid_mask=None):
    if actions.ndim not in (3, 4):
        raise ValueError("Actions must have shape [B,H,D] or [B,T,H,D].")
    mask = torch.ones(actions.shape[:-1], dtype=torch.bool, device=actions.device)
    if action_valid_mask is not None:
        if tuple(action_valid_mask.shape) != tuple(mask.shape):
            raise ValueError("action_valid_mask must match actions.shape[:-1].")
        mask = mask & action_valid_mask.to(device=actions.device, dtype=torch.bool)
    if time_valid_mask is not None:
        if actions.ndim != 4 or tuple(time_valid_mask.shape) != tuple(actions.shape[:2]):
            raise ValueError("Temporal action loss expects time_valid_mask with shape [B,T].")
        mask = mask & time_valid_mask.to(device=actions.device, dtype=torch.bool).unsqueeze(-1)
    return mask


def masked_action_mse(prediction, target, action_valid_mask=None, time_valid_mask=None):
    if prediction.shape != target.shape:
        raise ValueError("Action prediction and target must have identical shapes.")
    mask = valid_action_mask(target, action_valid_mask, time_valid_mask).unsqueeze(-1)
    # Accumulate in fp32 even when the network runs under BF16 autocast.
    squared_error = (prediction.float() - target.float()).square()
    numerator = torch.where(mask, squared_error, torch.zeros_like(squared_error)).sum()
    denominator = (mask.sum() * target.shape[-1]).clamp_min(1)
    return numerator / denominator
