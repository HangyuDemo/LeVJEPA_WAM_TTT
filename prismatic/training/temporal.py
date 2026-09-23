"""Segmented action-only TBPTT; one batch contains independent long sequences."""

from contextlib import nullcontext

import torch

from prismatic.util.action_loss import valid_action_mask
from prismatic.training.numerics import ensure_finite


def detach_fast_weights(value):
    """Preserve state values and namedtuple types, but release the prior graph."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: detach_fast_weights(child) for key, child in value.items()}
    if isinstance(value, tuple):
        children = [detach_fast_weights(child) for child in value]
        return type(value)(*children) if hasattr(value, "_fields") else tuple(children)
    if isinstance(value, list):
        return [detach_fast_weights(child) for child in value]
    return value


def _slice_time(value, start, end):
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _slice_time(child, start, end) for key, child in value.items()}
    return value[:, start:end]


def backward_temporal_segments(
    model, batch, segment_size, accum_divisor=1,
    move_to_device=lambda value: value, autocast_context=nullcontext,
    backward=True, check_finite=True,
):
    """Backward each segment immediately, carrying state only within this batch.

    No optimizer step occurs here. Each sample is one ordered, single-episode
    sequence; shuffled samples/microbatches always start with fresh memory.
    Weight segment means by valid action-token counts so tail padding cannot
    give short segments disproportionate influence. Full image sequences stay
    on CPU and only the current segment is moved to the accelerator.
    """
    actions = batch["actions"]
    if actions.ndim != 4 or segment_size < 2 or actions.shape[1] % segment_size:
        raise ValueError("Segmented TTT requires [B,T,H,D], segment_size>=2 and T divisible by it.")
    if batch.get("pair_pixel_values") is not None:
        raise ValueError("Segmented action-only TTT requires visual_token_pair_offset=0.")
    if accum_divisor < 1:
        raise ValueError("accum_divisor must be positive.")
    mask = valid_action_mask(actions, batch.get("action_valid_mask"), batch.get("time_valid_mask"))
    total_valid = int(mask.sum().item())
    if total_valid == 0:
        raise ValueError("A temporal training batch must contain valid action targets.")
    state = None
    total_loss = None
    for start in range(0, actions.shape[1], segment_size):
        end = start + segment_size
        segment = {key: batch[key] for key in ("input_ids", "attention_mask")}
        for key in ("pixel_values", "actions", "proprio", "time_valid_mask", "action_valid_mask"):
            segment[key] = _slice_time(batch.get(key), start, end)
        segment = move_to_device(segment)
        weight = int(mask[:, start:end].sum().item()) / total_valid
        with autocast_context():
            output = model(**segment, prev_fast_weights=state, return_fast_weights=True)
            if output.get("loss_visual_token_cosine") is not None:
                raise ValueError("Segmented TTT currently supports action loss only.")
            loss = output["loss"] * weight
        # Free each graph before the next forward. Accumulate slow gradients
        # across segments, then across microbatches in the outer training loop.
        if check_finite:
            ensure_finite(
                {"loss": loss, "fast_weights": output["next_fast_weights"]}, f"segment[{start}:{end}]"
            )
        if backward:
            (loss / accum_divisor).backward()
        state = detach_fast_weights(output["next_fast_weights"])
        total_loss = loss.detach() if total_loss is None else total_loss + loss.detach()
        del output, loss, segment
    # Deliberately do not return state: the next long sequence is independent.
    return {"loss": total_loss, "loss_action": total_loss}
