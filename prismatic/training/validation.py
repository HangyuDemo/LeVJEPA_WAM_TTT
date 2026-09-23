"""Read-only slow-parameter validation with local, adapting TTT state."""

import random
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader

from prismatic.training.temporal import backward_temporal_segments
from prismatic.util.action_loss import valid_action_mask


@contextmanager
def validation_context(model, seed):
    """Restore mixed train/eval modes and caller RNGs even when validation fails.

    Use no_grad, not inference_mode: TTT still needs inner-loop differentiation.
    No parameter .grad is cleared or populated here.
    """
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model.eval()
            with torch.no_grad():
                yield
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def evaluate_action_loss(
    model, datasets, collator, segment_size, seed=7,
    move_to_device=lambda value: value, autocast_context=nullcontext,
):
    """Token-weighted loss per suite and overall; batch=1 keeps fixed noise draws
    and sequence selection independent of the training batch size.
    """
    metrics = {}
    if not datasets:
        raise ValueError("Validation requires at least one nonempty suite.")
    numerator = denominator = 0.0
    sequence_count = 0
    with validation_context(model, seed):
        for suite, dataset in sorted(datasets.items()):
            suite_sum = suite_count = 0.0
            sequences = 0
            loader = DataLoader(dataset, batch_size=1, collate_fn=collator, num_workers=0)
            for batch in loader:
                count = int(valid_action_mask(
                    batch["actions"], batch.get("action_valid_mask"), batch.get("time_valid_mask")
                ).sum().item())
                result = backward_temporal_segments(
                    model, batch, segment_size, move_to_device=move_to_device,
                    autocast_context=autocast_context, backward=False, check_finite=True,
                )
                suite_sum += result["loss_action"].item() * count
                suite_count += count
                sequences += batch["actions"].shape[0]
            if suite_count == 0:
                raise ValueError(f"Validation suite {suite} has no valid action targets; check holdout/context.")
            metrics[f"Validation/{suite}/Loss Action"] = suite_sum / suite_count
            metrics[f"Validation/{suite}/Sequences"] = sequences
            numerator += suite_sum
            denominator += suite_count
            sequence_count += sequences
    metrics["Validation/Loss Action"] = numerator / denominator
    metrics["Validation/Valid Action Tokens"] = denominator
    metrics["Validation/Sequences"] = sequence_count
    return metrics
