"""Read-only slow-parameter validation with local, adapting TTT state."""

import random
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from prismatic.training.temporal import backward_temporal_segments
from prismatic.util.action_loss import valid_action_mask


def _distributed_context():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _collective_device():
    if dist.is_available() and dist.is_initialized() and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _seed_validation_rng(seed):
    """Seed host RNGs plus only the CUDA device owned by this process."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _check_equal_chunk_size(local_size, world_size):
    """Fail before an FSDP forward if validation iterators diverge across ranks."""
    if world_size == 1:
        return
    value = torch.tensor(local_size, dtype=torch.int64, device=_collective_device())
    gathered = [torch.empty_like(value) for _ in range(world_size)]
    dist.all_gather(gathered, value)
    sizes = [int(item.item()) for item in gathered]
    if len(set(sizes)) != 1:
        raise RuntimeError(f"Validation datasets differ across ranks: chunk sizes={sizes}.")


def _distributed_batches(loader, rank, world_size, seed_base):
    """Yield one distinct sample per rank while keeping FSDP forward counts equal.

    Every rank deterministically reads the same finite loader. A partial final
    group replays its first sample on otherwise idle ranks; those padding
    forwards participate in FSDP collectives but are excluded from metrics.
    """
    iterator = iter(loader)
    sample_index = 0
    while True:
        fallback = assigned = None
        chunk_size = 0
        for slot in range(world_size):
            try:
                _seed_validation_rng(seed_base + sample_index + slot)
                batch = next(iterator)
            except StopIteration:
                break
            if fallback is None:
                fallback = batch
            if slot == rank:
                assigned = batch
            chunk_size += 1
        _check_equal_chunk_size(chunk_size, world_size)
        if chunk_size == 0:
            return
        contributes = rank < chunk_size
        yield assigned if contributes else fallback, contributes, sample_index + (rank if contributes else 0)
        sample_index += chunk_size


def _sum_suite_stats(suite_sum, suite_count, sequences, world_size):
    if world_size == 1:
        return suite_sum, suite_count, sequences
    values = torch.tensor(
        [suite_sum, suite_count, sequences], dtype=torch.float64, device=_collective_device()
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0].item()), float(values[1].item()), int(round(values[2].item()))


@contextmanager
def validation_context(model, seed):
    """Restore mixed train/eval modes and caller RNGs even when validation fails.

    Use no_grad, not inference_mode: TTT still needs inner-loop differentiation.
    No parameter .grad is cleared or populated here.
    """
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    # Each torchrun process owns one current CUDA device. Forking every visible
    # device would make each rank touch its peers' GPUs during validation.
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            _seed_validation_rng(seed)
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
    """Token-weighted distributed loss per suite and overall.

    Batch size one and a per-sequence seed keep noise draws independent of the
    training batch size and distributed world size. Validation samples are
    sharded across ranks while every rank performs the same number of forwards,
    as required by FSDP.
    """
    metrics = {}
    if not datasets:
        raise ValueError("Validation requires at least one nonempty suite.")
    rank, world_size = _distributed_context()
    numerator = denominator = 0.0
    sequence_count = 0
    with validation_context(model, seed):
        for suite_index, (suite, dataset) in enumerate(sorted(datasets.items())):
            suite_sum = suite_count = 0.0
            sequences = 0
            suite_seed = seed + suite_index * 1_000_003
            loader = DataLoader(dataset, batch_size=1, collate_fn=collator, num_workers=0)
            for batch, contributes, sample_index in _distributed_batches(
                loader, rank, world_size, suite_seed
            ):
                # Stable across rank assignment and world size. Padding ranks
                # use the replayed sample's seed, though their result is ignored.
                _seed_validation_rng(suite_seed + sample_index)
                result = backward_temporal_segments(
                    model, batch, segment_size, move_to_device=move_to_device,
                    autocast_context=autocast_context, backward=False, check_finite=True,
                )
                if contributes:
                    count = int(valid_action_mask(
                        batch["actions"], batch.get("action_valid_mask"), batch.get("time_valid_mask")
                    ).sum().item())
                    suite_sum += result["loss_action"].item() * count
                    suite_count += count
                    sequences += batch["actions"].shape[0]
            suite_sum, suite_count, sequences = _sum_suite_stats(
                suite_sum, suite_count, sequences, world_size
            )
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
