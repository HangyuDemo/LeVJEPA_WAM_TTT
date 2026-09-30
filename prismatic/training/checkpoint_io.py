"""Publish complete checkpoint files without exposing a partially written latest."""

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import torch


def save_training_checkpoint(payload, checkpoint_path: Path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(dir=checkpoint_path.parent, prefix=".checkpoint-", suffix=".tmp", delete=False) as handle:
            pending = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, checkpoint_path)
        pending = None
        if checkpoint_path.name == "latest-checkpoint.pt":
            return
        # Keep latest as a standalone regular file for existing evaluation tools.
        with tempfile.NamedTemporaryFile(dir=checkpoint_path.parent, prefix=".latest-", suffix=".tmp", delete=False) as handle:
            pending = Path(handle.name)
            with checkpoint_path.open("rb") as source:
                shutil.copyfileobj(source, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, checkpoint_path.parent / "latest-checkpoint.pt")
        pending = None
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def preserve_best_checkpoint(source: Path, best_path: Path):
    """Atomically retain the best weights independently of a changing latest file.

    A hard link shares disk blocks while latest and best are identical. Replacing
    latest later leaves the previous best inode intact. Copy only when hard links
    are unavailable on the checkpoint filesystem.
    """
    source, best_path = Path(source).resolve(strict=True), Path(best_path)
    pending = best_path.parent / f".best-{uuid.uuid4().hex}.tmp"
    try:
        try:
            os.link(source, pending)
        except OSError:
            with pending.open("xb") as handle, source.open("rb") as saved:
                shutil.copyfileobj(saved, handle)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(pending, best_path)
    finally:
        pending.unlink(missing_ok=True)


def prune_step_checkpoints(checkpoint_dir: Path):
    """Discard redundant periodic files after latest and best are both safe."""
    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "latest-checkpoint.pt").is_file():
        raise FileNotFoundError("Cannot prune step checkpoints without latest-checkpoint.pt.")
    best = checkpoint_dir / "best-validation-checkpoint.pt"
    if best.is_symlink():
        preserve_best_checkpoint(best, best)
    for checkpoint in checkpoint_dir.glob("step-*.pt"):
        checkpoint.unlink()
