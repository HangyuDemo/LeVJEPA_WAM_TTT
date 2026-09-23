"""Publish complete checkpoint files without exposing a partially written latest."""

import os
import shutil
import tempfile
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
