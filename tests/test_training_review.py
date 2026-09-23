import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from prismatic.training.checkpoint_io import save_training_checkpoint
from prismatic.training.metrics import VLAMetrics


def test_accumulated_loss_logs_all_microbatches(tmp_path):
    metrics = VLAMetrics(("jsonl",), "accum", tmp_path, {}, grad_accumulation_steps=4)
    for value in (1.0, 2.0, 3.0, 10.0):
        loss = torch.tensor(value, requires_grad=True)
        metrics.commit(loss=loss, loss_action=loss)
    metrics.commit(global_step=1)
    metrics.push()
    entry = json.loads((tmp_path / "accum.jsonl").read_text())
    for name in ("VLA Train/Loss", "VLA Train/Loss Action", "VLA Train/Loss (Raw)"):
        assert entry[name] == 4.0
    assert all(not item.requires_grad for item in metrics.state["loss_action"])


@pytest.mark.skipif(os.environ.get("RUN_WANDB_OFFLINE_TEST") != "1", reason="Opt-in SDK service test needs local sockets")
def test_wandb_offline_tracker_records_validation(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    from prismatic.training.metrics import WandBTracker

    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    tracker = WandBTracker("validation-review", tmp_path, {}, enabled=True)
    run = wandb.run
    try:
        tracker.write(1, {"Validation/Loss Action": 0.125})
        run_directory = Path(run.dir).parent
    finally:
        tracker.finalize()
    # Explicit-step logging is buffered until the next step/finish. The SDK
    # fetches this final summary from the service after flushing history.
    from wandb.sdk.lib import proto_util

    summary = proto_util.dict_from_proto_list(run._final_summary.item)
    assert summary["Validation/Loss Action"] == 0.125
    assert next(run_directory.glob("*.wandb")).stat().st_size > 0


def test_checkpoint_publish_preserves_latest_if_serialization_fails(tmp_path, monkeypatch):
    import prismatic.training.checkpoint_io as checkpoint_io

    first = tmp_path / "step-000001.pt"
    latest = tmp_path / "latest-checkpoint.pt"
    save_training_checkpoint({"step": 1}, first)
    assert torch.load(first) == torch.load(latest) == {"step": 1}

    def interrupted_save(payload, handle):
        handle.write(b"incomplete checkpoint")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(checkpoint_io.torch, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted"):
        save_training_checkpoint({"step": 2}, tmp_path / "step-000002.pt")
    assert torch.load(latest) == {"step": 1}
    assert not (tmp_path / "step-000002.pt").exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_checkpoint_publish_preserves_latest_if_copy_fails(tmp_path, monkeypatch):
    import prismatic.training.checkpoint_io as checkpoint_io

    save_training_checkpoint({"step": 1}, tmp_path / "step-000001.pt")

    def interrupted_copy(source, handle):
        handle.write(b"incomplete latest")
        raise OSError("simulated disk full")

    monkeypatch.setattr(checkpoint_io.shutil, "copyfileobj", interrupted_copy)
    with pytest.raises(OSError, match="disk full"):
        save_training_checkpoint({"step": 2}, tmp_path / "step-000002.pt")
    assert torch.load(tmp_path / "latest-checkpoint.pt") == {"step": 1}
    assert torch.load(tmp_path / "step-000002.pt") == {"step": 2}
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("gb,interval", [(1, 1000), (4, 250)])
def test_four_route_names_and_validation_cadence(gb, interval):
    root = Path(__file__).resolve().parents[1]
    names = []
    for architecture in ("wrapper", "inline"):
        for memory in ("jepa-memory", "action-kv"):
            env = {key: value for key, value in os.environ.items() if not key.startswith((
                "TTT_", "VALIDATION_", "GLOBAL_BATCH", "PER_DEVICE_BATCH", "RUN_ID", "MAX_STEPS",
                "OBSERVATION_BUDGET", "SAVE_INTERVAL", "PARAMETER_CHECK_INTERVAL",
            ))}
            env.update(REPO_ROOT=str(root), TTT_MEMORY_TAG=memory, TTT_ARCHITECTURE=architecture,
                       GLOBAL_BATCH_SIZE=str(gb))
            result = subprocess.check_output([
                "bash", "-c", 'source "$REPO_ROOT/vla-scripts/ttt_round2_config.sh"',
            ], env=env, text=True)
            assert f"validation_interval={interval}" in result
            assert "/checkpoints/2nd_round" in result
            name = result.split("run_id=")[1].splitlines()[0]
            assert name.endswith("-val5-n16-s7")
            names.append(name)
    assert len(set(names)) == 4
