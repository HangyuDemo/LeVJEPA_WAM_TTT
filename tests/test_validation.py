import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prismatic.training.numerics import ensure_finite
from prismatic.training.temporal import backward_temporal_segments
from prismatic.training.validation import evaluate_action_loss, validation_context
from prismatic.vla.datasets.validation_split import episode_split_spec
from test_ttt_round2 import TinyTTTPolicy, ScalarPolicy, make_batch


def test_episode_partitions_are_disjoint_and_exhaustive():
    import tensorflow_datasets as tfds

    for n in (432, 400, 1000):
        spec = episode_split_spec(n, 5)
        ranges = [tfds.core.ReadInstruction.from_spec(spec[key]).to_absolute({"train": SimpleNamespace(num_examples=n)})[0]
                  for key in ("train_split", "validation_split")]
        indices = [set(range(part.from_ or 0, part.to if part.to is not None else n)) for part in ranges]
        assert not indices[0] & indices[1]
        assert indices[0] | indices[1] == set(range(n))
        assert len(indices[1]) == n * 5 // 100
    assert episode_split_spec(432, 0)["train_split"] == "train"
    with pytest.raises(ValueError):
        episode_split_spec(10, 5)


def test_middle_window_selection_skips_short_episodes():
    import tensorflow as tf
    from prismatic.vla.datasets.rlds.dataset import select_validation_window
    from prismatic.vla.datasets.rlds.traj_transforms import chunk_act_obs

    for length in (2, 4, 9):
        values = tf.range(length, dtype=tf.float32)
        traj = dict(action=values[:, None], observation={"proprio": values[:, None]},
                    task={"instruction": tf.fill([length], "task")}, dataset_name=tf.fill([length], "libero"),
                    absolute_action_mask=tf.zeros([length, 1], tf.bool))
        result = select_validation_window(chunk_act_obs(traj, 4, 2))
        windows = result["observation"]["proprio"].numpy()
        assert windows.shape[0] == (1 if length >= 4 else 0)
        if length >= 4:
            assert result["observation"]["pad_mask"].numpy().all()
            assert (np.diff(windows[0, :, 0]) == 1).all()


@pytest.mark.parametrize("architecture", ["inline", "wrapper"])
@pytest.mark.parametrize("source", ["jepa", "action_tokens"])
def test_validation_carries_fast_state_without_changing_weights_gradients_or_rng(architecture, source):
    model = TinyTTTPolicy(architecture, source)
    batch = make_batch(batch_size=1)
    model.train()
    model.model.transformer_blocks[0].eval()  # Preserve mixed modes, not just the root flag.
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    modes = [module.training for module in model.modules()]
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.ones_like(param)
    gradients = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    rng = torch.random.get_rng_state().clone()
    datasets = {"spatial": [batch], "object": [batch]}
    result = evaluate_action_loss(model, datasets, lambda items: items[0], 2)
    repeated = evaluate_action_loss(model, datasets, lambda items: items[0], 2)
    assert result == repeated
    assert torch.equal(rng, torch.random.get_rng_state())
    assert modes == [module.training for module in model.modules()]
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, before[name], rtol=0, atol=0)
        if name in gradients:
            torch.testing.assert_close(param.grad, gradients[name], rtol=0, atol=0)
        else:
            assert param.grad is None
    assert model.calls[0] is None and model.calls[4] is None
    for state in model.calls[3]:
        assert state.step.item() == 6
        assert all(not tensor.requires_grad for tensor in state.fast_weights.values())


def test_validation_loss_is_weighted_by_valid_targets():
    model = ScalarPolicy()
    first, second = make_batch(1), make_batch(1)
    first["actions"].fill_(0.5)
    second["actions"].fill_(1.5)
    second["action_valid_mask"][:, :, 1] = False
    result = evaluate_action_loss(model, {"first": [first], "second": [second]}, lambda items: items[0], 2)
    assert result["Validation/first/Loss Action"] == 0
    assert result["Validation/second/Loss Action"] == 1
    assert result["Validation/Loss Action"] == pytest.approx(1 / 3)
    assert model.weight.grad is None


def test_validation_restores_rng_and_modes_on_exception():
    model = ScalarPolicy().train()
    random.seed(123)
    np.random.seed(123)
    python_before, numpy_before = random.getstate(), np.random.get_state()
    with pytest.raises(RuntimeError):
        with validation_context(model, 7):
            assert not model.training
            random.random()
            np.random.rand()
            raise RuntimeError("bad validation")
    assert model.training
    assert random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])


def test_nonfinite_fast_state_fails_before_backward():
    class BadState(ScalarPolicy):
        def forward(self, **kwargs):
            output = super().forward(**kwargs)
            output["next_fast_weights"] = {"memory": torch.tensor(float("nan"))}
            return output

    model = BadState()
    with pytest.raises(FloatingPointError, match=r"segment\[0:2\].fast_weights.memory"):
        backward_temporal_segments(model, make_batch(1), 2)
    assert model.weight.grad is None
    with pytest.raises(FloatingPointError, match="gradient"):
        ensure_finite(torch.tensor(float("inf")), "gradient")


def test_training_loop_validates_saves_best_and_rejects_bad_gradient(tmp_path, monkeypatch):
    from prismatic.training.strategies.base_strategy import TrainingStrategy
    from prismatic.training.metrics import VLAMetrics
    import prismatic.training.strategies.base_strategy as base

    class Sequences(torch.utils.data.IterableDataset):
        def __iter__(self):
            while True:
                yield batch

        def __len__(self):
            return 8

    class Strategy(TrainingStrategy):
        def __init__(self, model):
            self.vlm = model
            self.max_steps = 2
            self.per_device_batch_size = self.global_batch_size = self.grad_accumulation_steps = 1
            self.worker_init_fn = None
            self.enable_mixed_precision_training = False
            self.mixed_precision_dtype = torch.bfloat16
            self.ttt_segment_size = 2
            self.ttt_require_full_context = True
            self.cpu_memory_log_interval = 0
            self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1)
            self.bad_gradient = False

        def _move_batch_to_device(self, batch):
            return batch

        def run_setup(self, *args):
            pass

        def clip_grad_norm(self):
            if self.bad_gradient:
                return torch.tensor(float("nan"))
            return torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), 1)

        def save_checkpoint(self, run_dir, global_step, epoch, train_loss, **kwargs):
            torch.save(self.vlm.state_dict(), run_dir / "checkpoints" /
                       f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt")

    monkeypatch.setattr(base.dist, "barrier", lambda: None)
    batch = make_batch(1)
    (tmp_path / "checkpoints").mkdir()
    strategy = Strategy(ScalarPolicy())
    metrics = VLAMetrics(("jsonl",), "test", tmp_path, {})
    strategy.run_vla_training(Sequences(), lambda items: items[0], metrics,
                              validation_datasets={"spatial": [batch]}, validation_interval=1)
    records = [json.loads(line) for line in (tmp_path / "validation-metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == [0, 1, 2]
    assert (tmp_path / "checkpoints/best-validation-checkpoint.pt").is_file()
    assert metrics.global_step == 2
    assert strategy.vlm.training

    strategy = Strategy(ScalarPolicy())
    strategy.bad_gradient = True
    weight = strategy.vlm.weight.detach().clone()
    metrics = VLAMetrics(("jsonl",), "failure", tmp_path, {})
    with pytest.raises(FloatingPointError, match="gradient_norm"):
        strategy.run_vla_training(Sequences(), lambda items: items[0], metrics)
    assert metrics.global_step == 0
    assert torch.equal(weight, strategy.vlm.weight)
    assert (tmp_path / "numerical-failure.json").is_file()
