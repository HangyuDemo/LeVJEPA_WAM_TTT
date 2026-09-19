from contextlib import nullcontext
from pathlib import Path
import os
import subprocess

import pytest
import torch

from prismatic.models.flow_matching_head.cross_attention_dit import DiT, TTTFastWeightState
from prismatic.training.temporal import backward_temporal_segments, detach_fast_weights
from prismatic.util.action_loss import masked_action_mse
from prismatic.conf.vla import VLAConfig


ROOT = Path(__file__).resolve().parents[1]


def test_segmented_config_rejects_unsupported_shapes_and_visual_loss():
    kwargs = dict(ttt_enabled=True, ttt_carry_between_segments=True, ttt_context_length=32,
                  ttt_tbptt_step_size=8, visual_token_pair_offset=0)
    VLAConfig(**kwargs)
    with pytest.raises(ValueError, match="divisible"):
        VLAConfig(**{**kwargs, "ttt_context_length":33})
    with pytest.raises(ValueError, match="visual_token_pair_offset"):
        VLAConfig(**{**kwargs, "visual_token_pair_offset":31})


def make_batch(batch_size=2, time=8):
    return dict(
        input_ids=torch.ones(batch_size, 3, dtype=torch.long),
        attention_mask=torch.ones(batch_size, 3, dtype=torch.bool),
        pixel_values={"vision": torch.randn(batch_size, time, 3, 8)},
        actions=torch.randn(batch_size, time, 2, 8),
        proprio=torch.zeros(batch_size, time, 4),
        time_valid_mask=torch.ones(batch_size, time, dtype=torch.bool),
        action_valid_mask=torch.ones(batch_size, time, 2, dtype=torch.bool),
    )


def test_action_mask_excludes_tail_and_invalid_time_from_value_and_gradient():
    pred = torch.ones(1, 2, 3, 2, requires_grad=True)
    target = torch.zeros_like(pred)
    target[:, 0, 1:] = 10000
    target[:, 1] = 10000
    action_mask = torch.tensor([[[True, False, False], [True, True, True]]])
    time_mask = torch.tensor([[True, False]])
    loss = masked_action_mse(pred, target, action_mask, time_mask)
    loss.backward()
    assert loss.item() == 1
    assert torch.count_nonzero(pred.grad).item() == 2
    torch.testing.assert_close(pred.grad[0, 0, 0], torch.ones(2))


def test_all_padding_has_finite_zero_loss_and_grad():
    pred = torch.randn(2, 3, 4, requires_grad=True)
    loss = masked_action_mse(pred, torch.zeros_like(pred), torch.zeros(2, 3, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(pred.grad).item() == 0


def test_mask_shape_is_validated():
    with pytest.raises(ValueError, match="action_valid_mask"):
        masked_action_mse(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.ones(1, 3))


class ScalarPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, actions, action_valid_mask, time_valid_mask, prev_fast_weights=None, **kwargs):
        loss = masked_action_mse(self.weight.expand_as(actions), actions, action_valid_mask, time_valid_mask)
        return {"loss": loss, "next_fast_weights": (self.weight * 2,)}


def test_segment_gradients_match_full_masked_objective_and_accumulation():
    batch = make_batch()
    batch["action_valid_mask"][:, 4:, 1] = False
    reference = ScalarPolicy()
    reference(**batch)["loss"].backward()
    segmented = ScalarPolicy()
    result = backward_temporal_segments(segmented, batch, 2)
    torch.testing.assert_close(segmented.weight.grad, reference.weight.grad)
    torch.testing.assert_close(result["loss"], reference(**batch)["loss"])
    accumulated = ScalarPolicy()
    for _ in range(4):
        backward_temporal_segments(accumulated, batch, 2, accum_divisor=4)
    torch.testing.assert_close(accumulated.weight.grad, reference.weight.grad)


class TinyTTTPolicy(torch.nn.Module):
    """Real DiT/TTT modules with a small CPU-only observation interface."""
    def __init__(self, architecture, memory_source):
        super().__init__()
        self.source = memory_source
        self.model = DiT(
            input_embedding_dim=8, num_attention_heads=2, attention_head_dim=4,
            cross_attention_dim=8, output_dim=8, num_layers=4, dropout=0,
            norm_type="ada_norm", interleave_self_attention=True,
            ttt_enabled=True, ttt_architecture=architecture,
            ttt_memory_source=memory_source,
            ttt_layer_indices=(1, 3) if architecture == "wrapper" else (0, 1, 2, 3),
            ttt_memory_dim=8, ttt_memory_hidden_dim=12, ttt_tbptt_step_size=2,
        )
        self.model.requires_grad_(False)
        for block in self.model.transformer_blocks:
            ttt = block.ttt_wrapper if block.ttt_wrapper is not None else block.ttt_layer
            if ttt is not None:
                ttt.requires_grad_(True)
        self.calls = []

    def forward(self, actions, pixel_values, time_valid_mask, action_valid_mask,
                prev_fast_weights=None, return_fast_weights=False, **kwargs):
        if prev_fast_weights is not None:
            for state in prev_fast_weights:
                assert isinstance(state, TTTFastWeightState)
                assert all(value.grad_fn is None for value in state.fast_weights.values())
        self.calls.append(prev_fast_weights)
        batch, time, horizon, dim = actions.shape
        memory = pixel_values["vision"].reshape(batch * time, 3, dim)
        pred, state = self.model(
            actions.reshape(batch * time, horizon, dim), encoder_hidden_states=memory,
            timestep=torch.zeros(batch * time, dtype=torch.long), time_steps=time,
            ttt_memory_tokens=memory if self.source == "jepa" else None,
            ttt_action_token_count=horizon, ttt_memory_source=self.source,
            time_valid_mask=time_valid_mask, prev_fast_weights=prev_fast_weights,
            return_fast_weights=True,
        )
        loss = masked_action_mse(pred.reshape_as(actions), actions, action_valid_mask, time_valid_mask)
        return {"loss": loss, "next_fast_weights": state}


@pytest.mark.parametrize("architecture", ["inline", "wrapper"])
@pytest.mark.parametrize("memory_source", ["jepa", "action_tokens"])
def test_four_routes_carry_detached_state_reset_and_keep_backbone_frozen(architecture, memory_source):
    torch.manual_seed(7)
    policy = TinyTTTPolicy(architecture, memory_source)
    batch = make_batch()
    frozen_before = {name: value.detach().clone() for name, value in policy.named_parameters() if not value.requires_grad}
    result = backward_temporal_segments(policy, batch, 2, autocast_context=nullcontext)
    assert torch.isfinite(result["loss"])
    assert len(policy.calls) == 4
    assert policy.calls[0] is None
    for index, states in enumerate(policy.calls[1:], 1):
        for state in states:
            torch.testing.assert_close(state.step, torch.full((2,), 2 * index, dtype=torch.long))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters() if p.requires_grad)
    assert all(p.grad is None for p in policy.parameters() if not p.requires_grad)
    torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4).step()
    for name, value in policy.named_parameters():
        if name in frozen_before:
            torch.testing.assert_close(value, frozen_before[name], rtol=0, atol=0)
    policy.zero_grad()
    backward_temporal_segments(policy, batch, 2)
    assert policy.calls[4] is None  # A new (possibly different-episode) sequence.


def test_detach_keeps_values_step_and_namedtuple_type():
    tensor = torch.randn(2, 3, requires_grad=True) * 2
    original = (TTTFastWeightState({"weight": tensor}, torch.tensor([8, 8])),)
    detached = detach_fast_weights(original)
    assert isinstance(detached[0], TTTFastWeightState)
    assert detached[0].fast_weights["weight"].grad_fn is None
    torch.testing.assert_close(detached[0].fast_weights["weight"], tensor)
    torch.testing.assert_close(detached[0].step, original[0].step)


@pytest.mark.parametrize("gb,steps,accum", [(1, 10000, 1), (4, 2500, 4)])
def test_round2_budget_and_paths(gb, steps, accum):
    env = {key: value for key, value in os.environ.items() if key not in (
        "MAX_STEPS", "RUN_ID", "TTT_CONTEXT_LENGTH", "TTT_TBPTT_STEP_SIZE", "PER_DEVICE_BATCH_SIZE",
        "OBSERVATION_BUDGET", "SAVE_INTERVAL", "TTT_ARCHITECTURE",
    )}
    env.update(REPO_ROOT=str(ROOT), TTT_MEMORY_TAG="jepa-memory", GLOBAL_BATCH_SIZE=str(gb))
    output = subprocess.check_output(["bash", "-c", 'source "$REPO_ROOT/vla-scripts/ttt_round2_config.sh"'], env=env, text=True)
    assert f"max_steps={steps}" in output
    assert f"grad_accumulation={accum}" in output
    assert "valid_observation_budget=320000" in output
    assert "/checkpoints/2nd_round" in output


def test_full_context_windows_never_cross_episode_and_preserve_tail_mask():
    import tensorflow as tf
    from prismatic.vla.datasets.rlds.traj_transforms import chunk_act_obs

    def trajectory(offset):
        values = tf.range(7, dtype=tf.float32) + offset
        return dict(action=values[:, None], observation={"proprio": values[:, None]},
                    task={"instruction": tf.fill([7], "task")}, dataset_name=tf.fill([7], "libero"),
                    absolute_action_mask=tf.zeros([7, 1], tf.bool))

    # Chunking is per episode, before interleaving/shuffle in the production pipeline.
    chunks = [chunk_act_obs(trajectory(offset), 4, 2) for offset in (0, 100)]
    for item, offset in zip(chunks, (0, 100)):
        keep = tf.reduce_all(item["observation"]["pad_mask"], axis=-1)
        windows = tf.boolean_mask(item["observation"]["proprio"], keep).numpy()
        assert windows.shape == (4, 4, 1)
        assert windows.min() >= offset and windows.max() <= offset + 6
        assert (windows[:, 1:] - windows[:, :-1] == 1).all()
        assert item["action_valid_mask"][-1].numpy().tolist() == [True, True, True, True, False, False]
