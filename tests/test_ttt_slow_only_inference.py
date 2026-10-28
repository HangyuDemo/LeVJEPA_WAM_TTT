"""The slow-only inference ablation uses the trained TTT MLP without online deltas."""

import pytest
import torch

from prismatic.models.flow_gr00t_action_head import FlowMatchingActionHead
from prismatic.models.flow_matching_head.cross_attention_dit import TemporalTTTLayer, TTTFastWeightState


@pytest.mark.parametrize("slots", [4, 18])
def test_slow_only_discards_all_fast_state_and_never_writes(slots):
    head = FlowMatchingActionHead.__new__(FlowMatchingActionHead)
    torch.nn.Module.__init__(head)
    head.action_encoder = torch.nn.Linear(1, 1)
    head._prepare_state = lambda proprio: proprio
    head.action_horizon = 2
    head.action_dim = 1
    head.num_inference_timesteps = 4
    head.num_timestep_buckets = 1000
    head.ttt_enabled = True
    head.ttt_inference_updates = 1
    head.ttt_inference_slow_only = True
    head.ttt_fixed_observation_position = True

    calls = []

    def fake_velocity(vl_embs, actions, timesteps_tensor, state, *, prev_fast_weights,
                      update_fast_weights, return_fast_weights, **kwargs):
        del vl_embs, timesteps_tensor, state, kwargs
        assert return_fast_weights
        calls.append((prev_fast_weights, update_fast_weights))
        # A real no-update TTT layer returns zero-position states with no deltas;
        # the action head must not propagate those as persistent memory.
        states = tuple(
            TTTFastWeightState(None, torch.zeros(actions.shape[0], dtype=torch.long))
            for _ in range(slots)
        )
        return torch.zeros_like(actions), states

    head._predict_velocity = fake_velocity
    old_states = tuple(
        TTTFastWeightState({"marker": torch.ones(2, 1)}, torch.tensor([5, 9]))
        for _ in range(slots)
    )
    for observation in range(2):
        _, next_states = head.predict_action(
            torch.zeros(2, 3, 4), torch.zeros(2, 1),
            prev_fast_weights=old_states if observation == 0 else None,
            return_fast_weights=True,
        )
        assert next_states is None

    assert calls == [(None, False)] * 8


def test_no_update_ttt_layer_uses_slow_parameters_and_preserves_zero_state():
    torch.manual_seed(0)
    layer = TemporalTTTLayer(8, 16, memory_dim=8, require_memory_tokens=True)
    layer.memory_out_layerscale.data.fill_(1)
    tokens = torch.randn(2, 3, 8)
    visual_tokens = torch.randn(2, 4, 8)

    before, state = layer(
        tokens, time_steps=1, memory_tokens=visual_tokens,
        prev_fast_weights=None, update_memory=False,
    )
    assert state.fast_weights is None
    torch.testing.assert_close(state.step, torch.zeros(2, dtype=torch.long))

    with torch.no_grad():
        layer.memory_to_output.bias.add_(1)
    after, next_state = layer(
        tokens, time_steps=1, memory_tokens=visual_tokens,
        prev_fast_weights=None, update_memory=False,
    )
    torch.testing.assert_close(after - before, torch.full_like(before, torch.tanh(torch.tensor(1.0))))
    assert next_state.fast_weights is None
    torch.testing.assert_close(next_state.step, torch.zeros(2, dtype=torch.long))
