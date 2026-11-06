"""The WE control has identical slow modules but no online state."""

import torch

from prismatic.models.flow_gr00t_action_head import FlowMatchingActionHead
from prismatic.models.flow_matching_head.cross_attention_dit import (
    RoboTTTStyleActionWrapper,
    TemporalTTTLayer,
    TTTFastWeightState,
)


def test_static_adapter_matches_slow_parameter_count_and_uses_current_visual_tokens():
    torch.manual_seed(7)
    full = TemporalTTTLayer(8, 16, memory_dim=8, require_memory_tokens=True)
    static = TemporalTTTLayer(8, 16, memory_dim=8, require_memory_tokens=True, static_adapter=True)
    assert sum(p.numel() for p in full.parameters()) == sum(p.numel() for p in static.parameters())

    with torch.no_grad():
        static.memory_out_layerscale.fill_(1)
    tokens = torch.randn(2, 3, 5, 8, requires_grad=True)
    visual = torch.randn(2, 3, 4, 8)
    out, state = static(tokens.flatten(0, 1), memory_tokens=visual.flatten(0, 1), time_steps=3)
    assert state is None
    out = out.unflatten(0, (2, 3))

    changed, state = static(
        tokens.flatten(0, 1), memory_tokens=(visual + 2).flatten(0, 1), time_steps=3,
        prev_fast_weights=TTTFastWeightState({"unused": torch.ones(2, 1)}, torch.tensor([5, 9])),
    )
    assert state is None
    assert not torch.allclose(out, changed.unflatten(0, (2, 3)))

    order = [2, 0, 1]
    reordered, state = static(
        tokens[:, order].flatten(0, 1), memory_tokens=visual[:, order].flatten(0, 1), time_steps=3
    )
    assert state is None
    torch.testing.assert_close(reordered.unflatten(0, (2, 3)), out[:, order])

    out.square().sum().backward()
    assert tokens.grad is not None
    for name in (
        "query_to_memory.weight", "memory_norm.weight", "memory.0.weight",
        "memory_to_output.weight", "to_forget_gate.3.weight", "learnable_lr", "memory_out_layerscale",
    ):
        parameter = dict(static.named_parameters())[name]
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_static_wrapper_changes_selected_tokens_without_state_or_future_token_residual():
    torch.manual_seed(11)
    wrapper = RoboTTTStyleActionWrapper(
        8, 16, memory_dim=8, memory_source="jepa_current", prefix_token_count=2,
        token_scope="state_register_action", static_adapter=True,
    )
    with torch.no_grad():
        wrapper.memory.memory_out_layerscale.fill_(1)
    tokens = torch.randn(2, 7, 8)
    visual = torch.randn(2, 4, 8)
    output, state = wrapper(tokens, action_token_count=2, memory_tokens=visual, time_steps=2)
    assert state is None
    torch.testing.assert_close(output[:, 2:-2], tokens[:, 2:-2])
    assert not torch.allclose(output[:, :2], tokens[:, :2])
    assert not torch.allclose(output[:, -2:], tokens[:, -2:])


def test_static_inference_never_updates_or_returns_fast_weight_state():
    head = FlowMatchingActionHead.__new__(FlowMatchingActionHead)
    torch.nn.Module.__init__(head)
    head.action_encoder = torch.nn.Linear(1, 1)
    head._prepare_state = lambda proprio: proprio
    head.action_horizon = 2
    head.action_dim = 1
    head.num_inference_timesteps = 4
    head.num_timestep_buckets = 1000
    head.ttt_enabled = True
    head.ttt_static_adapter = True
    head.ttt_inference_updates = 1
    calls = []

    def velocity(vl_embs, actions, timestep, state, *, prev_fast_weights,
                 update_fast_weights, return_fast_weights, **kwargs):
        del vl_embs, timestep, state, kwargs
        assert return_fast_weights
        calls.append((prev_fast_weights, update_fast_weights))
        return torch.zeros_like(actions), (TTTFastWeightState({"unused": torch.ones(1)}, 9),)

    head._predict_velocity = velocity
    old = (TTTFastWeightState({"unused": torch.ones(1)}, 5),)
    for _ in range(2):
        _, next_state = head.predict_action(
            torch.zeros(1, 3, 4), torch.zeros(1, 1), prev_fast_weights=old,
            return_fast_weights=True,
        )
        assert next_state is None
    assert calls == [(None, False)] * 8
