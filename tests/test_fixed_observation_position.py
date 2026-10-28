"""Inference-only TTT position tests for wrapper and inline state layouts."""

import pytest
import torch

from prismatic.models.flow_gr00t_action_head import FlowMatchingActionHead
from prismatic.models.flow_matching_head.cross_attention_dit import TTTFastWeightState
from prismatic.models.vlas.openvla import OpenVLA


def _recording_head(num_steps: int, updates: int, fixed_position: bool):
    head = FlowMatchingActionHead.__new__(FlowMatchingActionHead)
    torch.nn.Module.__init__(head)
    head.action_encoder = torch.nn.Linear(1, 1)
    head._prepare_state = lambda proprio: proprio
    head.action_horizon = 2
    head.action_dim = 1
    head.num_inference_timesteps = num_steps
    head.num_timestep_buckets = 1000
    head.ttt_enabled = True
    head.ttt_inference_updates = updates
    head.ttt_fixed_observation_position = fixed_position
    calls = []

    def fake_velocity(vl_embs, actions, timesteps_tensor, state, *, prev_fast_weights,
                      update_fast_weights, return_fast_weights, **kwargs):
        del vl_embs, timesteps_tensor, state, kwargs
        assert return_fast_weights
        calls.append((
            update_fast_weights,
            tuple(slot.step.clone() for slot in prev_fast_weights),
            tuple(slot.fast_weights["marker"].clone() for slot in prev_fast_weights),
        ))
        if update_fast_weights:
            next_states = tuple(
                TTTFastWeightState(
                    {"marker": slot.fast_weights["marker"] + 1}, slot.step + 1
                )
                for slot in prev_fast_weights
            )
        else:
            next_states = prev_fast_weights
        return torch.zeros_like(actions), next_states

    head._predict_velocity = fake_velocity
    return head, calls


def _states(slots: int):
    return tuple(
        TTTFastWeightState(
            {"marker": torch.full((2, 1), index, dtype=torch.long)},
            torch.tensor([5, 9], dtype=torch.long),
        )
        for index in range(slots)
    )


def _predict(head, states):
    _, next_states = head.predict_action(
        torch.zeros(2, 3, 4), torch.zeros(2, 1),
        prev_fast_weights=states, return_fast_weights=True,
    )
    return next_states


@pytest.mark.parametrize("architecture,slots", [("wrapper", 4), ("inline", 18)])
def test_fixed_position_keeps_updated_weights_but_commits_next_observation_step(architecture, slots):
    head, calls = _recording_head(num_steps=4, updates=1, fixed_position=True)
    head.ttt_architecture = architecture
    initial = _states(slots)

    after_a = _predict(head, initial)
    assert [write for write, _, _ in calls] == [True, False, False, False]
    for denoise_index, (_, steps, weights) in enumerate(calls):
        assert len(steps) == slots
        for slot_index, (step, weight) in enumerate(zip(steps, weights, strict=True)):
            torch.testing.assert_close(step, torch.tensor([5, 9]))
            torch.testing.assert_close(
                weight, torch.full((2, 1), slot_index + (denoise_index > 0), dtype=torch.long)
            )
    for slot_index, state in enumerate(after_a):
        torch.testing.assert_close(state.step, torch.tensor([6, 10]))
        torch.testing.assert_close(state.fast_weights["marker"],
                                   torch.full((2, 1), slot_index + 1, dtype=torch.long))

    calls.clear()
    after_b = _predict(head, after_a)
    assert [write for write, _, _ in calls] == [True, False, False, False]
    for _, steps, _ in calls:
        for step in steps:
            torch.testing.assert_close(step, torch.tensor([6, 10]))
    for state in after_b:
        torch.testing.assert_close(state.step, torch.tensor([7, 11]))


def test_switch_off_retains_original_first_denoise_positions():
    head, calls = _recording_head(num_steps=4, updates=1, fixed_position=False)
    result = _predict(head, _states(4))
    assert [int(steps[0][0]) for _, steps, _ in calls] == [5, 6, 6, 6]
    torch.testing.assert_close(result[0].step, torch.tensor([6, 10]))


def test_all_denoise_is_unchanged_even_if_fixed_position_switch_is_on():
    head, calls = _recording_head(num_steps=4, updates=4, fixed_position=True)
    result = _predict(head, _states(4))
    assert [write for write, _, _ in calls] == [True, True, True, True]
    assert [int(steps[0][0]) for _, steps, _ in calls] == [5, 6, 7, 8]
    torch.testing.assert_close(result[0].step, torch.tensor([9, 13]))


def test_one_denoise_has_same_state_with_or_without_switch():
    initial = _states(4)
    original_head, original_calls = _recording_head(num_steps=1, updates=1, fixed_position=False)
    fixed_head, fixed_calls = _recording_head(num_steps=1, updates=1, fixed_position=True)
    original = _predict(original_head, initial)
    fixed = _predict(fixed_head, initial)
    assert len(original_calls) == len(fixed_calls) == 1
    for old, new in zip(original, fixed, strict=True):
        torch.testing.assert_close(old.step, new.step)
        torch.testing.assert_close(old.fast_weights["marker"], new.fast_weights["marker"])


def test_new_episode_initial_position_is_zero_and_reset_drops_old_memory():
    committed = tuple(
        TTTFastWeightState({"marker": torch.ones(2, 1)}, torch.ones(2, dtype=torch.long))
        for _ in range(4)
    )
    query = FlowMatchingActionHead._read_memory_at_observation_position(
        committed, None, 2, torch.device("cpu")
    )
    for read, saved in zip(query, committed, strict=True):
        torch.testing.assert_close(read.step, torch.zeros(2, dtype=torch.long))
        assert read.fast_weights is saved.fast_weights
    model = OpenVLA.__new__(OpenVLA)
    torch.nn.Module.__init__(model)
    model._ttt_fast_weights = committed
    model.reset_ttt_memory()
    assert model._ttt_fast_weights is None
