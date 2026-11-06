"""The stateless control trains and validates on one observation at a time."""

import torch
from torch import nn

from prismatic.conf.vla import VLAConfig
from prismatic.training.validation import evaluate_action_loss


def test_static_visual_adapter_recipe_uses_one_observation_without_memory_state():
    cfg = VLAConfig(static_visual_adapter=True)
    assert cfg.ttt_static_adapter and cfg.ttt_enabled and cfg.train_ttt_only
    assert cfg.ttt_context_length == 1
    assert cfg.ttt_observation_stride == 1
    assert not cfg.ttt_carry_between_segments
    assert not cfg.ttt_require_full_context
    assert cfg.ttt_tbptt_step_size is None


def test_static_visual_adapter_validation_accepts_single_observations():
    class ActionLoss(nn.Module):
        def forward(self, *, actions, **kwargs):
            assert actions.shape == (1, 2, 1)
            assert "prev_fast_weights" not in kwargs
            assert "time_valid_mask" not in kwargs
            loss = actions.square().mean()
            return {"loss": loss, "loss_action": loss}

    dataset = [
        {"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([True]),
         "pixel_values": torch.zeros(1), "actions": torch.full((2, 1), float(value)),
         "proprio": torch.zeros(1)}
        for value in (2, 4)
    ]
    results = evaluate_action_loss(
        ActionLoss(), {"suite": dataset}, torch.utils.data.default_collate,
        segment_size=None, static_visual_adapter=True,
    )
    assert results["Validation/Loss Action"] == 10.0
    assert results["Validation/Valid Action Tokens"] == 4
