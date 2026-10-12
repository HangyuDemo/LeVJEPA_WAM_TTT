import torch

from prismatic.models.flow_matching_head.cross_attention_dit import TemporalTTTLayer
from prismatic.util.checkpoint_compat import (
    align_singleton_adamw_moments,
    upgrade_ttt_lr_model_state,
    upgrade_ttt_lr_optimizer_state,
)


def test_legacy_scalar_ttt_weight_loads_into_fsdp_safe_parameter():
    layer = TemporalTTTLayer(dim=8, memory_hidden_dim=16)
    legacy_state = layer.state_dict()
    legacy_state["learnable_lr"] = torch.tensor(0.125)

    assert tuple(layer.learnable_lr.shape) == (1,)
    assert upgrade_ttt_lr_model_state(legacy_state) == 1
    layer.load_state_dict(legacy_state)
    torch.testing.assert_close(layer.learnable_lr.detach(), torch.tensor([0.125]))


def test_legacy_scalar_adamw_moments_can_take_an_update():
    param = torch.nn.Parameter(torch.tensor([0.125]))
    optimizer = torch.optim.AdamW([param], lr=1e-3, foreach=True)
    legacy_state = {
        "state": {
            "layer.learnable_lr": {
                "step": torch.tensor(10.0),
                "exp_avg": torch.tensor(0.01),
                "exp_avg_sq": torch.tensor(0.001),
            }
        }
    }

    assert upgrade_ttt_lr_optimizer_state(legacy_state) == 2
    assert upgrade_ttt_lr_optimizer_state(legacy_state) == 0
    moments = legacy_state["state"]["layer.learnable_lr"]
    assert tuple(moments["exp_avg"].shape) == (1,)
    assert tuple(moments["exp_avg_sq"].shape) == (1,)
    optimizer.load_state_dict(
        {"state": {0: moments}, "param_groups": optimizer.state_dict()["param_groups"]}
    )
    param.grad = torch.ones_like(param)
    optimizer.step()
    assert torch.isfinite(param).all()


def test_local_singleton_moments_are_repaired_without_touching_step():
    param = torch.nn.Parameter(torch.tensor([0.125]))
    optimizer = torch.optim.AdamW([param], lr=1e-3, foreach=True)
    optimizer.state[param] = {
        "step": torch.tensor(10.0),
        "exp_avg": torch.tensor(0.01),
        "exp_avg_sq": torch.tensor(0.001),
    }

    assert align_singleton_adamw_moments(optimizer) == 2
    assert tuple(optimizer.state[param]["step"].shape) == ()
    assert tuple(optimizer.state[param]["exp_avg"].shape) == (1,)
    param.grad = torch.ones_like(param)
    optimizer.step()
