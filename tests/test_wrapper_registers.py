from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prismatic.models.flow_matching_head.cross_attention_dit import RoboTTTStyleActionWrapper, DiT
from prismatic.models.vlms.prismatic import PrismaticVLM


@pytest.mark.parametrize("source", ["jepa", "action_tokens"])
@pytest.mark.parametrize("prefix", [0, 3])
def test_wrapper_exact_selection_and_residual(source, prefix):
    wrapper = RoboTTTStyleActionWrapper(8, 12, memory_dim=8, memory_source=source,
                                        prefix_token_count=prefix)
    x = torch.randn(2, 9, 8)
    external = torch.randn(2, 4, 8)

    class Spy(nn.Module):
        def forward(self, tokens, memory_tokens, **kwargs):
            torch.testing.assert_close(tokens, torch.cat((x[:, :prefix], x[:, -2:]), dim=1))
            assert kwargs.get("kv_tokens") is None
            if source == "jepa":
                assert memory_tokens is external
            else:
                assert memory_tokens is None
            return torch.ones_like(tokens), "state"

    wrapper.memory = Spy()
    out, state = wrapper(x, action_token_count=2, memory_tokens=external, time_steps=1)
    expected = x.clone()
    expected[:, :prefix] += 1
    expected[:, -2:] += 1
    torch.testing.assert_close(out, expected)
    assert state == "state"


def test_wrapper_full_dit_implicit_kv_keeps_selected_queries_and_residual():
    wrapper = RoboTTTStyleActionWrapper(
        8, 12, memory_dim=8, memory_source="action_tokens",
        prefix_token_count=3, token_scope="state_register_action",
        action_kv_scope="full_dit_tokens",
    )
    x = torch.randn(2, 9, 8)

    class Spy(nn.Module):
        def forward(self, tokens, memory_tokens, kv_tokens, **kwargs):
            torch.testing.assert_close(tokens, torch.cat((x[:, :3], x[:, -2:]), dim=1))
            torch.testing.assert_close(kv_tokens, x)
            assert memory_tokens is None
            return torch.ones_like(tokens), "state"

    wrapper.memory = Spy()
    out, state = wrapper(x, action_token_count=2, memory_tokens=None, time_steps=1)
    expected = x.clone()
    expected[:, :3] += 1
    expected[:, -2:] += 1
    torch.testing.assert_close(out, expected)
    assert state == "state"


@pytest.mark.parametrize("source", ["jepa", "action_tokens"])
def test_register_gradient_through_frozen_dit_and_memory(source):
    torch.manual_seed(7)
    dit = DiT(input_embedding_dim=8, num_attention_heads=2, attention_head_dim=4,
              cross_attention_dim=8, output_dim=8, num_layers=4, dropout=0,
              final_dropout=False, norm_type="ada_norm", interleave_self_attention=True,
              ttt_enabled=True, ttt_architecture="wrapper", ttt_layer_indices=(1, 3),
              ttt_memory_source=source, ttt_memory_dim=8, ttt_memory_hidden_dim=12,
              ttt_wrapper_prefix_token_count=3)
    dit.requires_grad_(False)
    for block in dit.transformer_blocks:
        if block.ttt_wrapper is not None:
            block.ttt_wrapper.requires_grad_(True)
    registers = nn.Parameter(torch.randn(1, 2, 8))
    x = torch.cat((torch.randn(2, 1, 8), registers.expand(2, -1, -1),
                   torch.randn(2, 3, 8), torch.randn(2, 2, 8)), dim=1)
    memory = torch.randn(2, 3, 8)
    out, states = dit(x, encoder_hidden_states=memory, timestep=torch.zeros(2, dtype=torch.long),
                      time_steps=2, ttt_memory_source=source, ttt_action_token_count=2,
                      ttt_memory_tokens=memory if source == "jepa" else None,
                      return_fast_weights=True)
    out[:, -2:].square().mean().backward()
    assert registers.grad is not None and torch.isfinite(registers.grad).all()
    assert registers.grad.abs().sum() > 0
    assert len(states) == 2
    assert any(p.grad is not None for n, p in dit.named_parameters() if "ttt_wrapper" in n)
    assert all(p.grad is None for n, p in dit.named_parameters() if "ttt_wrapper" not in n)


@pytest.mark.parametrize("architecture,enabled", [("wrapper", False), ("wrapper", True), ("inline", False)])
def test_head_register_creation_and_training_scope(monkeypatch, architecture, enabled):
    import prismatic.models.flow_gr00t_action_head as head_module

    class StubDiT(nn.Module):
        def __init__(self, **cfg):
            super().__init__()
            self.prefix = cfg["ttt_wrapper_prefix_token_count"]
            block = nn.Module()
            block.ttt_wrapper = nn.Linear(2, 2)
            block.ttt_layer = None
            self.transformer_blocks = nn.ModuleList([block])

    monkeypatch.setattr(head_module, "DiT", StubDiT)
    head = head_module.FlowMatchingActionHead(4, 3, 8, 2, fm_hidden_size=8,
            fm_num_layers=4, ttt_enabled=True, ttt_architecture=architecture,
            ttt_wrapper_register_tokens=enabled)
    assert (head.register_tokens is not None) == (enabled or architecture == "inline")
    assert head.model.prefix == (17 if enabled and architecture == "wrapper" else 0)
    fake = SimpleNamespace(action_head=head, vision_backbone=nn.Linear(2, 2),
                           projector=nn.Linear(2, 2), llm_backbone=nn.Linear(2, 2),
                           visual_token_cosine_head=nn.Linear(2, 2), arch_specifier="test")
    fake.vision_backbone.identifier = "test"
    fake.named_parameters = head.named_parameters
    PrismaticVLM.freeze_for_training(fake, train_qvv_lora=False, train_action_head=False,
                                    train_ttt_only=True, train_visual_token_cosine_head=False)
    for name, param in head.named_parameters():
        expected = "ttt_wrapper" in name or (architecture == "wrapper" and name.startswith("register_tokens."))
        assert param.requires_grad == expected, name
