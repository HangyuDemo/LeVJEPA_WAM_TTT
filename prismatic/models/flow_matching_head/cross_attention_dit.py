# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import SinusoidalPositionalEmbedding, TimestepEmbedding, Timesteps
from torch import nn
from torch.func import functional_call, grad, vmap


# The state is deliberately explicit rather than stored on the module.  A robot
# rollout can therefore keep one state per environment and reset it at episode
# boundaries without relying on mutable module state.
TTTFastWeightState = namedtuple("TTTFastWeightState", ("fast_weights", "step"))


class CompatRMSNorm(nn.Module):
    """RMSNorm fallback for PyTorch versions before ``nn.RMSNorm``."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(dtype=input_dtype)


def _add_parameter_trees(left, right):
    if left is None:
        return right
    if right is None:
        return left
    return {name: left[name] + right[name] for name in left}


def _detach_parameter_tree(tree):
    if tree is None:
        return None
    return {name: value.detach() for name, value in tree.items()}


class TemporalTTTLayer(nn.Module):
    """RoboTTT-style fast-weight memory operating over environment time.

    The surrounding DiT is evaluated on a flattened ``[batch * time, tokens,
    dim]`` tensor.  This layer restores the time axis, updates a small MLP's
    *fast* parameters one robot timestep at a time, and returns a gated memory
    residual.  The MLP's ordinary parameters, Q/K/V projections, learning rate,
    and residual gate are slow (trained) parameters; only ``TTTFastWeightState``
    is carried between observations at deployment.
    """

    def __init__(
        self,
        dim: int,
        memory_hidden_dim: int,
        *,
        memory_dim: Optional[int] = None,
        require_memory_tokens: bool = False,
        learned_forget: bool = True,
        rope_theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.memory_dim = memory_dim or dim
        self.require_memory_tokens = require_memory_tokens
        if self.memory_dim % 2:
            raise ValueError("TemporalTTTLayer requires an even JEPA memory dimension for RoPE.")
        self.rope_theta = rope_theta
        # Keep the original QKV projection for the DiT query path.  Keys and
        # values are supplied separately by the WAM prediction head below.
        self.to_qkv = nn.Sequential(CompatRMSNorm(dim), nn.Linear(dim, 3 * dim))
        self.query_to_memory = nn.Linear(dim, self.memory_dim)
        self.memory_norm = CompatRMSNorm(self.memory_dim)
        self.memory = nn.Sequential(
            nn.Linear(self.memory_dim, memory_hidden_dim),
            nn.GELU(),
            nn.Linear(memory_hidden_dim, self.memory_dim),
        )
        self.memory_to_output = nn.Linear(self.memory_dim, dim)
        # Match robo_ttt's default inner-loop scale: the effective initial rate
        # is softplus(1e-2), rather than an extra fixed multiplier.
        self.learnable_lr = nn.Parameter(torch.tensor(1e-2))
        # Match TTTWrapper's small random residual gate initialization.
        self.memory_out_layerscale = nn.Parameter(torch.randn(dim) * 1e-4)
        self.learned_forget = learned_forget
        if learned_forget:
            self.to_forget_gate = nn.Sequential(
                CompatRMSNorm(dim),
                nn.Linear(dim, 2 * dim),
                nn.SiLU(),
                nn.Linear(2 * dim, 1),
                nn.Sigmoid(),
            )

    @staticmethod
    def _normalize_state(state, batch: int, device: torch.device):
        if state is None:
            return TTTFastWeightState(None, torch.zeros(batch, dtype=torch.long, device=device))
        if not isinstance(state, TTTFastWeightState):
            raise TypeError("TTT fast weights must be a TTTFastWeightState or None.")
        if isinstance(state.step, int):
            step = torch.full((batch,), state.step, dtype=torch.long, device=device)
        else:
            step = state.step.to(device=device, dtype=torch.long)
        if tuple(step.shape) != (batch,):
            raise ValueError(f"TTT state step must have shape {(batch,)}, got {tuple(step.shape)}.")
        if state.fast_weights is not None:
            first_weight = next(iter(state.fast_weights.values()))
            if first_weight.shape[0] != batch:
                raise ValueError(
                    f"TTT state batch size {first_weight.shape[0]} does not match input batch size {batch}."
                )
        return TTTFastWeightState(state.fast_weights, step)

    def _apply_rope(self, tensor: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Apply an absolute environment-timestep RoPE to Q/K tokens."""
        half_dim = tensor.shape[-1] // 2
        inv_freq = torch.arange(half_dim, device=tensor.device, dtype=torch.float32)
        inv_freq = self.rope_theta ** (-inv_freq / half_dim)
        phase = step.to(device=tensor.device, dtype=torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
        cos = phase.cos().to(dtype=tensor.dtype).unsqueeze(1)
        sin = phase.sin().to(dtype=tensor.dtype).unsqueeze(1)
        first, second = tensor[..., :half_dim], tensor[..., half_dim:]
        return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)

    def _batched_base_parameters(self, batch: int) -> dict[str, torch.Tensor]:
        return {
            name: parameter.unsqueeze(0).expand(batch, *parameter.shape)
            for name, parameter in self.memory.named_parameters()
        }

    def _memory_grads(
        self,
        memory_params: dict[str, torch.Tensor],
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        def loss_fn(params, one_key, one_value):
            prediction = functional_call(self.memory, params, (one_key,))
            return F.mse_loss(prediction, one_value)

        # ``torch.func.grad`` preserves the outer graph during training, so the
        # slow parameters learn an initialization and an update rule.  During
        # deployment the caller is under ``torch.no_grad()``; temporarily
        # enabling grad is still required to compute the inner update, but the
        # resulting fast weights must be detached or every observation would
        # retain a graph through the whole rollout.
        grad_fn = vmap(grad(loss_fn), in_dims=(0, 0, 0))
        if torch.is_grad_enabled():
            return grad_fn(memory_params, keys, values)
        with torch.enable_grad():
            gradients = grad_fn(memory_params, keys, values)
        return {name: gradient.detach() for name, gradient in gradients.items()}

    def _retrieve(self, memory_params: dict[str, torch.Tensor], queries: torch.Tensor) -> torch.Tensor:
        def retrieve_fn(params, one_query):
            return functional_call(self.memory, params, (one_query,))

        return vmap(retrieve_fn, in_dims=(0, 0))(memory_params, queries)

    def _step(
        self,
        tokens: torch.Tensor,
        state: TTTFastWeightState,
        *,
        memory_tokens: Optional[torch.Tensor],
        kv_tokens: Optional[torch.Tensor],
        update_memory: bool,
        valid_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, TTTFastWeightState]:
        batch = tokens.shape[0]
        q_hidden, k_hidden, v_hidden = self.to_qkv(tokens).chunk(3, dim=-1)
        if memory_tokens is not None and kv_tokens is not None:
            raise ValueError("TTT accepts either explicit memory_tokens or implicit kv_tokens, not both.")
        if kv_tokens is not None:
            if kv_tokens.ndim != 3 or kv_tokens.shape[0] != batch or kv_tokens.shape[-1] != self.dim:
                raise ValueError(
                    f"TTT implicit kv_tokens must have shape [B, N_kv, {self.dim}]."
                )
            if self.memory_dim != self.dim:
                raise ValueError("Implicit DiT K/V requires memory_dim to equal the DiT token dimension.")
            _, k, v = self.to_qkv(kv_tokens).chunk(3, dim=-1)
            q = q_hidden
        elif memory_tokens is None:
            # Backward-compatible path for direct users of TemporalTTTLayer.
            # Production JEPA-WAM calls always provide predicted V-JEPA tokens.
            if self.require_memory_tokens:
                raise ValueError("This TTT layer requires WAM-predicted JEPA memory_tokens.")
            if self.memory_dim != self.dim:
                raise ValueError("A JEPA memory tensor is required when memory_dim differs from DiT dim.")
            q, k, v = q_hidden, k_hidden, v_hidden
        else:
            if memory_tokens.ndim != 3 or memory_tokens.shape[0] != batch:
                raise ValueError(
                    "TTT JEPA memory must have shape [B, N_memory, D_jepa] for each timestep."
                )
            if memory_tokens.shape[-1] != self.memory_dim:
                raise ValueError(
                    f"TTT JEPA memory dim {memory_tokens.shape[-1]} does not match configured "
                    f"memory_dim={self.memory_dim}."
                )
            q = self.query_to_memory(q_hidden)
            k = self.memory_norm(memory_tokens)
            # Values remain in the predicted JEPA coordinate system.  The
            # fast-weight MLP therefore learns from WAM representations, not
            # RGB frames or DiT hidden states.
            v = memory_tokens
        q, k = self._apply_rope(q, state.step), self._apply_rope(k, state.step)

        base_params = self._batched_base_parameters(batch)
        memory_params = _add_parameter_trees(state.fast_weights, base_params)

        next_state = state
        if update_memory:
            grads = self._memory_grads(memory_params, k, v)
            learning_rate = F.softplus(self.learnable_lr)

            if self.learned_forget:
                forget = self.to_forget_gate(tokens.mean(dim=1)).squeeze(-1)
            else:
                forget = torch.ones(batch, dtype=tokens.dtype, device=tokens.device)
            if valid_mask is not None:
                forget = forget * valid_mask.to(device=tokens.device, dtype=forget.dtype)

            deltas = {
                name: -learning_rate * gradient * forget.view(batch, *([1] * (gradient.ndim - 1)))
                for name, gradient in grads.items()
            }
            next_fast_weights = _add_parameter_trees(state.fast_weights, deltas)
            step_increment = (
                torch.ones_like(state.step)
                if valid_mask is None
                else valid_mask.to(device=state.step.device, dtype=state.step.dtype)
            )
            next_state = TTTFastWeightState(next_fast_weights, state.step + step_increment)
            memory_params = _add_parameter_trees(next_fast_weights, base_params)

        memory_out = self.memory_to_output(self._retrieve(memory_params, q))
        return memory_out, next_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        memory_tokens: Optional[torch.Tensor] = None,
        kv_tokens: Optional[torch.Tensor] = None,
        time_steps: int,
        prev_fast_weights=None,
        update_memory: bool = True,
        time_valid_mask: Optional[torch.Tensor] = None,
        tbptt_step_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, TTTFastWeightState]:
        if hidden_states.ndim != 3:
            raise ValueError(f"TTT expects [B*T, N, D] hidden states, got {tuple(hidden_states.shape)}.")
        if time_steps < 1 or hidden_states.shape[0] % time_steps:
            raise ValueError("The flattened DiT batch must be divisible by `time_steps`.")

        batch = hidden_states.shape[0] // time_steps
        if memory_tokens is not None and kv_tokens is not None:
            raise ValueError("TTT accepts either explicit memory_tokens or implicit kv_tokens, not both.")
        if memory_tokens is not None:
            if memory_tokens.ndim != 3 or memory_tokens.shape[0] != hidden_states.shape[0]:
                raise ValueError(
                    "Flattened TTT JEPA memory must have shape [B*T, N_memory, D_jepa]."
                )
            flat_memory_tokens = memory_tokens
        else:
            flat_memory_tokens = None
        if kv_tokens is not None:
            if kv_tokens.ndim != 3 or kv_tokens.shape[0] != hidden_states.shape[0]:
                raise ValueError("Flattened implicit kv_tokens must have shape [B*T, N_kv, D_dit].")
            flat_kv_tokens = kv_tokens
        else:
            flat_kv_tokens = None
        if time_valid_mask is not None and tuple(time_valid_mask.shape) != (batch, time_steps):
            raise ValueError(
                f"time_valid_mask must have shape {(batch, time_steps)}, got {tuple(time_valid_mask.shape)}."
            )

        chunks = hidden_states.reshape(batch, time_steps, *hidden_states.shape[1:])
        memory_chunks = (
            None
            if flat_memory_tokens is None
            else flat_memory_tokens.reshape(batch, time_steps, *flat_memory_tokens.shape[1:])
        )
        kv_chunks = (
            None
            if flat_kv_tokens is None
            else flat_kv_tokens.reshape(batch, time_steps, *flat_kv_tokens.shape[1:])
        )
        state = self._normalize_state(prev_fast_weights, batch, hidden_states.device)
        outputs = []
        for index in range(time_steps):
            valid = None if time_valid_mask is None else time_valid_mask[:, index]
            memory_out, state = self._step(
                chunks[:, index],
                state,
                memory_tokens=None if memory_chunks is None else memory_chunks[:, index],
                kv_tokens=None if kv_chunks is None else kv_chunks[:, index],
                update_memory=update_memory,
                valid_mask=valid,
            )
            outputs.append(memory_out)
            if update_memory and tbptt_step_size is not None and (index + 1) % tbptt_step_size == 0:
                state = TTTFastWeightState(_detach_parameter_tree(state.fast_weights), state.step)

        memory_out = torch.stack(outputs, dim=1).reshape_as(hidden_states)
        return memory_out * self.memory_out_layerscale.tanh(), state


class RoboTTTStyleActionWrapper(nn.Module):
    """RoboTTT-style adapter attached to one selected DiT block.

    The memory sees action tokens and, when enabled, the state/register prefix.
    Unified scope returns the residual to state, registers and actions for both
    memory sources. Legacy scope retains the older source-specific behavior. The
    JEPA future-token middle stream is excluded from the direct residual. The
    surrounding DiT block remains the pretrained policy computation; this
    module adds a small, near-zero-gated recurrent residual exactly like the
    standalone ``robo_ttt.TTTWrapper``.  One independent fast-weight state is
    carried for every selected wrapper, rather than allocating a state slot
    for every DiT block.
    """

    def __init__(
        self,
        dim: int,
        memory_hidden_dim: int,
        *,
        memory_dim: Optional[int],
        memory_source: str,
        learned_forget: bool = True,
        prefix_token_count: int = 0,
        token_scope: str = "legacy",
        action_kv_scope: str = "query_tokens",
    ) -> None:
        super().__init__()
        if memory_source not in {"jepa", "action_tokens"}:
            raise ValueError(f"Unsupported RoboTTT memory source: {memory_source!r}.")
        self.memory_source = memory_source
        if prefix_token_count < 0:
            raise ValueError("prefix_token_count must be non-negative.")
        self.prefix_token_count = prefix_token_count
        if token_scope not in {"legacy", "state_register_action"}:
            raise ValueError(f"Unsupported TTT token scope: {token_scope!r}.")
        if token_scope == "state_register_action" and prefix_token_count < 2:
            raise ValueError("Unified TTT scope requires a state/register prefix.")
        self.token_scope = token_scope
        if action_kv_scope not in {"query_tokens", "full_dit_tokens"}:
            raise ValueError(f"Unsupported action K/V scope: {action_kv_scope!r}.")
        self.action_kv_scope = action_kv_scope
        self.memory = TemporalTTTLayer(
            dim,
            memory_hidden_dim,
            memory_dim=memory_dim,
            require_memory_tokens=memory_source == "jepa",
            learned_forget=learned_forget,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        action_token_count: int,
        memory_tokens: Optional[torch.Tensor],
        time_steps: int,
        prev_fast_weights=None,
        update_fast_weights: bool = True,
        time_valid_mask: Optional[torch.Tensor] = None,
        tbptt_step_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, object]:
        if action_token_count < 1 or action_token_count > hidden_states.shape[1]:
            raise ValueError(
                "RoboTTT action wrapper requires action_token_count within the DiT token sequence."
            )

        prefix = self.prefix_token_count
        if prefix + action_token_count > hidden_states.shape[1]:
            raise ValueError("TTT prefix and action tokens must not overlap.")
        # Layout: [state, registers, JEPA future tokens, actions]. Select the
        # state/register prefix plus action suffix, excluding JEPA future tokens.
        selected_tokens = torch.cat(
            (hidden_states[:, :prefix], hidden_states[:, -action_token_count:]), dim=1
        )
        memory_out, next_fast_weights = self.memory(
            selected_tokens,
            memory_tokens=None if self.memory_source == "action_tokens" else memory_tokens,
            kv_tokens=(
                hidden_states
                if self.memory_source == "action_tokens" and self.action_kv_scope == "full_dit_tokens"
                else None
            ),
            time_steps=time_steps,
            prev_fast_weights=prev_fast_weights,
            update_memory=update_fast_weights,
            time_valid_mask=time_valid_mask,
            tbptt_step_size=tbptt_step_size,
        )
        if self.memory_source == "jepa" and self.token_scope == "legacy":
            # JEPA representations remain the memory K/V source, while their
            # readout is applied only to the action-token suffix.
            return torch.cat(
                (
                    hidden_states[:, :-action_token_count],
                    hidden_states[:, -action_token_count:]
                    + memory_out[:, -action_token_count:],
                ),
                dim=1,
            ), next_fast_weights
        return torch.cat(
            (selected_tokens[:, :prefix] + memory_out[:, :prefix],
             hidden_states[:, prefix:-action_token_count],
             selected_tokens[:, prefix:] + memory_out[:, prefix:]),
            dim=1,
        ), next_fast_weights


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        ttt_layer: Optional[TemporalTTTLayer] = None,
        ttt_wrapper: Optional[RoboTTTStyleActionWrapper] = None,
        ttt_token_scope: str = "legacy",
        ttt_prefix_token_count: int = 0,
        ttt_action_kv_scope: str = "query_tokens",
    ):
        super().__init__()
        self.norm_type = norm_type
        self.ttt_token_scope = ttt_token_scope
        self.ttt_prefix_token_count = ttt_prefix_token_count
        self.ttt_action_kv_scope = ttt_action_kv_scope

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None
        self.ttt_layer = ttt_layer
        self.ttt_wrapper = ttt_wrapper

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
        time_steps: int = 1,
        ttt_memory_tokens: Optional[torch.Tensor] = None,
        prev_fast_weights=None,
        update_fast_weights: bool = True,
        time_valid_mask: Optional[torch.Tensor] = None,
        tbptt_step_size: Optional[int] = None,
        ttt_memory_source: str = "jepa",
        ttt_action_token_count: Optional[int] = None,
    ) -> tuple[torch.Tensor, object]:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # RoboTTT placement: attention residual -> TTT -> FFN.  Attention only
        # mixes tokens within an environment timestep; this layer mixes them
        # across environment time through its explicit fast-weight state.
        next_fast_weights = prev_fast_weights
        if self.ttt_wrapper is not None:
            if ttt_action_token_count is None:
                raise ValueError("RoboTTT action wrapper requires ttt_action_token_count.")
            hidden_states, next_fast_weights = self.ttt_wrapper(
                hidden_states,
                action_token_count=ttt_action_token_count,
                memory_tokens=ttt_memory_tokens,
                time_steps=time_steps,
                prev_fast_weights=prev_fast_weights,
                update_fast_weights=update_fast_weights,
                time_valid_mask=time_valid_mask,
                tbptt_step_size=tbptt_step_size,
            )
        elif self.ttt_layer is not None:
            # Unified scope selects state/registers plus actions, never the
            # JEPA future-token middle stream. Preserve legacy checkpoint scope.
            if ttt_action_token_count is None or not 1 <= ttt_action_token_count <= hidden_states.shape[1]:
                raise ValueError("Inline TTT requires ttt_action_token_count within the DiT token sequence.")
            ttt_tokens = hidden_states
            prefix = self.ttt_prefix_token_count
            if self.ttt_token_scope == "state_register_action":
                if prefix < 1 or prefix + ttt_action_token_count > hidden_states.shape[1]:
                    raise ValueError("TTT requires a non-overlapping state/register prefix and action suffix.")
                ttt_tokens = torch.cat(
                    (hidden_states[:, :prefix], hidden_states[:, -ttt_action_token_count:]), dim=1
                )
            elif ttt_memory_source == "action_tokens":
                ttt_tokens = hidden_states[:, -ttt_action_token_count:]
            ttt_memory, next_fast_weights = self.ttt_layer(
                ttt_tokens,
                memory_tokens=None if ttt_memory_source == "action_tokens" else ttt_memory_tokens,
                kv_tokens=(
                    hidden_states
                    if ttt_memory_source == "action_tokens"
                    and self.ttt_action_kv_scope == "full_dit_tokens"
                    else None
                ),
                time_steps=time_steps,
                prev_fast_weights=prev_fast_weights,
                update_memory=update_fast_weights,
                time_valid_mask=time_valid_mask,
                tbptt_step_size=tbptt_step_size,
            )
            if self.ttt_token_scope == "state_register_action":
                hidden_states = torch.cat(
                    (
                        hidden_states[:, :prefix] + ttt_memory[:, :prefix],
                        hidden_states[:, prefix:-ttt_action_token_count],
                        hidden_states[:, -ttt_action_token_count:] + ttt_memory[:, prefix:],
                    ), dim=1,
                )
            else:
                hidden_states = torch.cat(
                    (
                        hidden_states[:, :-ttt_action_token_count],
                        hidden_states[:, -ttt_action_token_count:]
                        + ttt_memory[:, -ttt_action_token_count:],
                    ), dim=1,
                )

        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states, next_fast_weights


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    DEFAULT_TTT_LAYER_INDICES = (3, 7, 11, 15)

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
        ttt_enabled: bool = False,
        ttt_memory_source: str = "jepa",
        ttt_architecture: str = "wrapper",
        ttt_wrapper_prefix_token_count: int = 0,
        ttt_token_scope: str = "legacy",
        ttt_action_kv_scope: str = "query_tokens",
        ttt_layer_indices: Optional[Sequence[int]] = None,
        ttt_memory_hidden_dim: Optional[int] = None,
        ttt_memory_dim: Optional[int] = None,
        ttt_tbptt_step_size: Optional[int] = None,
        ttt_learned_forget: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False
        self.ttt_enabled = ttt_enabled
        if ttt_token_scope not in {"legacy", "state_register_action"}:
            raise ValueError(f"Unsupported TTT token scope: {ttt_token_scope!r}.")
        if ttt_enabled and ttt_token_scope == "state_register_action" and ttt_wrapper_prefix_token_count < 2:
            raise ValueError("Unified TTT scope requires state and register tokens in the prefix.")
        if ttt_action_kv_scope not in {"query_tokens", "full_dit_tokens"}:
            raise ValueError(f"Unsupported action K/V scope: {ttt_action_kv_scope!r}.")
        if ttt_architecture not in {"inline", "wrapper"}:
            raise ValueError(
                "ttt_architecture must be either 'inline' or 'wrapper', "
                f"got {ttt_architecture!r}."
            )
        self.ttt_architecture = ttt_architecture
        if ttt_memory_source not in {"jepa", "action_tokens"}:
            raise ValueError(
                "ttt_memory_source must be either 'jepa' or 'action_tokens', "
                f"got {ttt_memory_source!r}."
            )
        self.ttt_memory_source = ttt_memory_source
        self.ttt_tbptt_step_size = ttt_tbptt_step_size

        if ttt_layer_indices is None or len(tuple(ttt_layer_indices)) == 0:
            if not ttt_enabled:
                ttt_layer_indices = ()
            elif ttt_architecture == "inline":
                ttt_layer_indices = tuple(range(num_layers))
            else:
                ttt_layer_indices = tuple(index for index in self.DEFAULT_TTT_LAYER_INDICES if index < num_layers)
        self.ttt_layer_indices = tuple(ttt_layer_indices)
        if any(index < 0 or index >= num_layers for index in self.ttt_layer_indices):
            raise ValueError(f"Invalid TTT layer indices {self.ttt_layer_indices} for a {num_layers}-layer DiT.")
        # Ignore an accidental layer selection when TTT is disabled.  This
        # keeps the feature flag authoritative and prevents constructing
        # memory modules that can never receive a valid JEPA memory input.
        ttt_layer_indices_set = set(self.ttt_layer_indices) if ttt_enabled else set()
        if ttt_enabled and not ttt_layer_indices_set:
            raise ValueError("TTT is enabled but no DiT layers were selected.")
        memory_dim = ttt_memory_dim or self.inner_dim
        memory_hidden_dim = ttt_memory_hidden_dim or (2 * memory_dim)

        compute_dtype = getattr(self.config, "compute_dtype", torch.float32)
        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim, compute_dtype=compute_dtype)

        all_blocks = []
        for idx in range(self.config.num_layers):
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None
            ttt_layer = None
            ttt_wrapper = None
            if idx in ttt_layer_indices_set:
                if ttt_architecture == "wrapper":
                    ttt_wrapper = RoboTTTStyleActionWrapper(
                        self.inner_dim,
                        memory_hidden_dim,
                        memory_dim=memory_dim,
                        memory_source=ttt_memory_source,
                        learned_forget=ttt_learned_forget,
                        prefix_token_count=ttt_wrapper_prefix_token_count,
                        token_scope=ttt_token_scope,
                        action_kv_scope=ttt_action_kv_scope,
                    )
                else:
                    ttt_layer = TemporalTTTLayer(
                        self.inner_dim,
                        memory_hidden_dim,
                        memory_dim=memory_dim,
                        require_memory_tokens=ttt_memory_source == "jepa",
                        learned_forget=ttt_learned_forget,
                    )
            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                    ttt_token_scope=ttt_token_scope,
                    ttt_prefix_token_count=ttt_wrapper_prefix_token_count,
                    ttt_action_kv_scope=ttt_action_kv_scope,
                    ttt_layer=ttt_layer,
                    ttt_wrapper=ttt_wrapper,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)

        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        return_all_hidden_states: bool = False,
        time_steps: int = 1,
        ttt_memory_tokens: Optional[torch.Tensor] = None,
        prev_fast_weights=None,
        return_fast_weights: bool = False,
        update_fast_weights: bool = True,
        time_valid_mask: Optional[torch.Tensor] = None,
        tbptt_step_size: Optional[int] = None,
        ttt_memory_source: Optional[str] = None,
        ttt_action_token_count: Optional[int] = None,
    ):
        if time_steps < 1 or hidden_states.shape[0] % time_steps:
            raise ValueError("The DiT batch dimension must be divisible by `time_steps`.")
        ttt_memory_source = ttt_memory_source or self.ttt_memory_source
        if time_valid_mask is not None:
            expected_shape = (hidden_states.shape[0] // time_steps, time_steps)
            if tuple(time_valid_mask.shape) != expected_shape:
                raise ValueError(f"Expected time_valid_mask shape {expected_shape}, got {tuple(time_valid_mask.shape)}.")
        if ttt_memory_tokens is not None:
            if ttt_memory_tokens.ndim != 3 or ttt_memory_tokens.shape[0] != hidden_states.shape[0]:
                raise ValueError("ttt_memory_tokens must have shape [B*T, N_memory, D_jepa].")
        selected_state_indices = {layer_index: state_index for state_index, layer_index in enumerate(self.ttt_layer_indices)}
        state_slots = len(self.ttt_layer_indices) if self.ttt_architecture == "wrapper" else len(self.transformer_blocks)
        if prev_fast_weights is None:
            prev_fast_weights = (None,) * state_slots
        if len(prev_fast_weights) != state_slots:
            raise ValueError(
                "Unexpected number of TTT fast-weight states: "
                f"expected {state_slots}, got {len(prev_fast_weights)}."
            )

        temb = self.timestep_encoder(timestep)

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        next_fast_weights = []
        for idx, block in enumerate(self.transformer_blocks):
            prev_state = (
                prev_fast_weights[selected_state_indices[idx]]
                if self.ttt_architecture == "wrapper" and idx in selected_state_indices
                else prev_fast_weights[idx] if self.ttt_architecture == "inline" else None
            )
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states, next_state = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                    time_steps=time_steps,
                    ttt_memory_tokens=ttt_memory_tokens,
                    prev_fast_weights=prev_state,
                    update_fast_weights=update_fast_weights,
                    time_valid_mask=time_valid_mask,
                    tbptt_step_size=tbptt_step_size or self.ttt_tbptt_step_size,
                    ttt_memory_source=ttt_memory_source,
                    ttt_action_token_count=ttt_action_token_count,
                )
            else:
                hidden_states, next_state = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    temb=temb,
                    time_steps=time_steps,
                    ttt_memory_tokens=ttt_memory_tokens,
                    prev_fast_weights=prev_state,
                    update_fast_weights=update_fast_weights,
                    time_valid_mask=time_valid_mask,
                    tbptt_step_size=tbptt_step_size or self.ttt_tbptt_step_size,
                    ttt_memory_source=ttt_memory_source,
                    ttt_action_token_count=ttt_action_token_count,
                )
            all_hidden_states.append(hidden_states)
            if self.ttt_architecture == "inline" or idx in selected_state_indices:
                next_fast_weights.append(next_state)

        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        prediction = self.proj_out_2(hidden_states)
        if return_all_hidden_states and return_fast_weights:
            return prediction, all_hidden_states, tuple(next_fast_weights)
        if return_all_hidden_states:
            return prediction, all_hidden_states
        if return_fast_weights:
            return prediction, tuple(next_fast_weights)
        return prediction
