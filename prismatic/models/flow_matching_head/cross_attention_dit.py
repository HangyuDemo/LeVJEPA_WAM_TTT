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
        # slow parameters learn an initialization and an update rule.
        with torch.enable_grad():
            return vmap(grad(loss_fn), in_dims=(0, 0, 0))(memory_params, keys, values)

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
        update_memory: bool,
        valid_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, TTTFastWeightState]:
        batch = tokens.shape[0]
        q_hidden, k_hidden, v_hidden = self.to_qkv(tokens).chunk(3, dim=-1)
        if memory_tokens is None:
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
                forget = forget * valid_mask.to(dtype=forget.dtype)

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
        if memory_tokens is not None:
            if memory_tokens.ndim != 3 or memory_tokens.shape[0] != hidden_states.shape[0]:
                raise ValueError(
                    "Flattened TTT JEPA memory must have shape [B*T, N_memory, D_jepa]."
                )
            flat_memory_tokens = memory_tokens
        else:
            flat_memory_tokens = None
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
        state = self._normalize_state(prev_fast_weights, batch, hidden_states.device)
        outputs = []
        for index in range(time_steps):
            valid = None if time_valid_mask is None else time_valid_mask[:, index]
            memory_out, state = self._step(
                chunks[:, index],
                state,
                memory_tokens=None if memory_chunks is None else memory_chunks[:, index],
                update_memory=update_memory,
                valid_mask=valid,
            )
            outputs.append(memory_out)
            if update_memory and tbptt_step_size is not None and (index + 1) % tbptt_step_size == 0:
                state = TTTFastWeightState(_detach_parameter_tree(state.fast_weights), state.step)

        memory_out = torch.stack(outputs, dim=1).reshape_as(hidden_states)
        return memory_out * self.memory_out_layerscale.tanh(), state


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
    ):
        super().__init__()
        self.norm_type = norm_type

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
        if self.ttt_layer is not None:
            ttt_memory, next_fast_weights = self.ttt_layer(
                hidden_states,
                memory_tokens=ttt_memory_tokens,
                time_steps=time_steps,
                prev_fast_weights=prev_fast_weights,
                update_memory=update_fast_weights,
                time_valid_mask=time_valid_mask,
                tbptt_step_size=tbptt_step_size,
            )
            hidden_states = hidden_states + ttt_memory

        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states, next_fast_weights


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

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
        self.ttt_tbptt_step_size = ttt_tbptt_step_size

        if ttt_layer_indices is None:
            ttt_layer_indices = tuple(range(num_layers)) if ttt_enabled else ()
        self.ttt_layer_indices = tuple(ttt_layer_indices)
        if any(index < 0 or index >= num_layers for index in self.ttt_layer_indices):
            raise ValueError(f"Invalid TTT layer indices {self.ttt_layer_indices} for a {num_layers}-layer DiT.")
        ttt_layer_indices_set = set(self.ttt_layer_indices)
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
            ttt_layer = (
                TemporalTTTLayer(
                    self.inner_dim,
                    memory_hidden_dim,
                    memory_dim=memory_dim,
                    require_memory_tokens=True,
                    learned_forget=ttt_learned_forget,
                )
                if idx in ttt_layer_indices_set
                else None
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
                    ttt_layer=ttt_layer,
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
    ):
        if time_steps < 1 or hidden_states.shape[0] % time_steps:
            raise ValueError("The DiT batch dimension must be divisible by `time_steps`.")
        if time_valid_mask is not None:
            expected_shape = (hidden_states.shape[0] // time_steps, time_steps)
            if tuple(time_valid_mask.shape) != expected_shape:
                raise ValueError(f"Expected time_valid_mask shape {expected_shape}, got {tuple(time_valid_mask.shape)}.")
        if ttt_memory_tokens is not None:
            if ttt_memory_tokens.ndim != 3 or ttt_memory_tokens.shape[0] != hidden_states.shape[0]:
                raise ValueError("ttt_memory_tokens must have shape [B*T, N_memory, D_jepa].")
        if prev_fast_weights is None:
            prev_fast_weights = (None,) * len(self.transformer_blocks)
        if len(prev_fast_weights) != len(self.transformer_blocks):
            raise ValueError("Expected one TTT state slot per DiT transformer block.")

        temb = self.timestep_encoder(timestep)

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        next_fast_weights = []
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states, next_state = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                    time_steps=time_steps,
                    ttt_memory_tokens=ttt_memory_tokens,
                    prev_fast_weights=prev_fast_weights[idx],
                    update_fast_weights=update_fast_weights,
                    time_valid_mask=time_valid_mask,
                    tbptt_step_size=tbptt_step_size or self.ttt_tbptt_step_size,
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
                    prev_fast_weights=prev_fast_weights[idx],
                    update_fast_weights=update_fast_weights,
                    time_valid_mask=time_valid_mask,
                    tbptt_step_size=tbptt_step_size or self.ttt_tbptt_step_size,
                )
            all_hidden_states.append(hidden_states)
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
