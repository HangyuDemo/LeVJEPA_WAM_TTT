from dataclasses import dataclass, field
from typing import Optional, Sequence
from types import SimpleNamespace

import torch
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig

from prismatic.models.flow_matching_head.action_encoder import ActionEncoder
from prismatic.models.flow_matching_head.cross_attention_dit import DiT


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


@dataclass
class FlowMatchingActionHeadConfig(PretrainedConfig):
    add_pos_embed: bool = field(default=True)
    diffusion_model_cfg: dict = field(default=None)
    input_embedding_dim: int = field(default=768)
    hidden_size: int = field(default=1024)
    max_seq_len: int = field(default=1024)
    action_dim: int = field(default=None)
    action_horizon: int = field(default=None)
    noise_beta_alpha: float = field(default=1.5)
    noise_beta_beta: float = field(default=1.0)
    noise_s: float = field(default=0.999)
    num_timestep_buckets: int = field(default=1000)
    num_inference_timesteps: int = field(default=4)
    num_target_vision_tokens: int = field(default=32)
    ttt_enabled: bool = field(default=False)
    ttt_num_register_tokens: int = field(default=16)
    ttt_layer_indices: tuple[int, ...] = field(default_factory=tuple)
    ttt_memory_hidden_dim: int | None = field(default=None)
    ttt_memory_dim: int | None = field(default=None)
    ttt_tbptt_step_size: int | None = field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowMatchingActionHead(nn.Module):
    """
    VLA-JEPA GR00T action head adapted to JEPA-WAM.

    The architecture is copied from VLA-JEPA; the main interface change is that
    it consumes the Qwen visual/task conditioning sequence from Prismatic
    instead of custom `<|embodied_action|>` tokenizer tokens.  The DiT action
    tokens query this sequence through cross-attention.
    """

    def __init__(
        self,
        d_proprio: int,
        d_action: int,
        d_llm: int,
        horizon: int,
        fm_hidden_size: int = 1024,
        fm_num_layers: int = 16,
        fm_num_inference_timesteps: int = 4,
        fm_num_timestep_buckets: int = 1000,
        fm_noise_beta_alpha: float = 1.5,
        fm_noise_beta_beta: float = 1.0,
        fm_noise_s: float = 0.999,
        fm_num_target_vision_tokens: int = 32,
        fm_add_pos_embed: bool = True,
        fm_max_seq_len: int = 1024,
        fm_state_dropout: float = 0.5,
        ttt_enabled: bool = False,
        ttt_num_register_tokens: int = 16,
        ttt_layer_indices: Optional[Sequence[int]] = None,
        ttt_memory_hidden_dim: int | None = None,
        ttt_memory_dim: int | None = None,
        ttt_tbptt_step_size: int | None = None,
    ):
        super().__init__()

        if d_proprio < 1 or d_action < 1 or horizon < 1:
            raise ValueError(
                "d_proprio, d_action, and horizon must be positive; "
                f"got d_proprio={d_proprio}, d_action={d_action}, horizon={horizon}."
            )
        if fm_hidden_size < 1 or fm_num_layers < 1:
            raise ValueError(
                "fm_hidden_size and fm_num_layers must be positive; "
                f"got hidden_size={fm_hidden_size}, num_layers={fm_num_layers}."
            )
        if fm_num_inference_timesteps < 1:
            raise ValueError("fm_num_inference_timesteps must be positive.")
        if fm_num_timestep_buckets < 1:
            raise ValueError("fm_num_timestep_buckets must be positive.")
        if fm_num_target_vision_tokens < 1 or fm_max_seq_len < 1:
            raise ValueError("fm_num_target_vision_tokens and fm_max_seq_len must be positive.")
        if fm_noise_beta_alpha <= 0 or fm_noise_beta_beta <= 0:
            raise ValueError("fm_noise_beta_alpha and fm_noise_beta_beta must be positive.")
        if not 0 < fm_noise_s <= 1:
            raise ValueError("fm_noise_s must be in the interval (0, 1].")
        if not 0 <= fm_state_dropout < 1:
            raise ValueError("fm_state_dropout must be in the interval [0, 1).")
        if ttt_num_register_tokens < 1:
            raise ValueError("ttt_num_register_tokens must be positive.")
        if ttt_memory_dim is not None and ttt_memory_dim < 1:
            raise ValueError("ttt_memory_dim must be positive when provided.")
        if ttt_memory_dim is not None and ttt_memory_dim % 2:
            raise ValueError("ttt_memory_dim must be even because TTT applies rotary position encoding.")
        if ttt_memory_hidden_dim is not None and ttt_memory_hidden_dim < 1:
            raise ValueError("ttt_memory_hidden_dim must be positive when provided.")
        if ttt_tbptt_step_size is not None and ttt_tbptt_step_size < 1:
            raise ValueError("ttt_tbptt_step_size must be positive or None.")

        action_model_cfg = {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32}
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]
        # Put the TTT layers inside the DiT blocks.  This matches RoboTTT's
        # placement: self/cross attention first mixes the current-step tokens,
        # then TTT reads/writes the cross-time state, and the block FFN follows.
        # The values written by the JEPA-WAM variant still come from the
        # WAM-predicted V-JEPA representation.
        resolved_ttt_layer_indices = tuple(ttt_layer_indices) if ttt_layer_indices else None
        diffusion_model_cfg = {
            **action_model_cfg,
            "cross_attention_dim": d_llm,
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_layers": fm_num_layers,
            "output_dim": fm_hidden_size,
            "positional_embeddings": None,
            "ttt_enabled": ttt_enabled,
            # ``None`` lets DiT select all blocks when TTT is enabled.  A
            # non-empty tuple selects only the requested blocks.
            "ttt_layer_indices": resolved_ttt_layer_indices,
            # TTT K/V stay in the native WAM/V-JEPA representation space;
            # this must be passed to DiT so its per-block memory MLP is built
            # with the same width as ``memory_tokens``.
            "ttt_memory_hidden_dim": ttt_memory_hidden_dim,
            "ttt_memory_dim": ttt_memory_dim,
            "ttt_tbptt_step_size": ttt_tbptt_step_size,
        }

        config = FlowMatchingActionHeadConfig(
            add_pos_embed=fm_add_pos_embed,
            diffusion_model_cfg=diffusion_model_cfg,
            input_embedding_dim=self.input_embedding_dim,
            hidden_size=fm_hidden_size,
            max_seq_len=fm_max_seq_len,
            action_dim=d_action,
            action_horizon=horizon,
            noise_beta_alpha=fm_noise_beta_alpha,
            noise_beta_beta=fm_noise_beta_beta,
            noise_s=fm_noise_s,
            num_timestep_buckets=fm_num_timestep_buckets,
            num_inference_timesteps=fm_num_inference_timesteps,
            num_target_vision_tokens=fm_num_target_vision_tokens,
            ttt_enabled=ttt_enabled,
            ttt_num_register_tokens=ttt_num_register_tokens,
            ttt_layer_indices=resolved_ttt_layer_indices or tuple(),
            ttt_memory_hidden_dim=ttt_memory_hidden_dim,
            ttt_memory_dim=ttt_memory_dim,
            ttt_tbptt_step_size=ttt_tbptt_step_size,
        )
        self.config = config
        self.full_config = SimpleNamespace(framework=SimpleNamespace(action_model=config))

        self.hidden_size = config.hidden_size
        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.ttt_enabled = ttt_enabled
        self.ttt_tbptt_step_size = ttt_tbptt_step_size

        self.state_encoder = MLP(
            input_dim=d_proprio,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.state_dropout = nn.Dropout(p=fm_state_dropout)
        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        self.register_tokens = None
        if self.ttt_enabled:
            if ttt_num_register_tokens < 1:
                raise ValueError("TTT requires at least one register token.")
            self.register_tokens = nn.Embedding(ttt_num_register_tokens, self.input_embedding_dim)
            nn.init.normal_(self.register_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def _prepare_state(self, proprio: torch.Tensor) -> torch.Tensor:
        return proprio

    @staticmethod
    def _flatten_time(tensor: torch.Tensor, batch: int, time_steps: int, name: str) -> torch.Tensor:
        if tensor.shape[:2] != (batch, time_steps):
            raise ValueError(
                f"Temporal `{name}` must start with [B, T] = {(batch, time_steps)}, got {tuple(tensor.shape)}."
            )
        return tensor.reshape(batch * time_steps, *tensor.shape[2:])

    def _predict_velocity(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        timesteps_tensor: torch.Tensor,
        state: torch.Tensor,
        *,
        memory_tokens: Optional[torch.Tensor] = None,
        prev_fast_weights=None,
        return_fast_weights: bool = False,
        update_fast_weights: bool = True,
        time_valid_mask: Optional[torch.Tensor] = None,
    ):
        temporal = actions.ndim == 4
        if temporal:
            batch, time_steps = actions.shape[:2]
            if vl_embs.ndim != 4:
                raise ValueError("Temporal actions require temporal [B, T, N, D] action conditioning.")
            if state.ndim != 3:
                raise ValueError("Temporal actions require temporal [B, T, D] proprioception.")
            if tuple(timesteps_tensor.shape) != (batch, time_steps):
                raise ValueError("Temporal flow timesteps must have shape [B, T].")
            if time_valid_mask is not None and tuple(time_valid_mask.shape) != (batch, time_steps):
                raise ValueError("time_valid_mask must have shape [B, T].")
            action_shape = actions.shape
            flat_vl_embs = self._flatten_time(vl_embs, batch, time_steps, "vl_embs")
            flat_actions = self._flatten_time(actions, batch, time_steps, "actions")
            flat_state = self._flatten_time(state, batch, time_steps, "proprio")
            flat_timesteps = timesteps_tensor.reshape(batch * time_steps)
            if memory_tokens is not None:
                flat_memory_tokens = self._flatten_time(memory_tokens, batch, time_steps, "memory_tokens")
            else:
                flat_memory_tokens = None
        else:
            if actions.ndim != 3 or vl_embs.ndim != 3 or state.ndim != 2:
                raise ValueError("Single-step action prediction expects [B, N, D], [B, H, A], and [B, D] tensors.")
            batch, time_steps = actions.shape[0], 1
            action_shape = actions.shape
            flat_vl_embs, flat_actions, flat_state = vl_embs, actions, state
            if timesteps_tensor.numel() != batch:
                raise ValueError("Single-step flow timesteps must contain one value per batch element.")
            flat_timesteps = timesteps_tensor.reshape(batch)
            if memory_tokens is not None:
                if memory_tokens.ndim != 3 or memory_tokens.shape[0] != batch:
                    raise ValueError("Single-step memory_tokens must have shape [B, N_memory, D_jepa].")
                flat_memory_tokens = memory_tokens
            else:
                flat_memory_tokens = None

        if self.ttt_enabled and flat_memory_tokens is None:
            raise ValueError("TTT-enabled JEPA-WAM action prediction requires predicted JEPA memory_tokens.")

        device = vl_embs.device
        flat_timesteps = flat_timesteps.to(device=device)
        if time_valid_mask is not None:
            time_valid_mask = time_valid_mask.to(device=device)
        compute_dtype = vl_embs.dtype
        flat_actions = flat_actions.to(dtype=compute_dtype)
        flat_state = flat_state.to(dtype=compute_dtype)
        if flat_memory_tokens is not None:
            flat_memory_tokens = flat_memory_tokens.to(dtype=compute_dtype)
        action_features = self.action_encoder(flat_actions, flat_timesteps)
        state_features = self.state_encoder(flat_state).unsqueeze(1)
        if state_features is not None:
            state_features = self.state_dropout(state_features)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(flat_vl_embs.shape[0], -1, -1)
        token_groups = [state_features]
        if self.register_tokens is not None:
            register_tokens = self.register_tokens.weight.unsqueeze(0).expand(flat_vl_embs.shape[0], -1, -1)
            token_groups.append(register_tokens)
        token_groups.extend((future_tokens, action_features))
        sa_embs = torch.cat(token_groups, dim=1)
        model_result = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=flat_vl_embs,
            timestep=flat_timesteps,
            return_all_hidden_states=False,
            time_steps=time_steps,
            ttt_memory_tokens=flat_memory_tokens,
            prev_fast_weights=prev_fast_weights,
            return_fast_weights=return_fast_weights,
            update_fast_weights=update_fast_weights,
            time_valid_mask=time_valid_mask,
            tbptt_step_size=self.ttt_tbptt_step_size,
        )
        if return_fast_weights:
            model_output, next_fast_weights = model_result
        else:
            model_output = model_result
            next_fast_weights = prev_fast_weights
        pred = self.action_decoder(model_output)
        pred = pred[:, -flat_actions.shape[1] :]
        if temporal:
            pred = pred.reshape(action_shape)
        if return_fast_weights:
            return pred, next_fast_weights
        return pred

    def forward(
        self,
        vl_embs: torch.Tensor,
        proprio: torch.Tensor,
        action_gt: torch.Tensor,
        memory_tokens: Optional[torch.Tensor] = None,
        time_valid_mask: Optional[torch.Tensor] = None,
        prev_fast_weights=None,
        return_fast_weights: bool = False,
        **_: object,
    ):
        compute_dtype = vl_embs.dtype
        state = self._prepare_state(proprio).to(dtype=compute_dtype)
        action_gt = action_gt.to(dtype=compute_dtype)
        noise = torch.randn(action_gt.shape, device=action_gt.device, dtype=action_gt.dtype)
        temporal = action_gt.ndim == 4
        if temporal:
            batch, time_steps = action_gt.shape[:2]
            t = self.sample_time(batch * time_steps, device=action_gt.device, dtype=action_gt.dtype)
            t = t.reshape(batch, time_steps, 1, 1)
            t_discretized = (t[..., 0, 0] * self.num_timestep_buckets).long()
        else:
            t = self.sample_time(action_gt.shape[0], device=action_gt.device, dtype=action_gt.dtype)
            t = t[:, None, None]
            t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

        noisy_trajectory = (1 - t) * noise + t * action_gt
        velocity = action_gt - noise
        use_ttt = getattr(self, "ttt_enabled", False)
        if return_fast_weights or use_ttt:
            pred_actions, next_fast_weights = self._predict_velocity(
                vl_embs,
                noisy_trajectory,
                t_discretized,
                state,
                memory_tokens=memory_tokens,
                prev_fast_weights=prev_fast_weights,
                return_fast_weights=True,
                time_valid_mask=time_valid_mask,
            )
        else:
            # Keep the original call shape for backward compatibility with
            # external action-head subclasses and existing checkpoints.
            pred_actions = self._predict_velocity(vl_embs, noisy_trajectory, t_discretized, state)
        squared_error = (pred_actions - velocity) ** 2
        if time_valid_mask is None:
            loss = squared_error.mean()
        else:
            if not temporal or tuple(time_valid_mask.shape) != tuple(action_gt.shape[:2]):
                raise ValueError("Temporal action loss expects time_valid_mask with shape [B, T].")
            mask = time_valid_mask.to(
                device=squared_error.device, dtype=squared_error.dtype
            ).unsqueeze(-1).unsqueeze(-1)
            loss = (squared_error * mask).sum() / mask.expand_as(squared_error).sum().clamp_min(1)
        if return_fast_weights:
            return loss, pred_actions, next_fast_weights
        return loss, pred_actions

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        proprio: torch.Tensor,
        *,
        memory_tokens: Optional[torch.Tensor] = None,
        prev_fast_weights=None,
        return_fast_weights: bool = False,
    ):
        state = self._prepare_state(proprio).to(dtype=vl_embs.dtype)
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        if num_steps < 1:
            raise ValueError("num_inference_timesteps must be positive.")
        dt = 1.0 / num_steps

        next_fast_weights = prev_fast_weights
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device, dtype=torch.long)
            if self.ttt_enabled:
                # One physical observation should create one TTT update.
                # The remaining flow-matching evaluations refine the same
                # action chunk and must only read the state written at the
                # first denoising step; otherwise one observation would be
                # counted as ``num_inference_timesteps`` environment steps.
                pred_velocity, next_fast_weights = self._predict_velocity(
                    vl_embs,
                    actions,
                    timesteps_tensor,
                    state,
                    memory_tokens=memory_tokens,
                    prev_fast_weights=next_fast_weights,
                    return_fast_weights=True,
                    update_fast_weights=(t == 0),
                )
            else:
                pred_velocity = self._predict_velocity(vl_embs, actions, timesteps_tensor, state)
            actions = actions + dt * pred_velocity
        if return_fast_weights:
            return actions, next_fast_weights
        return actions

    @torch.no_grad()
    def sample_action(
        self,
        vl_embs: torch.Tensor,
        proprio: torch.Tensor,
        num_steps: int | None = None,
        *,
        memory_tokens: Optional[torch.Tensor] = None,
        prev_fast_weights=None,
        return_fast_weights: bool = False,
    ):
        if num_steps is not None:
            if num_steps < 1:
                raise ValueError("num_steps must be positive when provided.")
            old_steps = self.num_inference_timesteps
            self.num_inference_timesteps = num_steps
            try:
                return self.predict_action(
                    vl_embs,
                    proprio,
                    memory_tokens=memory_tokens,
                    prev_fast_weights=prev_fast_weights,
                    return_fast_weights=return_fast_weights,
                )
            finally:
                self.num_inference_timesteps = old_steps
        return self.predict_action(
            vl_embs,
            proprio,
            memory_tokens=memory_tokens,
            prev_fast_weights=prev_fast_weights,
            return_fast_weights=return_fast_weights,
        )
