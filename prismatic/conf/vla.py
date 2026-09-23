"""Fixed public JEPA-WAM training configuration."""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Tuple, Union

from draccus import ChoiceRegistry

from prismatic.vla.constants import NUM_ACTIONS_CHUNK, NUM_TOKENS


@dataclass
class VLAConfig(ChoiceRegistry):
    """The released V-JEPA 2.1 + Qvv2.5 + Flow-GR00T recipe."""

    vla_id: str = "jepavla-qvv25-vjepa-224px+0_5b+mx-libero-90"
    base_vlm: Union[str, Path] = "prism-qvv25-vjepa21-vitl-384px+0_5b"

    data_mix: str = "libero_4_task_suites_no_noops"
    shuffle_buffer_size: int = 20_000

    max_steps: int = 40_000
    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03

    vjepa_checkpoint_path: Optional[str] = None
    # Separate path for the opt-in LeVJEPA visual encoder. Keeping this
    # distinct prevents a LeVJEPA run from accidentally loading V-JEPA 2.1.
    levjepa_checkpoint_path: Optional[str] = None
    # Select the visual encoder while keeping the JEPA-WAM action/TTT stack
    # unchanged.  If unset, use the backbone recorded by the base VLM run.
    vision_backbone_id: Optional[str] = None

    # Select which trainable JEPA-WAM modules are updated after loading the
    # frozen base VLM.  The default preserves the released recipe; disabling
    # Qvv LoRA and the visual head gives action-expert-only fine-tuning.
    train_qvv_lora: bool = True
    # A replacement vision encoder needs a fresh bridge into Qvv space; the
    # released projector was trained on V-JEPA features.
    train_projector: bool = False
    train_action_head: bool = True
    # RoboTTT-style pretraining: freeze the pretrained action head and update
    # only the newly inserted TTT slow parameters.
    train_ttt_only: bool = False
    train_visual_token_cosine_head: bool = True

    d_action: int = 7
    d_proprio: int = 8
    action_horizon: int = NUM_ACTIONS_CHUNK
    flow_gr00t_placeholder_tokens: int = NUM_TOKENS
    fm_hidden_size: int = 1024
    fm_num_layers: int = 16
    fm_num_inference_timesteps: int = 4
    fm_num_timestep_buckets: int = 1_000
    fm_noise_beta_alpha: float = 1.5
    fm_noise_beta_beta: float = 1.0
    fm_noise_s: float = 0.999
    fm_num_target_vision_tokens: int = 32
    fm_add_pos_embed: bool = True
    fm_max_seq_len: int = 1_024
    fm_state_dropout: float = 0.5

    # RoboTTT-style action-token fast-weight memory.  Disabled by default so
    # the released single-frame JEPA-WAM checkpoint remains load-compatible.
    ttt_enabled: bool = False
    # ``jepa`` preserves the existing explicit WAM-representation K/V route;
    # ``action_tokens`` is the compatibility name for implicit DiT K/V; the
    # separate ttt_action_kv_scope controls selected-token versus full-stream K/V.
    ttt_memory_source: str = "jepa"
    # ``inline`` is the original TTT-in-each-DiT-block implementation;
    # ``wrapper`` is the RoboTTT-style selected-layer adapter.
    ttt_architecture: str = "wrapper"
    ttt_context_length: int = 1
    # Opt-in round-2 training: context is the TOTAL sequence length. Each
    # forward/backward handles ttt_tbptt_step_size frames and carries state.
    ttt_carry_between_segments: bool = False
    ttt_require_full_context: bool = False
    ttt_num_register_tokens: int = 16
    # Legacy wrapper switch; unified scope always includes registers.
    ttt_wrapper_register_tokens: bool = True
    # New training default; checkpoint loading without this field uses legacy.
    ttt_token_scope: str = "state_register_action"
    # Implicit-memory route: use the complete post-attention DiT stream for K/V.
    ttt_action_kv_scope: str = "full_dit_tokens"
    # Empty means architecture-specific defaults: all DiT blocks for the
    # inline route, or (3, 7, 11, 15) for the selected-layer wrapper route.
    ttt_layer_indices: Tuple[int, ...] = ()
    ttt_memory_hidden_dim: Optional[int] = None
    # Dimension of the representation written into TTT.  By default this is
    # the frozen visual-encoder embedding dimension, not the DiT hidden dimension.
    ttt_memory_dim: Optional[int] = None
    ttt_tbptt_step_size: Optional[int] = 8

    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: Union[str, Tuple[str, ...]] = "all-linear"
    visual_token_pair_offset: int = 31
    lambda_visual_token_cosine: float = 0.5

    enable_gradient_checkpointing: bool = True
    enable_mixed_precision_training: bool = True
    reduce_in_full_precision: bool = True

    def __post_init__(self) -> None:
        # draccus/YAML may materialize a list or null for this optional tuple.
        # Normalize it once so validation and model construction see one shape.
        self.ttt_layer_indices = tuple(self.ttt_layer_indices or ())
        positive_fields = {
            "max_steps": self.max_steps,
            "expected_world_size": self.expected_world_size,
            "global_batch_size": self.global_batch_size,
            "per_device_batch_size": self.per_device_batch_size,
            "d_action": self.d_action,
            "d_proprio": self.d_proprio,
            "action_horizon": self.action_horizon,
            "flow_gr00t_placeholder_tokens": self.flow_gr00t_placeholder_tokens,
            "fm_hidden_size": self.fm_hidden_size,
            "fm_num_layers": self.fm_num_layers,
            "fm_num_inference_timesteps": self.fm_num_inference_timesteps,
            "fm_num_timestep_buckets": self.fm_num_timestep_buckets,
            "fm_num_target_vision_tokens": self.fm_num_target_vision_tokens,
            "fm_max_seq_len": self.fm_max_seq_len,
        }
        invalid = [name for name, value in positive_fields.items() if value < 1]
        if invalid:
            raise ValueError(f"These VLA dimensions/step counts must be positive: {', '.join(invalid)}.")
        if self.global_batch_size % self.per_device_batch_size != 0:
            raise ValueError("global_batch_size must be divisible by per_device_batch_size.")
        if self.global_batch_size % (self.per_device_batch_size * self.expected_world_size) != 0:
            raise ValueError(
                "global_batch_size must be divisible by per_device_batch_size * expected_world_size; "
                f"got global={self.global_batch_size}, per_device={self.per_device_batch_size}, "
                f"expected_world_size={self.expected_world_size}."
            )
        if self.fm_noise_beta_alpha <= 0 or self.fm_noise_beta_beta <= 0:
            raise ValueError("fm_noise_beta_alpha and fm_noise_beta_beta must be positive.")
        if not 0 < self.fm_noise_s <= 1:
            raise ValueError("fm_noise_s must be in the interval (0, 1].")
        if not 0 <= self.fm_state_dropout < 1:
            raise ValueError("fm_state_dropout must be in the interval [0, 1).")
        if self.ttt_memory_dim is not None and self.ttt_memory_dim < 1:
            raise ValueError("ttt_memory_dim must be positive when provided.")
        if self.ttt_memory_hidden_dim is not None and self.ttt_memory_hidden_dim < 1:
            raise ValueError("ttt_memory_hidden_dim must be positive when provided.")
        if any(index < 0 for index in self.ttt_layer_indices):
            raise ValueError("ttt_layer_indices must contain non-negative DiT block indices.")
        if len(set(self.ttt_layer_indices)) != len(self.ttt_layer_indices):
            raise ValueError("ttt_layer_indices must not contain duplicates.")
        if self.ttt_context_length < 1:
            raise ValueError("ttt_context_length must be positive.")
        if self.ttt_memory_source not in {"jepa", "action_tokens"}:
            raise ValueError("ttt_memory_source must be either 'jepa' or 'action_tokens'.")
        if self.ttt_architecture not in {"inline", "wrapper"}:
            raise ValueError("ttt_architecture must be either 'inline' or 'wrapper'.")
        if self.ttt_token_scope not in {"legacy", "state_register_action"}:
            raise ValueError("ttt_token_scope must be 'legacy' or 'state_register_action'.")
        if self.ttt_action_kv_scope not in {"query_tokens", "full_dit_tokens"}:
            raise ValueError("ttt_action_kv_scope must be 'query_tokens' or 'full_dit_tokens'.")
        if self.ttt_enabled and self.ttt_context_length < 2:
            raise ValueError("RoboTTT training requires ttt_context_length >= 2.")
        if self.ttt_num_register_tokens < 1:
            raise ValueError("ttt_num_register_tokens must be positive.")
        if self.ttt_tbptt_step_size is not None and self.ttt_tbptt_step_size < 1:
            raise ValueError("ttt_tbptt_step_size must be positive or None.")
        if self.ttt_carry_between_segments:
            if not self.ttt_enabled or not self.ttt_tbptt_step_size or self.ttt_tbptt_step_size < 2:
                raise ValueError("Segmented TTT requires ttt_enabled and tbptt_step_size>=2.")
            if self.ttt_context_length % self.ttt_tbptt_step_size:
                raise ValueError("ttt_context_length must be divisible by ttt_tbptt_step_size.")
            if self.visual_token_pair_offset != 0:
                raise ValueError("Segmented action-only TTT requires visual_token_pair_offset=0.")


Exp_JEPAVLA_Qvv25_VJEPA_0_5B_LIBERO_90 = VLAConfig


@unique
class VLARegistry(Enum):
    JEPAVLA_QVV25_VJEPA_224PX_0_5B_LIBERO_90 = VLAConfig

    @property
    def vla_id(self) -> str:
        return self.value.vla_id


VLAConfig.register_subclass(VLAConfig.vla_id, VLAConfig)
