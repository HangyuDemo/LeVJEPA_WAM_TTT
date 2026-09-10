"""
prismatic.py

PyTorch module implementing the fixed JEPA-WAM policy.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional, Type, Union

import torch
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy

from prismatic.models.action_heads import VisualTokenCosineHead
from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.flow_gr00t_action_head import FlowMatchingActionHead
from prismatic.models.vlms.base_vlm import VLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import MLPProjector
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, NUM_TOKENS, PROPRIO_DIM

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def _maybe_cuda_mem_snapshot(label: str):
    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    return {
        "label": label,
        "allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
    }


class PrismaticVLM(VLM):
    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        **kwargs,
    ) -> None:
        super().__init__(
            "prismatic",
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )

        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(vision_backbone.embed_dim)

        # The released base VLM uses the two-layer GELU projector.
        self.arch_specifier = arch_specifier
        if not arch_specifier.endswith("gelu-mlp"):
            raise ValueError(f"The public JEPA-WAM recipe requires a GELU MLP projector, got `{arch_specifier}`.")
        self.projector = MLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)

        # Trackers
        self.vision_backbone_requires_grad = False

        # Fixed public heads: Flow-GR00T plus final-layer visual-token cosine alignment.
        self.action_placeholder_tokens = kwargs.get("flow_gr00t_placeholder_tokens", NUM_TOKENS)
        self.lambda_visual_token_cosine = kwargs.get("lambda_visual_token_cosine", 0.5)
        jepa_dim = kwargs.get("d_jepa", vision_backbone.embed_dim)
        ttt_memory_dim = kwargs.get("ttt_memory_dim") or jepa_dim
        if kwargs.get("ttt_enabled", False) and ttt_memory_dim != jepa_dim:
            raise ValueError(
                "ttt_memory_dim must equal d_jepa: TTT memory is required to remain in native V-JEPA space."
            )
        self.action_head = FlowMatchingActionHead(
            d_proprio=kwargs.get("d_proprio", PROPRIO_DIM),
            d_action=kwargs.get("d_action", ACTION_DIM),
            d_llm=llm_backbone.embed_dim,
            horizon=kwargs.get("action_horizon", NUM_ACTIONS_CHUNK),
            fm_hidden_size=kwargs.get("fm_hidden_size", 1024),
            fm_num_layers=kwargs.get("fm_num_layers", 16),
            fm_num_inference_timesteps=kwargs.get("fm_num_inference_timesteps", 4),
            fm_num_timestep_buckets=kwargs.get("fm_num_timestep_buckets", 1000),
            fm_noise_beta_alpha=kwargs.get("fm_noise_beta_alpha", 1.5),
            fm_noise_beta_beta=kwargs.get("fm_noise_beta_beta", 1.0),
            fm_noise_s=kwargs.get("fm_noise_s", 0.999),
            fm_num_target_vision_tokens=kwargs.get("fm_num_target_vision_tokens", 32),
            fm_add_pos_embed=kwargs.get("fm_add_pos_embed", True),
            fm_max_seq_len=kwargs.get("fm_max_seq_len", 1024),
            fm_state_dropout=kwargs.get("fm_state_dropout", 0.5),
            ttt_enabled=kwargs.get("ttt_enabled", False),
            ttt_num_register_tokens=kwargs.get("ttt_num_register_tokens", 16),
            ttt_layer_indices=kwargs.get("ttt_layer_indices") or None,
            ttt_memory_hidden_dim=kwargs.get("ttt_memory_hidden_dim"),
            ttt_memory_dim=ttt_memory_dim,
            ttt_tbptt_step_size=kwargs.get("ttt_tbptt_step_size"),
        )
        self.visual_token_cosine_head = VisualTokenCosineHead(
            d_llm=llm_backbone.embed_dim,
            d_target=jepa_dim,
        )

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = [
            "vision_backbone",
            "llm_backbone",
            "projector",
            "action_head",
            "visual_token_cosine_head",
        ]
        self.trainable_module_keys = []
        overwatch.info("Initialized Flow-GR00T action head with %d placeholder tokens", self.action_placeholder_tokens)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        load_visual_token_cosine_head: bool = True,
        **kwargs,
    ) -> PrismaticVLM:
        """Initialize a PrismaticVLM from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vlm = cls(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )
        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"

        vlm.projector.load_state_dict(model_state_dict["projector"])
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])
        if "action_head" in model_state_dict:
            missing, unexpected = vlm.action_head.load_state_dict(
                model_state_dict["action_head"],
                # A pretrained non-TTT checkpoint has no parameters for newly
                # added TTT layers.  Keep those layers at their initialization
                # while restoring all matching DiT parameters.
                strict=False,
            )
            if (missing or unexpected) and not vlm.action_head.ttt_enabled:
                raise RuntimeError(
                    "Action-head checkpoint mismatch for a non-TTT model "
                    f"(missing={missing}, unexpected={unexpected})."
                )
            if missing or unexpected:
                overwatch.info(
                    "Action-head checkpoint partially loaded (expected for new TTT layers): "
                    "missing=%s unexpected=%s",
                    missing,
                    unexpected,
                )
        if "visual_token_cosine_head" in model_state_dict:
            if load_visual_token_cosine_head:
                missing, unexpected = vlm.visual_token_cosine_head.load_state_dict(
                    model_state_dict["visual_token_cosine_head"],
                    strict=False,
                )
                if missing or unexpected:
                    overwatch.info(
                        "Visual Token Cosine Head checkpoint mismatch ignored "
                        "(missing=%s unexpected=%s)",
                        missing,
                        unexpected,
                    )
            else:
                overwatch.info("Skipping visual_token_cosine_head checkpoint load by request.")

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        return vlm

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_for_training(
        self,
        *,
        train_qwen_lora: bool = True,
        train_action_head: bool = True,
        train_visual_token_cosine_head: bool = True,
    ) -> None:
        """Freeze the fixed base and select which JEPA-WAM modules to train.

        The vision encoder and projector remain frozen by design.  Qwen's
        non-LoRA weights are always frozen; ``train_qwen_lora`` controls only
        the LoRA adapters.  This makes action-expert-only fine-tuning explicit
        instead of relying on incidental parameter names.
        """
        self.vision_backbone.requires_grad_(False)
        self.projector.requires_grad_(False)
        self.llm_backbone.requires_grad_(False)

        lora_param_names = []
        for name, param in self.llm_backbone.named_parameters():
            if train_qwen_lora and "lora_" in name:
                param.requires_grad_(True)
                lora_param_names.append(name)
        if train_qwen_lora and not lora_param_names:
            raise RuntimeError("Qwen must be wrapped with LoRA before calling `freeze_for_training`.")

        self.action_head.requires_grad_(train_action_head)
        self.visual_token_cosine_head.requires_grad_(train_visual_token_cosine_head)
        self.trainable_module_keys = []
        if train_qwen_lora:
            self.trainable_module_keys.append("llm_backbone")
        if train_action_head:
            self.trainable_module_keys.append("action_head")
        if train_visual_token_cosine_head:
            self.trainable_module_keys.append("visual_token_cosine_head")
        if not self.trainable_module_keys:
            raise ValueError("At least one trainable JEPA-WAM module must be selected.")
        self.vision_backbone_requires_grad = False

        overwatch.info(f"[Frozen] =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
        overwatch.info(f"[Frozen] =>> Projector `{self.arch_specifier}`", ctx_level=1)
        if train_qwen_lora:
            overwatch.info(
                f"[TRAINABLE] =>> Qwen LoRA (`{len(lora_param_names)}` parameter groups matched)",
                ctx_level=1,
            )
        else:
            overwatch.info("[Frozen] =>> Qwen (including LoRA)", ctx_level=1)
        if train_action_head:
            overwatch.info(
                f"[TRAINABLE] =>> Flow-GR00T Action Head (placeholders={self.action_placeholder_tokens})",
                ctx_level=1,
            )
        else:
            overwatch.info("[Frozen] =>> Flow-GR00T Action Head", ctx_level=1)
        if train_visual_token_cosine_head:
            overwatch.info("[TRAINABLE] =>> Visual Token Cosine Head", ctx_level=1)
        else:
            overwatch.info("[Frozen] =>> Visual Token Cosine Head", ctx_level=1)

        overwatch.debug("##################################################")
        overwatch.debug("#####      Trainable Network Parameters:     #####")
        overwatch.debug("##################################################")
        for name, param in self.named_parameters():
            if param.requires_grad:
                overwatch.debug(name)

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> projector and fixed action/alignment heads
        head_classes = {MLPProjector, FlowMatchingActionHead, VisualTokenCosineHead}

        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes=head_classes,
        )

        # Return union (_or_) over constituent policies
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    @staticmethod
    def _select_action_memory(
        llm_hidden: torch.Tensor,
        fused_attention_mask: torch.Tensor,
        num_action_tokens: int,
    ) -> torch.Tensor:
        """Use the final action-placeholder span from the padded sequence."""
        if llm_hidden.shape[1] < num_action_tokens:
            raise ValueError("Input sequence is shorter than the configured action placeholder span.")
        return llm_hidden[:, -num_action_tokens:, :]

    @staticmethod
    def _flatten_temporal_tree(value, batch: int, time_steps: int):
        """Flatten a leading [B, T] axis on tensor/dict vision inputs."""
        if isinstance(value, torch.Tensor):
            if value.shape[:2] != (batch, time_steps):
                raise ValueError(
                    f"Temporal vision input must start with [B, T] = {(batch, time_steps)}, got {tuple(value.shape)}."
                )
            return value.reshape(batch * time_steps, *value.shape[2:])
        if isinstance(value, dict):
            return {
                key: PrismaticVLM._flatten_temporal_tree(child, batch, time_steps)
                for key, child in value.items()
            }
        raise TypeError(f"Unsupported temporal vision input type `{type(value)}`.")

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Union[torch.FloatTensor, Dict[str, torch.Tensor]],
        pair_pixel_values: Optional[Union[torch.FloatTensor, Dict[str, torch.Tensor]]] = None,
        actions: Optional[torch.FloatTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
        time_valid_mask: Optional[torch.Tensor] = None,
        prev_fast_weights=None,
        return_fast_weights: bool = False,
    ) -> dict:
        """Run the fixed JEPA-WAM action and visual-alignment forward pass."""
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Expected input_ids and attention_mask with matching [B, L] shapes.")

        # Legacy callers sometimes carry a singleton action axis [B, 1, H, A].
        # Treat only real multi-step windows as temporal; the singleton case
        # keeps the released single-frame path unchanged.
        temporal = actions is not None and actions.ndim == 4 and actions.shape[1] > 1
        if temporal:
            batch, time_steps = actions.shape[:2]
            if proprio is None or proprio.ndim != 3 or proprio.shape[:2] != (batch, time_steps):
                raise ValueError("Temporal actions require proprioception with shape [B, T, D].")
            if time_valid_mask is not None and tuple(time_valid_mask.shape) != (batch, time_steps):
                raise ValueError("time_valid_mask must have shape [B, T].")
            model_pixel_values = self._flatten_temporal_tree(pixel_values, batch, time_steps)
            model_pair_pixel_values = (
                self._flatten_temporal_tree(pair_pixel_values, batch, time_steps)
                if pair_pixel_values is not None
                else None
            )
            model_input_ids = input_ids.repeat_interleave(time_steps, dim=0)
            model_attention_mask = attention_mask.repeat_interleave(time_steps, dim=0)
        else:
            batch, time_steps = input_ids.shape[0], 1
            model_pixel_values = pixel_values
            model_pair_pixel_values = pair_pixel_values
            model_input_ids = input_ids
            model_attention_mask = attention_mask

        memory_stats = [] if getattr(self, "debug_memory_stats", False) else None

        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            patch_features = self.vision_backbone(model_pixel_values)
        if memory_stats is not None and (snap := _maybe_cuda_mem_snapshot("after_vision_encode")) is not None:
            memory_stats.append(snap)

        pair_vjepa_target = None
        if model_pair_pixel_values is not None:
            if not hasattr(self.vision_backbone, "encode_pair"):
                raise TypeError("The configured vision backbone does not implement paired-frame encoding.")
            with torch.no_grad():
                pair_vjepa_target = self.vision_backbone.encode_pair(model_pair_pixel_values)

        projected_patch_embeddings = self.projector(patch_features)
        if memory_stats is not None and (snap := _maybe_cuda_mem_snapshot("after_projector")) is not None:
            memory_stats.append(snap)

        projected_patch_attention_mask = torch.ones(
            projected_patch_embeddings.shape[:2],
            dtype=model_attention_mask.dtype,
            device=model_attention_mask.device,
        )
        input_embeddings = self.llm_backbone.embed_input_ids(model_input_ids)
        fused_embeddings = torch.cat(
            [
                input_embeddings[:, :1, :],
                projected_patch_embeddings,
                input_embeddings[:, 1:, :],
            ],
            dim=1,
        )
        fused_attention_mask = torch.cat(
            [
                model_attention_mask[:, :1],
                projected_patch_attention_mask,
                model_attention_mask[:, 1:],
            ],
            dim=1,
        )

        llm_output = self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=fused_embeddings,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        if memory_stats is not None and (snap := _maybe_cuda_mem_snapshot("after_llm_forward")) is not None:
            memory_stats.append(snap)
        if llm_output.hidden_states is None:
            raise RuntimeError("Qwen did not return hidden states.")

        llm_hidden = llm_output.hidden_states[-1]
        vision_token_count = projected_patch_embeddings.shape[1]
        vision_memory = llm_hidden[:, 1 : 1 + vision_token_count, :]
        action_memory = self._select_action_memory(
            llm_hidden,
            fused_attention_mask,
            self.action_placeholder_tokens,
        )
        # JEPA-WAM DiT conditioning: use only the action-placeholder hidden
        # states.  These tokens already contain the fused visual, language,
        # and task context needed by the action expert.  Keep the raw visual
        # Qwen tokens separate: they are mapped below to the WAM-predicted
        # JEPA representation and used exclusively as TTT memory.
        vl_condition = action_memory

        # This is the WAM prediction (Y_hat) in V-JEPA space.  It is computed from
        # the current observation and is therefore available at deployment.
        # The paired V-JEPA target (Y) below remains stop-gradient supervision
        # only; it is never used as TTT memory.
        wam_representation = None
        if self.action_head.ttt_enabled or (pair_vjepa_target is not None and self.training):
            # Cosine supervision leaves the prediction scale unconstrained;
            # normalize before writing it into fast weights for stable TTT
            # updates while preserving its V-JEPA direction.
            wam_representation = F.normalize(
                self.visual_token_cosine_head.align_dimension(vision_memory), dim=-1
            )

        if isinstance(model_pixel_values, torch.Tensor):
            num_views = model_pixel_values.shape[1] if model_pixel_values.ndim == 5 else 1
        else:
            example = next(iter(model_pixel_values.values()))
            num_views = example.shape[1] if example.ndim == 5 else 1
        if vision_token_count % num_views != 0:
            raise ValueError(
                f"Vision token count {vision_token_count} is not divisible by num views {num_views}."
            )

        total_loss = llm_hidden.new_zeros(())
        loss_action = None
        loss_visual_token_cosine = None

        if (actions is None) != (proprio is None):
            raise ValueError("Actions and proprio must be provided together.")
        if actions is not None:
            if not temporal and actions.ndim == 4 and actions.shape[1] == 1:
                actions = actions.squeeze(1)
            if actions.ndim == 2:
                actions = actions.unsqueeze(1)
            if not temporal and proprio.ndim == 3 and proprio.shape[1] == 1:
                proprio = proprio.squeeze(1)

            action_result = self.action_head(
                (
                    vl_condition.reshape(
                        batch, time_steps, vl_condition.shape[1], vl_condition.shape[2]
                    )
                    if temporal
                    else vl_condition
                ),
                proprio,
                actions,
                memory_tokens=(
                    wam_representation.reshape(
                        batch, time_steps, wam_representation.shape[1], wam_representation.shape[2]
                    )
                    if temporal and wam_representation is not None
                    else wam_representation
                ),
                time_valid_mask=time_valid_mask,
                prev_fast_weights=prev_fast_weights,
                return_fast_weights=return_fast_weights,
            )
            if return_fast_weights:
                loss_action, _, next_fast_weights = action_result
            else:
                loss_action, _ = action_result
            total_loss = total_loss + loss_action

        if pair_vjepa_target is not None and self.training:
            if wam_representation is None:
                raise RuntimeError("WAM visual prediction is required for the V-JEPA alignment loss.")
            if pair_vjepa_target.shape[2] != 1:
                raise ValueError(
                    "Visual-token cosine supervision expects one temporal target token, "
                    f"got {tuple(pair_vjepa_target.shape)}."
                )
            target_grid = pair_vjepa_target.squeeze(2)
            target_visual_tokens = target_grid.reshape(
                target_grid.shape[0],
                target_grid.shape[1] * target_grid.shape[2] * target_grid.shape[3],
                target_grid.shape[-1],
            ).detach()
            if vision_memory.shape[1] != target_visual_tokens.shape[1]:
                raise ValueError(
                    f"Visual token count mismatch: prediction={vision_memory.shape[1]}, "
                    f"target={target_visual_tokens.shape[1]}."
                )

            visual_memory_for_loss = (
                wam_representation.reshape(
                    batch, time_steps, wam_representation.shape[1], wam_representation.shape[2]
                )
                if temporal
                else wam_representation
            )
            target_for_loss = (
                target_visual_tokens.reshape(
                    batch, time_steps, target_visual_tokens.shape[1], target_visual_tokens.shape[2]
                )
                if temporal
                else target_visual_tokens
            )
            loss_visual_token_cosine = self.visual_token_cosine_head.compute_align_loss_cosine(
                visual_memory_for_loss,
                target_for_loss,
                time_valid_mask=time_valid_mask if temporal else None,
            )
            total_loss = total_loss + self.lambda_visual_token_cosine * loss_visual_token_cosine
            if memory_stats is not None and (
                snap := _maybe_cuda_mem_snapshot("after_visual_token_cosine_head")
            ) is not None:
                memory_stats.append(snap)

        output_hidden = (
            llm_hidden.reshape(batch, time_steps, llm_hidden.shape[1], llm_hidden.shape[2]) if temporal else llm_hidden
        )
        output = {"loss": total_loss, "llm_hidden": output_hidden}
        output["vl_condition"] = (
            vl_condition.reshape(batch, time_steps, vl_condition.shape[1], vl_condition.shape[2])
            if temporal
            else vl_condition
        )
        if wam_representation is not None:
            output["wam_representation"] = (
                wam_representation.reshape(
                    batch, time_steps, wam_representation.shape[1], wam_representation.shape[2]
                )
                if temporal
                else wam_representation
            )
        if loss_action is not None:
            output["loss_action"] = loss_action
        if loss_visual_token_cosine is not None:
            output["loss_visual_token_cosine"] = loss_visual_token_cosine
        if return_fast_weights:
            output["next_fast_weights"] = next_fast_weights if actions is not None else prev_fast_weights
        if memory_stats is not None:
            output["memory_stats"] = memory_stats
        return output
