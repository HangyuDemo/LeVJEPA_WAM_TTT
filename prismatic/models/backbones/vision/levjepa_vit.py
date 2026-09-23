"""LeVJEPA vision-backbone adapter for JEPA-WAM.

LeVJEPA is exposed through the HuggingFace ``AutoModel`` interface.  This
adapter keeps the same contract as the V-JEPA 2.1 wrapper used by Prismatic:
the public ``forward`` returns dense patch tokens (without CLS), while
``encode_pair`` returns the patch tokens for the future member of a two-frame
pair as a JEPA target.
"""

import os
from functools import partial
from typing import Callable, Optional, Tuple

import torch
from torch.distributed.fsdp.wrap import _module_wrap_policy
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
from transformers import AutoModel

from prismatic.models.backbones.vision.base_vision import ImageTransform, VisionBackbone


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LeVJEPAViTBackbone(VisionBackbone):
    """Frozen LeVJEPA feature extractor with the JEPA-WAM backbone contract."""

    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        checkpoint_path: Optional[str] = None,
        num_frames: int = 16,
        freeze_backbone: bool = True,
    ) -> None:
        if image_resize_strategy not in {"resize-naive", "resize-crop"}:
            raise ValueError(
                f"Image Resize Strategy `{image_resize_strategy}` is not supported for LeVJEPA."
            )
        checkpoint_path = checkpoint_path or os.environ.get("LEVJEPA_CHECKPOINT_PATH")
        if not checkpoint_path:
            raise ValueError(
                "A LeVJEPA checkpoint is required. Pass `levjepa_checkpoint_path` or "
                "set LEVJEPA_CHECKPOINT_PATH."
            )

        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        self.num_frames = int(num_frames)
        self.patch_size = 16
        self.freeze_backbone = freeze_backbone
        self.featurizer = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True)
        self.featurizer = self.featurizer.to(dtype=torch.bfloat16)
        if freeze_backbone:
            self.featurizer.requires_grad_(False)
            self.featurizer.eval()

        config = self.featurizer.config
        self._embed_dim = int(getattr(config, "hidden_size", getattr(config, "embed_dim", 0)))
        if self._embed_dim <= 0:
            raise ValueError("Could not determine LeVJEPA hidden dimension from its config.")
        self._num_spatial_patches = (self.default_image_size // self.patch_size) ** 2

        self.image_transform = Compose(
            [
                Resize(
                    (self.default_image_size, self.default_image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                ToTensor(),
                Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _run_encoder(self, clips: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, 3, T, H, W]`` and return patch tokens including time."""
        if clips.ndim != 5:
            raise ValueError(f"Expected clips with shape [B, 3, T, H, W], got {tuple(clips.shape)}")
        device = next(self.featurizer.parameters()).device
        dtype = next(self.featurizer.parameters()).dtype
        clips = clips.to(device=device, dtype=dtype)
        output = self.featurizer(pixel_values=clips)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, dict):
            hidden = output.get("last_hidden_state")
        if hidden is None:
            raise RuntimeError("LeVJEPA did not return `last_hidden_state`.")
        if hidden.shape[1] < 2:
            raise RuntimeError("LeVJEPA output does not contain CLS and patch tokens.")
        return hidden[:, 1:, :]

    def _as_released_clip(self, frames: torch.Tensor) -> torch.Tensor:
        """Make a fixed-length clip, repeating the final frame when necessary."""
        if frames.ndim != 5:
            raise ValueError(f"Expected frames [B, 3, T, H, W], got {tuple(frames.shape)}")
        t = frames.shape[2]
        if t == self.num_frames:
            return frames
        if t > self.num_frames:
            return frames[:, :, -self.num_frames :]
        return torch.cat((frames, frames[:, :, -1:].expand(-1, -1, self.num_frames - t, -1, -1)), dim=2)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode current image views and return one spatial slice per view."""
        if pixel_values.ndim == 4:
            # [B, 3, H, W] -> one view per sample.
            batch_size, channels, height, width = pixel_values.shape
            num_views = 1
            images = pixel_values
        elif pixel_values.ndim == 5:
            # Prismatic's 5-D image input is [B, V, 3, H, W].
            batch_size, num_views, channels, height, width = pixel_values.shape
            images = pixel_values.reshape(batch_size * num_views, channels, height, width)
        else:
            raise ValueError(
                "Expected image tensor [B, 3, H, W] or multi-view tensor [B, V, 3, H, W], "
                f"got {tuple(pixel_values.shape)}"
            )

        clips = images.unsqueeze(2).expand(-1, -1, self.num_frames, -1, -1)
        patch_tokens = self._run_encoder(clips)
        expected = self.num_frames * self._num_spatial_patches
        if patch_tokens.shape[1] != expected:
            raise ValueError(
                f"Unexpected LeVJEPA token count: got {patch_tokens.shape[1]}, expected {expected}."
            )
        # Keep only the last temporal slice so Qvv sees the same per-view
        # token interface as the existing image backbone.
        patch_tokens = patch_tokens[:, -self._num_spatial_patches :, :]
        if num_views > 1:
            patch_tokens = patch_tokens.reshape(batch_size, num_views * self._num_spatial_patches, -1)
        return patch_tokens

    def encode_pair(self, pair_pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode [current, future] pairs and return future patch targets."""
        if pair_pixel_values.ndim == 5:
            # [B, T, 3, H, W]
            batch_size, num_frames, channels, height, width = pair_pixel_values.shape
            num_views = 1
            clips = pair_pixel_values.permute(0, 2, 1, 3, 4)
        elif pair_pixel_values.ndim == 6:
            # [B, V, T, 3, H, W]
            batch_size, num_views, num_frames, channels, height, width = pair_pixel_values.shape
            clips = pair_pixel_values.reshape(batch_size * num_views, num_frames, channels, height, width)
            clips = clips.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(
                "Expected pair images [B, T, 3, H, W] or [B, V, T, 3, H, W], "
                f"got {tuple(pair_pixel_values.shape)}"
            )

        clips = self._as_released_clip(clips)
        patch_tokens = self._run_encoder(clips)
        expected = self.num_frames * self._num_spatial_patches
        if patch_tokens.shape[1] != expected:
            raise ValueError(
                f"Unexpected LeVJEPA pair token count: got {patch_tokens.shape[1]}, expected {expected}."
            )
        future_tokens = patch_tokens[:, -self._num_spatial_patches :, :]
        future_tokens = future_tokens.reshape(
            batch_size, num_views, 1, self.default_image_size // self.patch_size,
            self.default_image_size // self.patch_size, self._embed_dim
        )
        return future_tokens.detach()

    def get_fsdp_wrapping_policy(self) -> Callable:
        return partial(_module_wrap_policy, module_classes={self.featurizer.__class__})

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return (3, self.default_image_size, self.default_image_size)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_patches(self) -> int:
        return self._num_spatial_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
