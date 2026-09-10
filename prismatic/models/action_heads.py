"""Visual-token alignment head used by the released JEPA-WAM objective."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualTokenCosineHead(nn.Module):
    def __init__(self, d_llm: int, d_target: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_llm, 2 * d_target, bias=True)
        self.act_fn1 = nn.GELU()
        self.fc2 = nn.Linear(2 * d_target, d_target, bias=True)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def align_dimension(self, llm_embedding: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn1(self.fc1(llm_embedding)))

    @staticmethod
    def compute_align_loss_cosine(
        prediction: torch.Tensor,
        target: torch.Tensor,
        time_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction = F.normalize(prediction, dim=-1)
        target = F.normalize(target, dim=-1)
        distances = 1 - (prediction * target).sum(dim=-1)
        if time_valid_mask is None:
            return distances.mean()
        if distances.ndim < 3 or tuple(time_valid_mask.shape) != tuple(distances.shape[:2]):
            raise ValueError(
                "Temporal visual alignment expects time_valid_mask [B, T] and distances [B, T, ...]."
            )
        mask = time_valid_mask.to(dtype=distances.dtype)
        for _ in range(distances.ndim - 2):
            mask = mask.unsqueeze(-1)
        return (distances * mask).sum() / mask.expand_as(distances).sum().clamp_min(1)

    def forward(
        self,
        llm_emb: torch.Tensor,
        target_emb: torch.Tensor,
        time_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.align_dimension(llm_emb)
        return self.compute_align_loss_cosine(projected, target_emb.detach(), time_valid_mask), projected
