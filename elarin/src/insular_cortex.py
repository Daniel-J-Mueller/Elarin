"""Project motor embeddings into interoceptive space."""

from __future__ import annotations

import torch
from torch import nn


class InsularCortex(nn.Module):
    """Transforms motor cortex outputs before feedback."""

    def __init__(self, in_dim: int = 768, intero_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, intero_dim)
        self.act = nn.Tanh()

    @torch.no_grad()
    def forward(self, motor_emb: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(motor_emb))
