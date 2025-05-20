"""Emotional valence scoring approximating the amygdala."""

from __future__ import annotations

import torch
from torch import nn


class Amygdala(nn.Module):
    """Assign a valence score to context embeddings."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 64, device: str = "cpu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.device = device
        self.to(device)

    @torch.no_grad()
    def evaluate(self, embedding: torch.Tensor) -> float:
        """Return valence in ``[-1, 1]`` for ``embedding``."""
        val = self.net(embedding.to(self.device))
        return float(val.squeeze())
