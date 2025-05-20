"""Emotional valence scoring approximating the amygdala."""

from __future__ import annotations

import torch
from torch import nn

from .utils.adapters import FatigueLoRA, LongTermLoRA


class Amygdala(nn.Module):
    """Assign a valence score to context embeddings."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 64, device: str = "cpu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.short_lora = FatigueLoRA(input_dim, 1, device=device)
        self.long_lora = LongTermLoRA(input_dim, 1, device=device)
        self.act = nn.Tanh()
        self.device = device
        self.to(device)

    @torch.no_grad()
    def evaluate(self, embedding: torch.Tensor) -> float:
        """Return valence in ``[-1, 1]`` for ``embedding``."""
        emb = embedding.to(self.device)
        val = self.net(emb) + self.short_lora(emb) + self.long_lora(emb)
        val = self.act(val)
        return float(val.squeeze())
