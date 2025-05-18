"""Novelty and importance detector."""

import torch
from torch import nn


class SalienceCortex(nn.Module):
    """Compute a salience score for fused embeddings."""

    def __init__(self, input_dim: int = 256, device: str = "cpu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.device = device
        self.to(device)

    @torch.no_grad()
    def score(self, embedding: torch.Tensor) -> float:
        val = self.net(embedding.to(self.device))
        return float(val)
