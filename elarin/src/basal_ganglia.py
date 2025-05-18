"""Action gating network."""

import torch
from torch import nn


class BasalGanglia(nn.Module):
    """Simple Go/No-Go gating using a small MLP."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 64, device: str = "cpu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.device = device
        self.to(device)

    @torch.no_grad()
    def gate(self, embedding: torch.Tensor) -> bool:
        prob = float(self.net(embedding.to(self.device)))
        return prob > 0.5
