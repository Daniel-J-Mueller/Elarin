"""Executive gating influences approximating the prefrontal cortex."""

from __future__ import annotations

import torch
from torch import nn


class PrefrontalCortex(nn.Module):
    """Simple network producing gating modulation factors."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 128, device: str = "cpu") -> None:
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
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Return modulation factor in ``[0,1]`` for ``context``."""
        return self.net(context.to(self.device))
