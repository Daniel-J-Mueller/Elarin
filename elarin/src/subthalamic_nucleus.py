"""Decision inertia mechanism delaying impulsive actions."""

from __future__ import annotations

import torch
from torch import nn

from .utils.sentinel import SentinelLinear


class SubthalamicNucleus(nn.Module):
    """Predict inhibitory factor based on current context."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 64, device: str = "cpu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(input_dim, hidden_dim),
            nn.ReLU(),
            SentinelLinear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.device = device
        self.to(device)

    @torch.no_grad()
    def inhibition(self, context: torch.Tensor) -> float:
        """Return a value in ``[0,1]`` describing how much to slow actions."""
        ctx = context.to(self.device)
        return float(self.net(ctx))
