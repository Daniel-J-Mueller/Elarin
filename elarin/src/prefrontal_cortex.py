"""Executive gating influences approximating the prefrontal cortex."""

from __future__ import annotations

import torch
from torch import nn

from .utils.adapters import FatigueLoRA, LongTermLoRA
from .utils.sentinel import SentinelLinear


class PrefrontalCortex(nn.Module):
    """Network modelling executive gating over sensations and actions."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 128, device: str = "cpu") -> None:
        super().__init__()
        # Single value that modulates Go/No-Go probability
        self.action_net = nn.Sequential(
            SentinelLinear(input_dim, hidden_dim),
            nn.ReLU(),
            SentinelLinear(hidden_dim, 1),
        )
        # Per-modality filter weights [vision, audio, intero]
        self.filter_net = nn.Sequential(
            SentinelLinear(input_dim, hidden_dim),
            nn.ReLU(),
            SentinelLinear(hidden_dim, 3),
        )
        self.short_lora = FatigueLoRA(input_dim, 1, device=device)
        self.long_lora = LongTermLoRA(input_dim, 1, device=device)
        self.act = nn.Sigmoid()
        self.device = device
        self.to(device)

    @torch.no_grad()
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Return action modulation factor in ``[0,1]`` for ``context``."""
        ctx = context.to(self.device)
        out = self.action_net(ctx)
        out = out + self.short_lora(ctx) + self.long_lora(ctx)
        return self.act(out)

    @torch.no_grad()
    def filter_weights(self, context: torch.Tensor) -> dict[str, float]:
        """Return per-modality gating weights in ``[0,1]``."""
        ctx = context.to(self.device)
        w = self.act(self.filter_net(ctx)).squeeze(0)
        return {"vision": float(w[0]), "audio": float(w[1]), "intero": float(w[2])}
