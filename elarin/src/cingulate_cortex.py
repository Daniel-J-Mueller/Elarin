"""Emotional relay approximating the cingulate cortex."""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from .utils.sentinel import SentinelLinear
from .utils.adapters import FatigueLoRA, LongTermLoRA


class CingulateCortex(nn.Module):
    """Translate pain and emotion into modulation signals."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        device: str = "cpu",
        persist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(input_dim, hidden_dim),
            nn.ReLU(),
            SentinelLinear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.short_lora = FatigueLoRA(input_dim, 1, device=device)
        self.long_lora = LongTermLoRA(input_dim, 1, device=device)
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self.load_state_dict(torch.load(self.persist_path, map_location=device))

    @torch.no_grad()
    def modulate(self, pain: torch.Tensor) -> float:
        val = pain.view(1, -1).to(self.device)
        out = self.net(val) + self.short_lora(val) + self.long_lora(val)
        return float(out.squeeze())

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
