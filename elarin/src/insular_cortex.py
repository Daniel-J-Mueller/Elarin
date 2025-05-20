"""Project motor embeddings into interoceptive space."""

from __future__ import annotations

import torch
from torch import nn
from pathlib import Path


class FatigueLoRA(nn.Module):
    """LoRA adapter with short-term fatigue dynamics."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        r: int = 4,
        decay: float = 0.9,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.zeros(r, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        self.register_buffer("fatigue", torch.ones(out_dim))
        self.register_buffer(
            "recovery", torch.rand(out_dim) * 0.01 + 0.005
        )
        self.decay = decay
        self.r = r
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (x @ self.A.t()) @ self.B.t() / max(1, self.r)
        out = out * self.fatigue.view(1, -1)
        with torch.no_grad():
            usage = out.abs().mean(dim=0).view(-1)
            decay_factor = torch.pow(torch.as_tensor(self.decay, device=usage.device), usage)
            self.fatigue.mul_(decay_factor)
            self.fatigue.add_(self.recovery)
            self.fatigue.clamp_(0.0, 1.0)
        return out


class LongTermLoRA(nn.Module):
    """Simple LoRA adapter for long-term filtering."""

    def __init__(self, in_dim: int, out_dim: int, r: int = 4, device: str = "cpu") -> None:
        super().__init__()
        self.A = nn.Parameter(torch.zeros(r, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        self.r = r
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A.t()) @ self.B.t() / max(1, self.r)


class InsularCortex(nn.Module):
    """Transforms motor cortex outputs before feedback."""

    def __init__(
        self,
        in_dim: int = 768,
        intero_dim: int = 768,
        device: str = "cpu",
        persist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, intero_dim)
        self.short_lora = FatigueLoRA(in_dim, intero_dim, device=device)
        self.long_lora = LongTermLoRA(in_dim, intero_dim, device=device)
        self.act = nn.Tanh()
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            state = torch.load(self.persist_path, map_location=device)
            self.load_state_dict(state)

    @torch.no_grad()
    def forward(self, motor_emb: torch.Tensor) -> torch.Tensor:
        motor_emb = motor_emb.to(self.device)
        base = self.proj(motor_emb)
        out = base + self.short_lora(motor_emb) + self.long_lora(motor_emb)
        return self.act(out)

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
