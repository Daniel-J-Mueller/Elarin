"""Multimodal fusion hub approximating the brain's DMN."""

from typing import Dict

import torch
from torch import nn


class DefaultModeNetwork(nn.Module):
    """Fuse sensory embeddings and produce routed context vectors."""

    def __init__(self, vision_dim: int = 128, audio_dim: int = 128, intero_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        fusion_in = vision_dim + audio_dim + intero_dim
        self.fusion = nn.Linear(fusion_in, hidden_dim)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @torch.no_grad()
    def forward(self, vision: torch.Tensor, audio: torch.Tensor, intero: torch.Tensor) -> torch.Tensor:
        """Return fused context embedding."""
        combined = torch.cat([vision, audio, intero], dim=-1)
        hidden = torch.relu(self.fusion(combined))
        return self.router(hidden)
