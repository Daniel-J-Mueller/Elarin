"""Cross-hemisphere communication bridge."""

from __future__ import annotations

import torch
from torch import nn

from .utils.adapters import FatigueLoRA, LongTermLoRA


class CorpusCallosum(nn.Module):
    """Placeholder pass-through layer approximating the corpus callosum."""

    def __init__(self, embed_dim: int = 768, device: str = "cpu") -> None:
        super().__init__()
        self.bridge = nn.Identity()
        self.short_lora = FatigueLoRA(embed_dim, embed_dim, device=device)
        self.long_lora = LongTermLoRA(embed_dim, embed_dim, device=device)
        self.device = device
        self.to(device)

    @torch.no_grad()
    def transfer(self, embedding: torch.Tensor) -> torch.Tensor:
        """Relay ``embedding`` across hemispheres."""
        emb = embedding.to(self.device)
        adj = self.short_lora(emb) + self.long_lora(emb)
        return self.bridge(emb + adj)
