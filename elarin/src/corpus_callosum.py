"""Cross-hemisphere communication bridge."""

from __future__ import annotations

import torch
from torch import nn


class CorpusCallosum(nn.Module):
    """Placeholder pass-through layer approximating the corpus callosum."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.bridge = nn.Identity()
        self.device = device
        self.to(device)

    @torch.no_grad()
    def transfer(self, embedding: torch.Tensor) -> torch.Tensor:
        """Relay ``embedding`` across hemispheres."""
        return self.bridge(embedding.to(self.device))
