"""Transform motor outputs into interoceptive feedback."""

import torch
from torch import nn


class InsularCortex(nn.Module):
    """Projects motor embeddings into a damped interoceptive space."""

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256, device: str = "cpu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(self.device))
