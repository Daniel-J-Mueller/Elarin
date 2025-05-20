"""Motor error correction approximating the cerebellum."""

import torch
from torch import nn


class Cerebellum(nn.Module):
    """Predict corrective adjustments for motor embeddings."""

    def __init__(
        self,
        vision_dim: int = 128,
        motor_dim: int = 768,
        hidden_dim: int = 256,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim + motor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, motor_dim),
        )
        self.device = device
        self.to(device)

    @torch.no_grad()
    def adjust(self, motor_emb: torch.Tensor, vision_feat: torch.Tensor) -> torch.Tensor:
        """Return adjusted motor embedding using visual feedback."""
        inp = torch.cat([motor_emb.to(self.device), vision_feat.to(self.device)], dim=-1)
        correction = self.net(inp)
        return motor_emb.to(self.device) + correction
