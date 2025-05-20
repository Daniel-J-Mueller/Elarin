"""Action gating network."""

import torch
from torch import nn


class BasalGanglia(nn.Module):
    """Go/No-Go gating modulated by dopaminergic state."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 64,
        device: str = "cpu",
        axis: "HypothalamusPituitaryAxis | None" = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.device = device
        self.axis = axis
        self.to(device)

    @torch.no_grad()
    def gate(self, embedding: torch.Tensor) -> bool:
        prob = float(self.net(embedding.to(self.device)))
        # Modulate gating probability using hormone levels if available
        if self.axis is not None:
            mod = 0.5 * float(self.axis.dopamine) - 0.3 * float(self.axis.serotonin)
            prob += mod
        prob = max(0.0, min(1.0, prob))
        return prob > 0.4
