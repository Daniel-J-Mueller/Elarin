"""Action gating network."""

import torch
from torch import nn

from .utils.sentinel import SentinelLinear
from .subthalamic_nucleus import SubthalamicNucleus


class BasalGanglia(nn.Module):
    """Go/No-Go gating modulated by dopaminergic state."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 64,
        device: str = "cpu",
        axis: "HypothalamusPituitaryAxis | None" = None,
        prefrontal: "PrefrontalCortex | None" = None,
        stn: "SubthalamicNucleus | None" = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(input_dim, hidden_dim),
            nn.ReLU(),
            SentinelLinear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.device = device
        self.axis = axis
        self.prefrontal = prefrontal
        self.stn = stn
        self.to(device)

    @torch.no_grad()
    def gate(self, embedding: torch.Tensor) -> bool:
        prob = float(self.net(embedding.to(self.device)))
        # Modulate gating probability using hormone levels if available
        if self.axis is not None:
            mod = 0.5 * float(self.axis.dopamine) - 0.3 * float(self.axis.serotonin)
            prob += mod
        if self.prefrontal is not None:
            pf = float(self.prefrontal(embedding.to(self.prefrontal.device)))
            prob *= pf
        if self.stn is not None:
            prob *= 1.0 - float(self.stn.inhibition(embedding))
        prob = max(0.0, min(1.0, prob))
        return prob > 0.4
