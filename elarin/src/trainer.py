"""Online adaptation utilities for Elarin."""

from typing import Iterable

import torch
from torch import nn


class Trainer:
    """Very small placeholder trainer implementing simple Hebbian-like updates."""

    def __init__(self, lr: float = 1e-4, decay: float = 0.999):
        self.lr = lr
        self.decay = decay

    @torch.no_grad()
    def step(self, modules: Iterable[nn.Module], activations: torch.Tensor) -> None:
        """Apply a trivial Hebbian update to all adapter weights.

        Parameters
        ----------
        modules:
            Collection of modules containing parameters to update.
        activations:
            Activation tensor used to compute outer-product updates.
        """
        outer = torch.einsum("bi,bj->ij", activations, activations)
        for module in modules:
            for p in module.parameters():
                if p.requires_grad:
                    p.mul_(self.decay)
                    p.add_(self.lr * outer[: p.shape[0], : p.shape[1]])


if __name__ == "__main__":
    # Minimal smoke test
    trainer = Trainer()
    lin = nn.Linear(4, 4)
    act = torch.randn(1, 4)
    trainer.step([lin], act)
    print("updated", lin.weight.norm().item())
