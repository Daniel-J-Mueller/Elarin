"""Online adaptation utilities for Elarin."""

from typing import Iterable

import torch
from torch import nn


class Trainer:
    """Very small placeholder trainer implementing simple Hebbian-like updates."""

    def __init__(self, lr: float = 1e-4, decay: float = 0.999, novelty_alpha: float = 0.9):
        self.lr = lr
        self.decay = decay
        self.novelty_alpha = novelty_alpha
        self._prev_activation: torch.Tensor | None = None

    def reset(self) -> None:
        """Clear any stored activation history."""
        self._prev_activation = None

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
        act = activations.squeeze(0)

        # Compute novelty based on similarity to recent activation
        novelty = 1.0
        if self._prev_activation is not None:
            prev = self._prev_activation.to(act.device)
            sim = torch.nn.functional.cosine_similarity(act, prev, dim=0)
            novelty = float(1.0 - sim.clamp(min=0.0).item())

            # Update running average of activation
            self._prev_activation.mul_(self.novelty_alpha)
            self._prev_activation.add_((1.0 - self.novelty_alpha) * act.cpu())
        else:
            self._prev_activation = act.cpu().clone()

        scaled_lr = self.lr * novelty
        for module in modules:
            for p in module.parameters():
                if not p.requires_grad:
                    continue

                p.mul_(self.decay)

                if p.ndim == 1:
                    length = min(p.shape[0], act.shape[0])
                    p[:length].add_(scaled_lr * act.to(p.device)[:length])
                else:
                    rows = min(p.shape[0], outer.shape[0])
                    cols = min(p.shape[1], outer.shape[1])
                    p[:rows, :cols].add_(
                        scaled_lr * outer.to(p.device)[:rows, :cols]
                    )

    @torch.no_grad()
    def align(self, modules: Iterable[nn.Module], target: torch.Tensor, actual: torch.Tensor) -> None:
        """Adjust parameters to make ``actual`` closer to ``target``.

        ``target`` and ``actual`` may originate from different devices
        (e.g. DMN vs. motor cortex).  ``actual`` is therefore moved to the
        device of ``target`` before computing the error so that operations
        succeed regardless of their source locations.
        """

        if target.device != actual.device:
            actual = actual.to(target.device)

        error = target - actual
        adjust = torch.einsum("bi,bj->ij", target, error)
        for module in modules:
            for p in module.parameters():
                if not p.requires_grad:
                    continue

                p.mul_(self.decay)

                if p.ndim == 1:
                    length = min(p.shape[0], error.shape[1])
                    grad = error.mean(dim=0).to(p.device)[:length]
                    p[:length].add_(self.lr * grad)
                else:
                    rows = min(p.shape[0], adjust.shape[0])
                    cols = min(p.shape[1], adjust.shape[1])
                    p[:rows, :cols].add_(self.lr * adjust.to(p.device)[:rows, :cols])


if __name__ == "__main__":
    # Minimal smoke test
    trainer = Trainer()
    lin = nn.Linear(4, 4)
    act = torch.randn(1, 4)
    trainer.step([lin], act)
    print("updated", lin.weight.norm().item())
