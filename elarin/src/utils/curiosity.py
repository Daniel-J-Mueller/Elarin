"""Track novelty of produced tokens for intrinsic motivation."""

from __future__ import annotations

from typing import Dict


class CuriosityTracker:
    """Count token usage and provide novelty bonuses."""

    def __init__(self) -> None:
        self.counts: Dict[int, int] = {}

    def bonus(self, token_id: int) -> float:
        """Return intrinsic bonus for ``token_id`` based on how rarely it was used."""
        count = self.counts.get(token_id, 0)
        return 1.0 / (1.0 + float(count))

    def update(self, token_id: int) -> None:
        """Record another occurrence of ``token_id``."""
        self.counts[token_id] = self.counts.get(token_id, 0) + 1

    def state_dict(self) -> Dict[int, int]:
        return dict(self.counts)

    def load_state_dict(self, state: Dict[int, int]) -> None:
        self.counts = dict(state)
