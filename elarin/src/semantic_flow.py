"""Associative token flow for coarse semantic sequencing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


class SemanticFlow:
    """Store probabilities of token transitions.

    Each token is referenced by its index from the token table.  Observed
    sequences update transition counts which are later normalized when
    retrieving probabilities or sampling the next token.
    """

    def __init__(self, vocab_size: int, persist_path: Optional[str] = None) -> None:
        self.vocab_size = vocab_size
        self.transitions: Dict[int, Dict[int, float]] = {}
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.transitions = {
                int(k): {int(k2): float(v2) for k2, v2 in v.items()} for k, v in data.items()
            }

    def observe(self, token_ids: Iterable[int]) -> None:
        """Record transitions observed in ``token_ids``."""
        iterator = iter(token_ids)
        prev = next(iterator, None)
        for idx in iterator:
            if prev is None:
                prev = idx
                continue
            dest = self.transitions.setdefault(int(prev), {})
            dest[idx] = dest.get(idx, 0.0) + 1.0
            prev = idx

    def next_probabilities(self, idx: int) -> Dict[int, float]:
        """Return normalized probabilities of tokens following ``idx``."""
        dest = self.transitions.get(int(idx))
        if not dest:
            return {}
        total = float(sum(dest.values()))
        if total <= 0.0:
            return {}
        return {i: count / total for i, count in dest.items()}

    def sample_next(self, idx: int, temperature: float = 1.0) -> int | None:
        """Sample the next token index after ``idx``."""
        probs = self.next_probabilities(idx)
        if not probs:
            return None
        tokens = list(probs.keys())
        weights = np.array(list(probs.values()), dtype=np.float64)
        if temperature != 1.0:
            weights = weights ** (1.0 / max(temperature, 1e-5))
            weights = weights / weights.sum()
        choice = int(np.random.choice(tokens, p=weights))
        return choice

    def save(self, path: str | None = None) -> None:
        """Persist the transition table to ``path`` or ``persist_path``."""
        target = Path(path) if path else self.persist_path
        if not target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.transitions, f)
