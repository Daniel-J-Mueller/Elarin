"""Simplistic episodic memory with cross-modal associations."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class Hippocampus:
    """Episodic memory storing embeddings for multiple modalities."""

    def __init__(self, dims: Dict[str, int], capacity: int = 1000) -> None:
        self.dims = dims
        self.capacity = capacity
        # Each entry in ``memory`` is a mapping ``modality -> embedding``
        self.memory: List[Dict[str, np.ndarray]] = []

    def add_episode(self, episode: Dict[str, np.ndarray]) -> None:
        """Store a set of embeddings for different modalities."""
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        # Ensure all arrays are float32 for consistency
        clean = {m: emb.astype(np.float32) for m, emb in episode.items()}
        self.memory.append(clean)

    def query(self, modality: str, embedding: np.ndarray, k: int = 5) -> Dict[str, np.ndarray]:
        """Retrieve averaged embeddings from the closest episodes.

        Parameters
        ----------
        modality:
            The key used to compare against stored episodes.
        embedding:
            The query embedding of the same modality.
        """

        if not self.memory:
            return {}
        emb = embedding.astype(np.float32)
        scores = []
        for ep in self.memory:
            if modality not in ep:
                scores.append(-1.0)
                continue
            m = ep[modality]
            score = float(np.dot(emb, m) / (np.linalg.norm(emb) * np.linalg.norm(m) + 1e-8))
            scores.append(score)

        idx = np.argsort(scores)[-k:][::-1]
        collected: Dict[str, List[np.ndarray]] = {m: [] for m in self.dims}
        for i in idx:
            ep = self.memory[i]
            for m, val in ep.items():
                collected.setdefault(m, []).append(val)

        return {
            m: np.mean(vals, axis=0) for m, vals in collected.items() if len(vals) > 0
        }

    def decay(self, rate: float = 0.99) -> None:
        """Gradually weaken all stored embeddings."""
        for ep in self.memory:
            for m, val in ep.items():
                ep[m] = val * rate

    def clear(self) -> None:
        """Remove all stored episodes."""
        self.memory.clear()
