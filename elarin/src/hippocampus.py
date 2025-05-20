"""Simplistic episodic memory with cross-modal associations."""

from __future__ import annotations

from typing import Dict, List, Optional

from pathlib import Path

import numpy as np


class Hippocampus:
    """Episodic memory storing embeddings for multiple modalities."""

    def __init__(self, dims: Dict[str, int], capacity: int = 1000, persist_path: Optional[str] = None) -> None:
        self.dims = dims
        self.capacity = capacity
        # Each entry is a mapping ``modality -> embedding`` plus optional ``valence``
        self.memory: List[Dict[str, np.ndarray | float]] = []
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            try:
                self.memory = np.load(self.persist_path, allow_pickle=True).tolist()
            except Exception:
                self.memory = []

    def add_episode(self, episode: Dict[str, np.ndarray], valence: float = 0.0) -> None:
        """Store a set of embeddings for different modalities with a valence tag."""
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        # Ensure all arrays are float32 for consistency
        clean = {}
        for m, emb in episode.items():
            if m in self.dims and emb.shape[-1] != self.dims[m]:
                continue
            clean[m] = emb.astype(np.float32)
        clean["valence"] = float(valence)
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
            if m.ndim > 1:
                m = m.mean(axis=0)
            if m.shape[0] != emb.shape[0]:
                scores.append(-1.0)
                continue
            score = float(
                np.dot(emb, m)
                / (np.linalg.norm(emb) * np.linalg.norm(m) + 1e-8)
            )
            scores.append(score)

        idx = np.argsort(scores)[-k:][::-1]
        collected: Dict[str, List[np.ndarray]] = {m: [] for m in self.dims}
        valences: List[float] = []
        for i in idx:
            ep = self.memory[i]
            for m, val in ep.items():
                if m == "valence":
                    valences.append(float(val))
                    continue
                if val.ndim > 1:
                    val = val.mean(axis=0)
                collected.setdefault(m, []).append(val)

        result = {
            m: np.mean(vals, axis=0) for m, vals in collected.items() if len(vals) > 0
        }
        if valences:
            result["valence"] = float(np.mean(valences))
        return result

    def decay(self, rate: float = 0.99) -> None:
        """Gradually weaken all stored embeddings."""
        for ep in self.memory:
            for m, val in ep.items():
                if m == "valence":
                    ep[m] = float(val) * rate
                else:
                    ep[m] = val * rate

    def clear(self) -> None:
        """Remove all stored episodes."""
        self.memory.clear()

    def save(self) -> None:
        """Persist memory to disk if ``persist_path`` is set."""
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.persist_path, np.array(self.memory, dtype=object), allow_pickle=True)
