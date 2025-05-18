"""Minimal episodic memory using cosine similarity."""

from typing import List

import numpy as np


class Hippocampus:
    """In-memory list of embeddings with naive retrieval."""

    def __init__(self, dim: int, capacity: int = 1000):
        self.dim = dim
        self.capacity = capacity
        self.memory: List[np.ndarray] = []

    def add(self, embedding: np.ndarray) -> None:
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(embedding.astype(np.float32))

    def query(self, embedding: np.ndarray, k: int = 5) -> List[np.ndarray]:
        if not self.memory:
            return []
        emb = embedding.astype(np.float32)
        scores = [float(np.dot(emb, m) / (np.linalg.norm(emb) * np.linalg.norm(m) + 1e-8)) for m in self.memory]
        idx = np.argsort(scores)[-k:][::-1]
        return [self.memory[i] for i in idx]
