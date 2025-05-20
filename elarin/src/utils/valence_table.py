#!/usr/bin/env python3
"""Generate valence phrase embeddings for quick lookup."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Dict, List

import numpy as np
import torch

# Allow running as a standalone tool or within the package
try:
    from .logger import get_logger
    from ..language_areas.wernickes_area import WernickesArea
except ImportError:  # pragma: no cover - fallback when executed directly
    from logger import get_logger
    from ..language_areas.wernickes_area import WernickesArea


DEFAULT_PHRASES: Dict[str, List[str]] = {
    "positive": [
        "well done",
        "that is great",
        "I like this",
        "excellent work",
        "pleased with that",
    ],
    "negative": [
        "that's bad",
        "I dislike this",
        "not good",
        "terrible idea",
        "stop that",
    ],
}


def generate(model_dir: str | Path, output: Path, device: str = "cpu",
             phrases: Dict[str, List[str]] | None = None) -> None:
    """Create ``output`` containing embeddings for ``phrases``."""
    phrases = phrases or DEFAULT_PHRASES
    logger = get_logger("valence_table")

    wa = WernickesArea(str(model_dir), device=device)

    data: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for label, texts in phrases.items():
            emb = wa.encode(texts).mean(dim=1)
            data[label] = emb.cpu().numpy().astype(np.float32)
            logger.info(f"encoded {len(texts)} {label} phrases")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, data, allow_pickle=True)
    logger.info(f"saved valence table to {output}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate valence phrase table")
    parser.add_argument("--model_dir", default="models/gpt2")
    parser.add_argument("--output", default="persistent/valence.npy")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    generate(args.model_dir, Path(args.output), device=args.device)


if __name__ == "__main__":  # pragma: no cover - CLI usage
    main()
