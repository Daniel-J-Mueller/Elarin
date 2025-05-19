#!/usr/bin/env python3
"""Utility for generating token embeddings from Wernicke's Area."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model

# Resolve the base directory of the ``elarin`` package so relative paths work
# regardless of the current working directory.
BASE_DIR = Path(__file__).resolve().parents[2]


def generate(model_dir: str | Path, output: Path, device: str = "cpu") -> None:
    """Create a table of embeddings for every tokenizer token."""
    model_path = Path(model_dir)
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path

    output_path = output
    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2Model.from_pretrained(model_path)
    model.to(device)
    model.eval()

    embeddings = np.zeros((len(tokenizer), model.config.n_embd), dtype=np.float32)
    tokens: list[str] = []

    with torch.no_grad():
        for tok_id in range(len(tokenizer)):
            ids = torch.tensor([[tok_id]], device=device)
            out = model(input_ids=ids)
            emb = out.last_hidden_state.squeeze(0).squeeze(0).cpu().numpy()
            embeddings[tok_id] = emb.astype(np.float32)
            tokens.append(tokenizer.decode([tok_id], skip_special_tokens=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, {"tokens": tokens, "embeddings": embeddings}, allow_pickle=True)
    print(f"saved {len(tokens)} embeddings to {output_path}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Wernicke token table")
    parser.add_argument("--model_dir", default="models/gpt2")
    parser.add_argument("--output", default="elarin/persistent/token_embeddings.npy")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    generate(args.model_dir, Path(args.output), device=args.device)


if __name__ == "__main__":
    main()

