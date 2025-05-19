#!/usr/bin/env python3
"""Utility for generating token embeddings from Wernicke's Area."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm

# Resolve the base directory of the ``elarin`` package so relative paths work
# regardless of the current working directory.
BASE_DIR = Path(__file__).resolve().parents[2]


def generate(
    model_dir: str | Path,
    output: Path,
    device: str = "cpu",
    batch_size: int = 1024,
) -> None:
    """Create a table of embeddings for every tokenizer token."""
    model_path = Path(model_dir)
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path

    output_path = output
    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2Model.from_pretrained(model_path)

    if device.startswith("cuda"):
        device_ids = [0, 1, 2, 3]
        if torch.cuda.device_count() < len(device_ids):
            device_ids = list(range(torch.cuda.device_count()))
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    model.eval()

    embeddings = np.zeros((len(tokenizer), model.config.n_embd), dtype=np.float32)
    tokens = [tokenizer.decode([i], skip_special_tokens=False) for i in range(len(tokenizer))]

    with torch.no_grad():
        for start in tqdm(range(0, len(tokenizer), batch_size), desc="Embedding tokens"):
            end = min(start + batch_size, len(tokenizer))
            ids = torch.arange(start, end, device=device).unsqueeze(1)
            out = model(input_ids=ids)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
            embeddings[start:end] = emb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, {"tokens": tokens, "embeddings": embeddings}, allow_pickle=True)
    print(f"saved {len(tokens)} embeddings to {output_path}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Wernicke token table")
    parser.add_argument("--model_dir", default="models/gpt2")
    parser.add_argument("--output", default="elarin/persistent/token_embeddings.npy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args(argv)

    generate(args.model_dir, Path(args.output), device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

