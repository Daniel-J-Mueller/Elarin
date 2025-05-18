import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Iterable

class BrocasArea:
    """Simple wrapper around GPT-2 for text generation.

    This module represents a language motor region. It expects hidden
    state embeddings from upstream modules (e.g. :class:`WernickesArea`)
    and produces plain text output. The model weights are frozen; online
    training should be performed via LoRA adapters as described in
    ``AGENTS.md``.
    """

    def __init__(self, model_dir: str):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.eval()

    @torch.no_grad()
    def decode(self, hidden: torch.Tensor) -> Iterable[str]:
        """Generate text from hidden state embeddings."""
        outputs = self.model(inputs_embeds=hidden)
        ids = torch.argmax(outputs.logits, dim=-1)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

if __name__ == "__main__":
    area = BrocasArea("../models/gpt2")
    dummy = torch.zeros(1, 1, area.model.config.n_embd)
    print(area.decode(dummy))
