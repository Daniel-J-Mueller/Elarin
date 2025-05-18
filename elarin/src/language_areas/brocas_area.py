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

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def decode(self, hidden: torch.Tensor, temperature: float = 1.0) -> Iterable[str]:
        """Generate text from hidden state embeddings.

        ``temperature`` adjusts the sampling distribution.  A value of ``1.0``
        corresponds to the model's default logits while higher values yield more
        random output.  ``temperature`` must be > 0.
        """
        embeds = hidden.to(self.device)
        if embeds.dim() == 2:
            # ``GPT2LMHeadModel`` expects a sequence dimension. When a single
            # embedding is provided, add a length-1 dimension.
            embeds = embeds.unsqueeze(1)
        outputs = self.model(inputs_embeds=embeds)
        logits = outputs.logits / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        # ``multinomial`` expects a 2D tensor. Squeeze the sequence dimension for
        # single-token decoding.
        sample = torch.multinomial(probs.squeeze(1), num_samples=1)
        return self.tokenizer.batch_decode(sample, skip_special_tokens=True)

if __name__ == "__main__":
    area = BrocasArea("../models/gpt2")
    dummy = torch.zeros(1, 1, area.model.config.n_embd)
    print(area.decode(dummy))
