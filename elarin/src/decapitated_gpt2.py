import torch
from transformers import GPT2Tokenizer, GPT2Model
from typing import Iterable

class DecapitatedGPT2:
    """GPT-2 encoder that exposes only hidden states.

    The LM head is discarded so no token logits are produced.
    Text is tokenized only transiently and never stored.
    """

    def __init__(self, model_dir: str):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2Model.from_pretrained(model_dir)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: Iterable[str]) -> torch.Tensor:
        """Return hidden state embeddings for given ``texts``.

        Tokens are immediately discarded after computing embeddings.
        """
        tokens = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        outputs = self.model(**tokens)
        hidden = outputs.last_hidden_state
        del tokens  # ensure tokens don't persist
        return hidden
