import torch
from transformers import GPT2Tokenizer, GPT2Model
from typing import Iterable

class WernickesArea:
    """Front half of GPT-2 used for semantic encoding.

    The language modeling head is discarded so only hidden state
    embeddings are produced. Tokens are transient and never kept in
    memory after encoding.
    """

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        # ``GPT2Tokenizer`` does not define a padding token by default which
        # breaks batching when ``padding=True`` is requested.  Use the EOS token
        # as padding to keep sequence length consistent across calls.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2Model.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts: Iterable[str]) -> torch.Tensor:
        """Return hidden state embeddings for given ``texts``.

        Tokens are immediately discarded after computing embeddings.
        """
        tokens = self.tokenizer(list(texts), return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**tokens)
        hidden = outputs.last_hidden_state
        del tokens  # ensure tokens don't persist
        return hidden
