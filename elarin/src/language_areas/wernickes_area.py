import torch
from transformers import GPT2Tokenizer, GPT2Model
from typing import Iterable, Optional
import numpy as np
from pathlib import Path

class WernickesArea:
    """Front half of GPT-2 used for semantic encoding.

    The language modeling head is discarded so only hidden state
    embeddings are produced. Tokens are transient and never kept in
    memory after encoding.
    """

    def __init__(self, model_dir: str, device: str = "cpu", token_table_path: Optional[str] = None):
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

        self.token_table: Optional[torch.Tensor] = None
        if token_table_path:
            path = Path(token_table_path)
            if path.exists():
                data = np.load(path, allow_pickle=True).item()
                table = torch.tensor(data["embeddings"], dtype=torch.float32)
                self.token_table = table.to(device)

    @torch.no_grad()
    def encode(self, texts: Iterable[str]) -> torch.Tensor:
        """Return hidden state embeddings for given ``texts``.

        Tokens are immediately discarded after computing embeddings.
        """
        # ``max_length`` previously forced sequences to pad to 1024 tokens which
        # meant short inputs ended up dominated by ``pad_token`` embeddings.
        # That drowned out the actual content when selecting the last token.
        # Remove the explicit length so padding is only applied up to the
        # longest input in the batch.
        tokens = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        outputs = self.model(**tokens)
        hidden = outputs.last_hidden_state
        del tokens  # ensure tokens don't persist
        return hidden

    @torch.no_grad()
    def lookup_tokens(self, token_ids: Iterable[int]) -> torch.Tensor:
        """Return embeddings for ``token_ids`` from the precomputed table."""
        if self.token_table is None:
            raise ValueError("token table not loaded")
        ids = torch.tensor(list(token_ids), dtype=torch.long, device=self.device)
        embeds = self.token_table[ids]
        return embeds.unsqueeze(1)

