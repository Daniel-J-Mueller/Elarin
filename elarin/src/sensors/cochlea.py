"""Microphone audio encoder using Whisper."""

from typing import Iterable

import torch
from transformers import WhisperProcessor, WhisperModel


class Cochlea:
    """Stream audio into Whisper to obtain embeddings."""

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperModel.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, audio: Iterable[torch.Tensor]) -> torch.Tensor:
        # ``audio`` is expected to be an iterable of 1D tensors (samples)
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16000).to(self.device)
        features = self.model.encoder(inputs.input_features).last_hidden_state
        return features
