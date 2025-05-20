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
        """Return encoder features for ``audio`` samples."""
        # ``audio`` may be provided on GPU but Whisper's feature extractor
        # operates on CPU numpy arrays. Convert each tensor appropriately.
        cpu_audio = [a.detach().cpu().numpy() for a in audio]
        inputs = self.processor(cpu_audio, return_tensors="pt", sampling_rate=16000)
        inputs = inputs.to(self.device)
        features = self.model.encoder(inputs.input_features).last_hidden_state
        return features
