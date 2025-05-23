import numpy as np
import torch
from transformers import pipeline

class KokoroTTS:
    """Simple wrapper around the hexgrad/Kokoro-82M text-to-speech model."""

    def __init__(self, model_dir: str, device: str = "cpu", samplerate: int = 16000) -> None:
        device_idx = 0 if device.startswith("cuda") else -1
        self.pipe = pipeline("text-to-speech", model=model_dir, device=device_idx)
        self.sample_rate = samplerate

    @torch.no_grad()
    def synthesize(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(0, dtype=np.float32)
        output = self.pipe(text)
        audio = output["audio"]
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        return audio.astype(np.float32)
