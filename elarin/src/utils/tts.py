#!/usr/bin/env python3
"""
tts.py – Fully local Kokoro-82M loader for Elarin

This wrapper:
  1) Patches huggingface_hub.hf_hub_download so local folders work for ANY file.
  2) Patches kokoro.model.KModel.MODEL_NAMES to accept your folder.
  3) Calls KPipeline("a", model_dir, device) so there are no unexpected kwargs.
"""

import os
import numpy as np
import torch

# 1) Monkey-patch HF Hub so passing a local folder just constructs a local path.
import huggingface_hub
from huggingface_hub import hf_hub_download as _real_hf_hub_download

def _local_hf_hub_download(repo_id: str, filename: str, *args, **kwargs) -> str:
    # If repo_id is a directory, return the local file path (config.json, voices/*.pt, etc.)
    if os.path.isdir(repo_id):
        return os.path.join(repo_id, filename)
    # Otherwise fall back to the real HF downloader
    return _real_hf_hub_download(repo_id, filename, *args, **kwargs)

# Override both entrypoints
huggingface_hub.hf_hub_download = _local_hf_hub_download
try:
    import kokoro.model as _kokoro_model
    _kokoro_model.hf_hub_download = _local_hf_hub_download
except ImportError:
    pass

# 2) Patch KModel.MODEL_NAMES so your folder is recognized
from kokoro.model import KModel
DEFAULT_REPO = "hexgrad/Kokoro-82M"
def _patch_model_names(model_dir: str):
    if DEFAULT_REPO in KModel.MODEL_NAMES:
        KModel.MODEL_NAMES[model_dir] = KModel.MODEL_NAMES[DEFAULT_REPO]

# 3) Import the pipeline
from kokoro import KPipeline

class KokoroTTS:
    """Local-only Kokoro-82M TTS loader."""

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        samplerate: int = 24000,
        voice: str = "af_jessica",
    ) -> None:
        """
        model_dir : str
            Path to your local `kokoro-82m` folder (e.g. "/home/daniel/.../models/kokoro-82m").
        device : str
            Torch device specifier, e.g. "cuda:3" or "cpu".
        samplerate : int
            Output sample rate (defaults to 24000).
        voice : str
            Voice preset (defaults to "kokoro").
        """
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Kokoro model dir not found: {model_dir}")

        # Let KModel know about this path
        _patch_model_names(model_dir)

        # Positional init: (lang_code, repo_id/model_dir, device)
        self.pipe = KPipeline("a", model_dir, device)
        self.sample_rate = samplerate
        self.voice = voice

    @torch.no_grad()
    def synthesize(self, text: str) -> np.ndarray:
        """
        Convert `text` → speech. Returns a NumPy float32 array
        of raw PCM samples at self.sample_rate.
        """
        if not text:
            return np.zeros(0, dtype=np.float32)

        chunks = []
        for *_, audio in self.pipe(text, voice=self.voice):
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            chunks.append(audio)

        if chunks:
            out = np.concatenate(chunks)
        else:
            out = np.zeros(0, dtype=np.float32)
        return out.astype(np.float32)
