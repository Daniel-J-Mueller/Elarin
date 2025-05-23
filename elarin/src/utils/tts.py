import numpy as np
import torch
from kokoro import KPipeline

class KokoroTTS:
    """Wrapper around the hexgrad/Kokoro-82M text-to-speech model."""

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        samplerate: int = 24000,
        voice: str = "kokoro",
    ) -> None:
        """Initialise the Kokoro TTS pipeline.

        Parameters
        ----------
        model_dir:
            Path to the downloaded ``hexgrad/Kokoro-82M`` model directory.
        device:
            PyTorch device identifier.
        samplerate:
            Target audio sample rate.
        voice:
            Voice preset to use when generating speech.
        """

        # The Kokoro pipeline expects ``model_dir`` as the first positional
        # argument rather than the ``model_path`` keyword used previously.
        # Passing the wrong keyword caused ``TypeError`` during startup. Some
        # versions also interpret the second positional argument as
        # ``lang_code`` which caused ``multiple values for lang_code`` errors
        # when specifying it by keyword. Supplying the language code
        # positionally avoids that issue and keeps the GPU device selectable.
        self.pipe = KPipeline(model_dir, "a", device=device)
        self.sample_rate = samplerate
        self.voice = voice

    @torch.no_grad()
    def synthesize(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(0, dtype=np.float32)
        chunks = []
        for _, _, audio in self.pipe(text, voice=self.voice):
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            chunks.append(audio)
        if chunks:
            audio_arr = np.concatenate(chunks)
        else:
            audio_arr = np.zeros(0, dtype=np.float32)
        return audio_arr.astype(np.float32)
