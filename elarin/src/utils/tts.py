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

        # ``KPipeline`` has seen API changes across versions.  Newer releases
        # expect a language code as the first positional argument followed by
        # the model directory, while older versions simply took the directory
        # (optionally named ``model_dir``) and the device.  To support either
        # variant we inspect the constructor signature and call accordingly.
        import inspect

        kp_init = inspect.signature(KPipeline.__init__)
        params = kp_init.parameters

        if "lang_code" in params:
            # Newer API - language code first then model directory.
            self.pipe = KPipeline("a", model_dir=model_dir, device=device)
        else:
            # Older API - only the directory and device.
            self.pipe = KPipeline(model_dir, device=device)
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
