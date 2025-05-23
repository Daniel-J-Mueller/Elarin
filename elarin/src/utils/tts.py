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

        # The Kokoro library changed the ``KPipeline`` constructor to expect
        # the language code first followed by the path to the model directory.
        # Passing the directory as the first positional argument now results in
        # it being treated as ``lang_code`` which triggers an assertion error.
        # To remain compatible with the updated API we explicitly provide the
        # language code (``"a"`` for American English) as the first argument and
        # pass ``model_dir`` second.  Using keywords keeps backwards
        # compatibility with older versions that still accept ``model_dir`` by
        # name while allowing the GPU device to be selected.
        self.pipe = KPipeline("a", model_dir=model_dir, device=device)
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
