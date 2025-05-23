import threading
from typing import Iterable
import numpy as np
import sounddevice as sd


def play_audio(audio: Iterable[float] | np.ndarray, samplerate: int = 16000) -> None:
    """Play ``audio`` asynchronously using ``sounddevice``."""

    arr = np.array(list(audio), dtype=np.float32).flatten()
    if arr.size == 0:
        return

    def _play() -> None:
        sd.play(arr, samplerate=samplerate)
        sd.wait()

    threading.Thread(target=_play, daemon=True).start()
