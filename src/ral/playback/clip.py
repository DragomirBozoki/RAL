"""Loading audio clips for playback (earcons, test sentences, demos)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ral.core.errors import FormatError
from ral.dsp.resample import Resampler

__all__ = ["load_wav"]


def load_wav(path: str | Path, rate: int = 16_000) -> npt.NDArray[np.float32]:
    """A 16-bit PCM WAV file as mono float32 at ``rate`` (channels averaged).

    Raises:
        FormatError: If the file is not 16-bit PCM.
    """
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise FormatError(f"{path} is not 16-bit PCM; convert it first.")
        src_rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, channels)
    mono = (pcm.astype(np.float32) / 32768.0).mean(axis=1, keepdims=True, dtype=np.float32)
    if src_rate == rate:
        same: npt.NDArray[np.float32] = mono[:, 0]
        return same
    r = Resampler(src_rate, rate, 1)
    out = np.concatenate([r.process(mono), r.flush()])
    return np.ascontiguousarray(out[:, 0], dtype=np.float32)
