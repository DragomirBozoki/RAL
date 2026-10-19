"""Signal levels shared by echo control, VAD and barge-in."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["SILENCE_DB", "db_to_gain", "rms_db"]

SILENCE_DB = -120.0
"""Level reported for digital silence, instead of minus infinity."""


def rms_db(samples: npt.NDArray[Any]) -> float:
    """RMS level of float samples in dBFS (a full-scale sine is -3 dBFS)."""
    if samples.size == 0:
        return SILENCE_DB
    power = float(np.mean(np.square(samples, dtype=np.float64)))
    if power <= 1e-12:
        return SILENCE_DB
    return max(SILENCE_DB, 10.0 * float(np.log10(power)))


def db_to_gain(db: float) -> float:
    """Amplitude factor for a level change in dB."""
    return float(10.0 ** (db / 20.0))
