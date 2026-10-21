"""Test signals for ``ral probe``: exponential sine sweep and its inverse filter.

The exponential (logarithmic) sweep spends equal time per octave. Convolving the
recording with the time-reversed sweep, attenuated by 6 dB per octave to undo the sweep's
pink spectrum, yields the system's impulse response with harmonic distortion pushed to
negative times, where it does not disturb the linear response.

Reference: A. Farina, "Simultaneous measurement of impulse response and distortion with a
swept-sine technique", 108th AES Convention, 2000.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ral.bench.synthetic import fft_convolve

__all__ = ["deconvolve", "exp_sweep", "inverse_filter"]

Signal = npt.NDArray[np.float64]


def exp_sweep(
    rate: int,
    seconds: float = 2.0,
    f1: float = 100.0,
    f2: float | None = None,
    level_db: float = -20.0,
) -> Signal:
    """An exponential sine sweep from ``f1`` to ``f2`` with 10 ms fades at both ends."""
    f2 = f2 or 0.45 * rate
    n = int(seconds * rate)
    t = np.arange(n) / rate
    k = np.log(f2 / f1)
    x = np.sin(2 * np.pi * f1 * seconds / k * (np.exp(t * k / seconds) - 1.0))
    fade = min(n // 4, int(0.01 * rate))
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(fade) / fade))
    x[:fade] *= ramp
    x[n - fade :] *= ramp[::-1]
    return np.asarray(x * 10 ** (level_db / 20) * np.sqrt(2), dtype=np.float64)


def inverse_filter(sweep: Signal, rate: int, f1: float = 100.0, f2: float | None = None) -> Signal:
    """Time-reversed sweep with a -6 dB/octave envelope, scaled so ``sweep * inv`` peaks at 1."""
    f2 = f2 or 0.45 * rate
    n = len(sweep)
    seconds = n / rate
    t = np.arange(n) / rate
    envelope = np.exp(-t * np.log(f2 / f1) / seconds)
    inv = sweep[::-1] * envelope
    peak = float(np.max(np.abs(fft_convolve(np.concatenate([sweep, np.zeros(n)]), inv))))
    return np.asarray(inv / peak, dtype=np.float64)


def deconvolve(recording: Signal, inv: Signal) -> Signal:
    """Impulse response; index 0 is the first sample of the sweep's playback."""
    full = fft_convolve(np.concatenate([recording, np.zeros(len(inv))]), inv)
    return np.asarray(full[len(inv) - 1 : len(inv) - 1 + len(recording)], dtype=np.float64)
