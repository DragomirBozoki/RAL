"""Synthetic echo scenes for tests and benchmarks without a robot (DESIGN section 8).

A scene is a far-end signal (what the robot plays), its echo at the mic (the far end
convolved with a delayed, decaying impulse response), a near-end talker and a noise
floor. Everything is deterministic for a given seed, so tests can assert on numbers.

The voices are synthetic (harmonics with gliding pitch, formant weighting, syllable and
word gating). They exercise echo control and energy-based detectors; speech models such
as Silero VAD need real speech clips instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["EchoScene", "echo_path", "echo_scene", "fft_convolve", "voice"]

Signal = npt.NDArray[np.float64]
Span = tuple[float, float]


def fft_convolve(x: Signal, h: Signal) -> Signal:
    """Linear convolution of ``x`` with ``h``, truncated to ``len(x)``."""
    n = len(x) + len(h) - 1
    size = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(h, size), size)
    return np.asarray(y[: len(x)], dtype=np.float64)


def _gate(
    n: int, rate: int, rng: np.random.Generator, on: Span, off: Span, ramp_s: float = 0.04
) -> Signal:
    """Word-like on/off gating with raised-cosine ramps."""
    gate = np.zeros(n)
    i = 0
    while i < n:
        length = int(rng.uniform(*on) * rate)
        gate[i : i + length] = 1.0
        i += length + int(rng.uniform(*off) * rate)
    ramp = np.hanning(max(3, int(ramp_s * rate)))
    return np.convolve(gate, ramp / ramp.sum(), mode="same")


def voice(n: int, rate: int, f0: float, seed: int) -> Signal:
    """A voice-like signal with unit RMS: harmonic, formant-shaped, syllables and words."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate
    win = max(1, rate // 5)
    drift = np.convolve(rng.standard_normal(n), np.ones(win) / win, mode="same")
    pitch = f0 * (1 + 0.12 * np.sin(2 * np.pi * 0.6 * t + rng.uniform(0, 6)) + 3 * drift)
    phase = 2 * np.pi * np.cumsum(pitch) / rate
    formants = (rng.uniform(450, 750), rng.uniform(1100, 1800), rng.uniform(2300, 2900))
    x: Signal = np.zeros(n)
    for h in range(1, 40):
        freq = h * pitch
        weight = 0.05 + sum(np.exp(-(((freq - f) / 180.0) ** 2)) for f in formants)
        weight = np.where(freq < 0.45 * rate, weight, 0.0)
        x += weight * np.sin(h * phase) / np.sqrt(h)
    fricative = np.diff(rng.standard_normal(n + 1)) * 0.3
    fricative *= _gate(n, rate, rng, (0.05, 0.12), (0.4, 1.5))
    syllables = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 6))
    x = (x * syllables + fricative) * _gate(n, rate, rng, (0.4, 1.4), (0.1, 0.5))
    return np.asarray(x / np.sqrt(np.mean(x**2)), dtype=np.float64)


def echo_path(
    delay_ms: float,
    rate: int,
    seed: int,
    *,
    tail_s: float = 0.25,
    decay_s: float = 0.05,
    tail_db: float = -6.0,
) -> Signal:
    """Direct path after ``delay_ms`` plus a decaying random tail (reverberation).

    ``decay_s`` is the tail's 1/e amplitude time; ``tail_db`` its total energy relative
    to the direct path (a speaker close to the mics: the direct sound dominates).
    """
    rng = np.random.default_rng(seed)
    d = round(delay_ms * rate / 1000)
    h = np.zeros(d + int(tail_s * rate))
    h[d] = 1.0
    n = np.arange(len(h) - d - 1)
    tail = rng.standard_normal(len(n)) * np.exp(-n / (decay_s * rate))
    h[d + 1 :] = tail * np.sqrt(10 ** (tail_db / 10) / np.sum(tail**2))
    return h


def _mask(n: int, rate: int, spans: Sequence[Span]) -> npt.NDArray[np.bool_]:
    m = np.zeros(n, dtype=bool)
    for start, end in spans:
        m[int(start * rate) : int(end * rate)] = True
    return m


@dataclass
class EchoScene:
    """Signals of one synthetic scene, all of the same length, float64 at ``rate``."""

    rate: int
    far: Signal
    echo: Signal
    near: Signal
    noise: Signal

    @property
    def mic(self) -> Signal:
        """What the microphone captures."""
        return self.echo + self.near + self.noise

    @property
    def frame_samples(self) -> int:
        return self.rate // 100

    def frame_power(self, x: Signal) -> Signal:
        """Mean power of each 10 ms frame of ``x``."""
        f = self.frame_samples
        n = len(x) // f
        return np.asarray(np.mean(x[: n * f].reshape(n, f) ** 2, axis=1), dtype=np.float64)

    def far_only_frames(
        self, after_s: float = 0.0, echo_floor_db: float = -55.0
    ) -> npt.NDArray[np.bool_]:
        """Frames with audible echo, no near-end talker, from ``after_s`` on."""
        echo = self.frame_power(self.echo) > 10 ** (echo_floor_db / 10)
        quiet = self.frame_power(self.near) == 0.0
        t = np.arange(len(echo)) * 0.01
        return np.asarray(echo & quiet & (t >= after_s), dtype=bool)

    def near_frames(self) -> npt.NDArray[np.bool_]:
        """Frames where the near-end talker is active."""
        return np.asarray(self.frame_power(self.near) > 1e-8, dtype=bool)


def echo_scene(
    *,
    seconds: float = 10.0,
    rate: int = 16_000,
    delay_ms: float = 20.0,
    far_db: float = -20.0,
    echo_gain_db: float = -3.0,
    near_db: float = -28.0,
    noise_db: float = -65.0,
    far_spans: Sequence[Span] | None = None,
    near_spans: Sequence[Span] = (),
    seed: int = 1,
) -> EchoScene:
    """Build a scene. Spans are ``(start_s, end_s)``; the far end plays everywhere by default.

    Levels are dBFS of the unit-RMS voices; ``echo_gain_db`` is the speaker-to-mic gain
    (a loud echo, as on a robot whose speaker sits near its mics).
    """
    n = int(seconds * rate)
    far = voice(n, rate, 120.0, seed) * 10 ** (far_db / 20)
    if far_spans is not None:
        far = far * _mask(n, rate, far_spans)
    near = voice(n, rate, 210.0, seed + 1) * 10 ** (near_db / 20) * _mask(n, rate, near_spans)
    echo = fft_convolve(far, echo_path(delay_ms, rate, seed + 2)) * 10 ** (echo_gain_db / 20)
    noise = np.random.default_rng(seed + 3).standard_normal(n) * 10 ** (noise_db / 20)
    return EchoScene(rate, far, echo, near, noise)
