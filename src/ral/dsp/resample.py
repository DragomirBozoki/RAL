"""Streaming polyphase resampler for fixed rational rate ratios.

Converts ``in_rate`` to ``out_rate`` with ``L / M = out_rate / in_rate`` reduced by their
greatest common divisor (48 kHz to 16 kHz is 1/3, 44.1 kHz to 16 kHz is 160/441).
Conceptually the input is upsampled by ``L`` (zeros between samples), low-pass filtered,
and downsampled by ``M``; the polyphase form computes only the outputs that are kept, each
as a dot product of ``K`` input samples with one of ``L`` sub-filters.

Why not a library: ``samplerate`` (libsamplerate) has no wheel for CPython 3.10 on Linux
aarch64, which is exactly the Jetson and ROS 2 Humble target, and engine ratios are fixed.
Drift compensation (a slowly varying ratio near 1) is a separate stage,
:class:`ral.dsp.drift.DriftCorrector`, after this one.

Alignment. The prototype filter is linear-phase with an odd length ``N``, so it delays the
upsampled signal by ``D = (N - 1) / 2`` samples. The resampler compensates by reading ``D``
samples ahead: output sample ``k`` estimates the signal at input time ``k * M / L``
exactly. The price is lookahead (``D / L`` input samples) before an output is ready; that
is latency, not a timestamp error, so frame timestamps can be carried over unchanged.

Filter: windowed sinc with a Kaiser window, ``beta = 0.1102 * (A - 8.7)`` for a stopband
attenuation of ``A`` dB, cutoff below the lower Nyquist frequency so the transition band
ends before aliases fold back. References: J. F. Kaiser, "Nonrecursive digital filter
design using the I0-sinh window function", Proc. IEEE ISCAS, 1974; R. E. Crochiere and
L. R. Rabiner, "Multirate Digital Signal Processing", Prentice-Hall, 1983, ch. 4
(polyphase structures); A. V. Oppenheim and R. W. Schafer, "Discrete-Time Signal
Processing", section 7.6 (Kaiser window design).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from ral.core.errors import FormatError

__all__ = ["Resampler"]

Float32Array = npt.NDArray[np.float32]


class Resampler:
    """Streams float32 audio from ``in_rate`` to ``out_rate``.

    Args:
        in_rate: Input sample rate in Hz.
        out_rate: Output sample rate in Hz.
        channels: Channels per sample; all channels share one filter.
        zero_crossings: Sinc zero crossings on each side of the centre, counted at the
            lower of the two rates. More gives a sharper transition and more latency.
        rolloff: Cutoff as a fraction of the lower Nyquist frequency.
        attenuation_db: Stopband attenuation the Kaiser window is designed for.

    Raises:
        FormatError: If an argument is out of range.
    """

    def __init__(
        self,
        in_rate: int,
        out_rate: int,
        channels: int,
        *,
        zero_crossings: int = 24,
        rolloff: float = 0.88,
        attenuation_db: float = 80.0,
    ) -> None:
        if in_rate <= 0 or out_rate <= 0:
            raise FormatError(f"Sample rates must be positive, got {in_rate} and {out_rate}.")
        if channels <= 0:
            raise FormatError(f"channels must be at least 1, got {channels}.")
        if zero_crossings < 1:
            raise FormatError(f"zero_crossings must be at least 1, got {zero_crossings}.")
        if not 0.0 < rolloff <= 1.0:
            raise FormatError(f"rolloff must be in (0, 1], got {rolloff}.")
        g = math.gcd(in_rate, out_rate)
        self.in_rate = in_rate
        self.out_rate = out_rate
        self.channels = channels
        self.up = out_rate // g  # L
        self.down = in_rate // g  # M
        self.passthrough = self.up == self.down
        if self.passthrough:
            self._taps = 1
            self._delay = 0
            self._bank: Float32Array = np.ones((1, 1), dtype=np.float32)
        else:
            self._bank, self._taps, self._delay = self._design(
                zero_crossings, rolloff, attenuation_db
            )
        self.reset()

    def _design(
        self, zero_crossings: int, rolloff: float, attenuation_db: float
    ) -> tuple[Float32Array, int, int]:
        L, M = self.up, self.down
        r = max(L, M)
        half = zero_crossings * r  # D, in samples at the upsampled rate
        n = np.arange(-half, half + 1, dtype=np.float64)
        cutoff = 0.5 * rolloff / r  # cycles per upsampled sample
        beta = 0.1102 * (attenuation_db - 8.7) if attenuation_db > 50 else 5.0
        h = 2.0 * cutoff * np.sinc(2.0 * cutoff * n) * np.kaiser(n.size, beta)
        h *= L / h.sum()  # unity DC gain after zero-stuffing by L
        taps = -(-h.size // L)
        padded = np.zeros(taps * L)
        padded[: h.size] = h
        # bank[p, j] = h[p + j * L]: sub-filter p multiplies x[n - j].
        bank = padded.reshape(taps, L).T.astype(np.float32)
        return np.ascontiguousarray(bank), taps, half

    @property
    def lookahead(self) -> int:
        """Input samples an output sample needs beyond its own time."""
        return -(-self._delay // self.up)

    def reset(self) -> None:
        """Forget all input; the next sample is treated as the start of a new stream."""
        # Input history starts with `taps` virtual zeros before sample 0.
        self._hist: Float32Array = np.zeros((self._taps, self.channels), dtype=np.float32)
        self._hist_start = -self._taps  # absolute input index of _hist[0]
        self._consumed = 0  # input samples received
        self._produced = 0  # output samples emitted

    def process(self, samples: Float32Array) -> Float32Array:
        """Feed ``(n, channels)`` float32 samples; returns every output now computable."""
        if samples.ndim != 2 or samples.shape[1] != self.channels:
            raise FormatError(f"Block has shape {samples.shape}, expected (n, {self.channels}).")
        if self.passthrough:
            self._consumed += samples.shape[0]
            self._produced += samples.shape[0]
            return samples.astype(np.float32, copy=True)
        self._hist = np.concatenate([self._hist, samples.astype(np.float32, copy=False)])
        self._consumed += samples.shape[0]
        out = self._compute(self._consumed - 1)
        self._trim()
        return out

    def flush(self) -> Float32Array:
        """Pad with silence so every output up to the last input sample is emitted."""
        if self.passthrough or self._consumed == 0:
            return np.zeros((0, self.channels), dtype=np.float32)
        pad = self.lookahead + 1
        self._hist = np.concatenate([self._hist, np.zeros((pad, self.channels), np.float32)])
        # Outputs up to the time of the last real input sample.
        last_time_out = (self._consumed - 1) * self.up // self.down
        out = self._compute(self._consumed - 1 + pad, max_k=last_time_out)
        self.reset()
        return out

    def _compute(self, last_input: int, max_k: int | None = None) -> Float32Array:
        L, M, D = self.up, self.down, self._delay
        # Output k reads input index n_k = floor((k * M + D) / L) and needs n_k <= last_input.
        k_end = (last_input * L + L - 1 - D) // M  # largest k with n_k <= last_input
        if max_k is not None:
            k_end = min(k_end, max_k)
        k = np.arange(self._produced, k_end + 1, dtype=np.int64)
        if k.size == 0:
            return np.zeros((0, self.channels), dtype=np.float32)
        pos = k * M + D
        n_k = pos // L
        phase = pos % L
        # Gather windows x[n_k - j] for j = 0..taps-1 from the history.
        idx = (n_k - self._hist_start)[:, None] - np.arange(self._taps)[None, :]
        windows = self._hist[idx]  # (len(k), taps, channels)
        coeffs = self._bank[phase]  # (len(k), taps)
        out: Float32Array = np.einsum("kj,kjc->kc", coeffs, windows).astype(np.float32)
        self._produced = int(k_end) + 1
        return out

    def _trim(self) -> None:
        # Keep only the history the next output can still reach.
        next_n = (self._produced * self.down + self._delay) // self.up
        keep_from = next_n - (self._taps - 1)
        drop = keep_from - self._hist_start
        if drop > 0:
            self._hist = self._hist[drop:]
            self._hist_start += drop
