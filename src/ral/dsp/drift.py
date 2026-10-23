"""Clock drift between playback and capture, and its compensation (DESIGN section 5).

When the speaker and the microphones run on different clocks (two USB devices, two
boards), their sample rates differ by some parts per million. The reference then gains or
loses samples against the echo: 100 ppm is 6 ms per minute, enough to push AEC3 out of
alignment within a session.

:class:`DriftEstimator` measures one stream's real sample rate on the RAL clock from the
timestamp observations its Framer sees (sample index against capture or DAC time). A
least-squares line over a long window averages out callback jitter of milliseconds;
the slope is the sample period. Drift is the ratio of the two streams' relative rates.

:class:`DriftCorrector` resamples the reference by a ratio close to 1 that may change
slowly, with windowed-sinc interpolation at fractional positions (band-limited
interpolation, J. O. Smith, "Digital Audio Resampling Home Page", CCRMA, 2002; Kaiser
window as in :mod:`ral.dsp.resample`). The fixed engine-rate conversion stays in the
polyphase resampler; this stage only removes the ppm-level mismatch.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
import numpy.typing as npt

from ral.core.errors import FormatError

__all__ = ["DriftCorrector", "DriftEstimator", "drift_ppm"]

Float32Array = npt.NDArray[np.float32]


class DriftEstimator:
    """Real sample rate of one stream, measured on the RAL clock.

    Args:
        nominal_rate: The rate the stream claims.
        window_s: How much history the fit uses.
        min_span_s: How much history it needs before reporting.
        every_s: Keep one observation per this interval (decimation).
        refit_s: Recompute the line at most this often (callers ask every frame).
    """

    def __init__(
        self,
        nominal_rate: int,
        *,
        window_s: float = 120.0,
        min_span_s: float = 10.0,
        every_s: float = 0.01,
        refit_s: float = 1.0,
    ) -> None:
        self.nominal_rate = nominal_rate
        self.window_ns = int(window_s * 1e9)
        self.min_span_ns = int(min_span_s * 1e9)
        self.every_ns = int(every_s * 1e9)
        self.refit_ns = int(refit_s * 1e9)
        self._cached: float | None = None
        self._cached_at: int | None = None
        self._points: deque[tuple[int, int]] = deque()
        # Observations arrive on one thread (the stream's consumer) and estimates are read
        # on another (the DSP worker).
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._points.clear()
            self._cached = self._cached_at = None

    def observe(self, index: int, t_ns: int) -> None:
        """Sample ``index`` of the stream was at the ADC or DAC at ``t_ns``."""
        with self._lock:
            if self._points and t_ns - self._points[-1][1] < self.every_ns:
                return
            if self._points and index <= self._points[-1][0]:
                self._points.clear()  # a restart: the line no longer applies
            self._points.append((index, t_ns))
            while self._points and t_ns - self._points[0][1] > self.window_ns:
                self._points.popleft()

    @property
    def span_s(self) -> float:
        with self._lock:
            if len(self._points) < 2:
                return 0.0
            return (self._points[-1][1] - self._points[0][1]) / 1e9

    def rate_hz(self) -> float | None:
        """Measured sample rate, or None until ``min_span_s`` of history exists."""
        with self._lock:
            if not self._points:
                return None
            newest = self._points[-1][1]
            if self._cached_at is not None and newest - self._cached_at < self.refit_ns:
                return self._cached
            points = list(self._points)
        if len(points) < 8 or points[-1][1] - points[0][1] < self.min_span_ns:
            return None
        idx = np.array([p[0] for p in points], dtype=np.float64)
        t = np.array([p[1] for p in points], dtype=np.float64)
        idx -= idx[0]
        t -= t[0]
        # t = a + b * index: b is the sample period in ns. Centre the data for precision.
        b = float(np.polyfit(idx, t, 1)[0])
        rate = 1e9 / b if b > 0 else None
        with self._lock:
            self._cached, self._cached_at = rate, newest
        return rate

    def relative(self) -> float | None:
        """Measured rate over nominal rate (1.0001 is 100 ppm fast)."""
        rate = self.rate_hz()
        return None if rate is None else rate / self.nominal_rate


def drift_ppm(reference: DriftEstimator, capture: DriftEstimator) -> float | None:
    """How much faster the reference clock runs than the capture clock, in ppm."""
    r, c = reference.relative(), capture.relative()
    if r is None or c is None:
        return None
    return (r / c - 1.0) * 1e6


class DriftCorrector:
    """Mono resampler with a ratio near 1 that may change between calls.

    ``ratio`` is output samples per input sample. A reference clock that runs ``e`` faster
    than the mics produces ``1 + e`` samples per mic sample, so it is corrected with
    ``ratio = 1 / (1 + e)``.

    Args:
        half_taps: Sinc taps on each side of the interpolation point.
        max_ppm: Largest correction accepted (a larger one means a broken estimate).
    """

    def __init__(self, half_taps: int = 16, max_ppm: float = 2000.0, phases: int = 1024) -> None:
        self.half_taps = half_taps
        self.max_dev = max_ppm * 1e-6
        self._ratio = 1.0
        self._offsets = np.arange(-half_taps + 1, half_taps + 1, dtype=np.int64)
        # Kernel table: the windowed sinc at `phases` + 1 fractional positions, so each
        # output needs a table lookup and a linear blend instead of sinc and Bessel calls.
        self._phases = phases
        frac = np.arange(phases + 1, dtype=np.float64) / phases
        x = self._offsets[None, :].astype(np.float64) - frac[:, None]
        beta = 8.0
        arg = np.clip(1.0 - (x / half_taps) ** 2, 0.0, None)
        window = np.i0(beta * np.sqrt(arg)) / np.i0(beta)
        self._table = np.sinc(x) * window
        self.reset()

    @property
    def ratio(self) -> float:
        return self._ratio

    @ratio.setter
    def ratio(self, value: float) -> None:
        if not (1 - self.max_dev) <= value <= (1 + self.max_dev):
            raise FormatError(
                f"Drift correction ratio {value} is beyond {self.max_dev * 1e6:.0f} ppm; "
                f"check the clocks (or the estimate)."
            )
        self._ratio = float(value)

    def reset(self) -> None:
        self._hist: npt.NDArray[np.float64] = np.zeros(self.half_taps, dtype=np.float64)
        self._hist_start = -self.half_taps  # absolute input index of _hist[0]
        self._pos = float(self.half_taps)  # next output position: the first real sample
        self.positions: npt.NDArray[np.float64] = np.zeros(0, np.float64)
        """Absolute input position of each sample the last :meth:`process` returned."""
        self.consumed = 0
        self.produced = 0

    def _kernel(self, frac: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        phase = frac * self._phases
        i = np.minimum(phase.astype(np.int64), self._phases - 1)
        a = (phase - i)[:, None]
        kernel: npt.NDArray[np.float64] = (1.0 - a) * self._table[i] + a * self._table[i + 1]
        return kernel

    def process(self, samples: Float32Array) -> Float32Array:
        """Feed mono samples; returns every output sample now computable."""
        x = np.asarray(samples, dtype=np.float64).reshape(-1)
        self._hist = np.concatenate([self._hist, x])
        self.consumed += x.shape[0]
        step = 1.0 / self._ratio
        # Outputs need half_taps samples after floor(position).
        last = self._hist.shape[0] - self.half_taps - 1
        if self._pos > last:
            self.positions = np.zeros(0, np.float64)
            return np.zeros(0, np.float32)
        count = int(np.floor((last - self._pos) / step)) + 1
        pos: npt.NDArray[np.float64] = self._pos + step * np.arange(count, dtype=np.float64)
        base = np.floor(pos).astype(np.int64)
        frac: npt.NDArray[np.float64] = pos - base
        idx = base[:, None] + self._offsets[None, :]
        out = np.sum(self._hist[idx] * self._kernel(frac), axis=1)
        self.positions = pos + self._hist_start
        self._pos = float(pos[-1] + step)
        # Drop history no longer needed, keeping positions relative to the new start.
        drop = max(0, int(np.floor(self._pos)) - self.half_taps)
        if drop:
            self._hist = self._hist[drop:]
            self._pos -= drop
            self._hist_start += drop
        self.produced += count
        result: Float32Array = out.astype(np.float32)
        return result
