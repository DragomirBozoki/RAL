"""Online echo-delay tracking with GCC-PHAT, and the shared reference lead (DESIGN 5).

``software_ref`` feeds echo control every reference frame whose estimated DAC time is
earlier than ``t_mic + lead``. That only works if the reference reaches the canceller
before its echo; if the DAC-time estimate is off by more than ``lead`` (an output
latency the backend misreports, a ROS 2 sink with a wrong ``latency_ms``), echo control
breaks down (ERLE about 10 dB, ``spikes/apm_feasibility.py``).

:class:`DelayTracker` measures the delay between the reference, placed at its estimated
DAC time, and its echo in the microphone, placed at its ADC time, once a second over a
one-second window. Generalized cross-correlation with the phase transform whitens both
spectra so the correlation peak stays sharp in a reverberant room (C. Knapp and G.
Carter, "The generalized correlation method for estimation of time delay", IEEE Trans.
ASSP, 1976). A peak is accepted only if it stands well above the rest of the correlation,
and the reported delay is the median of the last accepted estimates.

If the measured delay ``d`` gets close to ``-lead`` (the reference would arrive late), the
tracker raises the shared :class:`LeadControl` to ``-d + margin``, up to ``max_lead_ms``;
it never lowers the lead below the profile's value.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ral.core.processing import FrameContext
from ral.core.types import AudioFrame
from ral.dsp.echo import MicChoice, select_mic
from ral.dsp.lead import LeadControl

__all__ = ["DelayTracker", "LeadControl", "gcc_phat"]

Signal = npt.NDArray[np.float64]


def gcc_phat(x: Signal, y: Signal, max_lag: int) -> tuple[int, float]:
    """Lag of ``x`` behind ``y`` (positive: ``x`` is later) and the peak's prominence.

    Prominence is the peak over the mean absolute correlation within the lag range.
    """
    n = len(x) + len(y)
    size = 1 << (n - 1).bit_length()
    cross = np.fft.rfft(x, size) * np.conj(np.fft.rfft(y, size))
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.irfft(cross, size)
    lags = np.concatenate([corr[-max_lag:], corr[: max_lag + 1]])  # lag -max..+max
    i = int(np.argmax(np.abs(lags)))
    mean = float(np.mean(np.abs(lags))) + 1e-12
    return i - max_lag, float(abs(lags[i]) / mean)


class DelayTracker:
    """Measures the reference-to-echo delay and keeps the lead safe.

    Put it first in the capture stages: it reads the selected mic and ``ctx.reference``
    and passes the frame through unchanged. It writes ``info["echo_delay_ms"]`` once it
    has an estimate.

    Args:
        lead: The shared lead to raise when the reference is at risk of arriving late.
        mic: Mic channel to correlate.
        window_s: Correlation window.
        interval_s: Time between estimates.
        max_delay_ms: Largest delay searched in either direction.
        margin_ms: How far ahead of its echo the reference must stay.
        max_lead_ms: Upper bound for the lead (AEC3's delay range).
        min_prominence: How far the correlation peak must stand out to count.
        min_level_db: Reference and mic must be at least this loud (dBFS).
    """

    def __init__(
        self,
        lead: LeadControl | None = None,
        *,
        mic: MicChoice = 0,
        window_s: float = 1.0,
        interval_s: float = 1.0,
        max_delay_ms: int = 300,
        margin_ms: int = 10,
        max_lead_ms: int = 200,
        min_prominence: float = 10.0,
        min_level_db: float = -50.0,
        history: int = 5,
    ) -> None:
        self.lead = lead or LeadControl()
        self.mic = mic
        self.window_s = window_s
        self.interval_ns = int(interval_s * 1e9)
        self.max_delay_ms = max_delay_ms
        self.margin_ms = margin_ms
        self.max_lead_ms = max_lead_ms
        self.min_prominence = min_prominence
        self.min_level = 10 ** (min_level_db / 10)
        self._history = history
        self.reset()

    def reset(self) -> None:
        self._mic: deque[tuple[int, npt.NDArray[np.float32]]] = deque()
        self._ref: deque[tuple[int, npt.NDArray[np.float32]]] = deque()
        self._next_ns: int | None = None
        self._estimates: deque[float] = deque(maxlen=self._history)
        self.delay_ms: float | None = None
        """Median of the recent accepted estimates (echo time minus reference DAC time)."""
        self.estimates = 0
        self.rejected = 0

    def process(self, frame: AudioFrame, ctx: FrameContext) -> AudioFrame:
        rate = frame.fmt.sample_rate
        keep_ns = int((self.window_s + 2 * self.max_delay_ms / 1000 + 0.5) * 1e9)
        self._mic.append((frame.t_ns, select_mic(frame, self.mic)[:, 0].copy()))
        for ref in ctx.reference:
            self._ref.append((ref.t_ns, ref.data[:, 0].copy()))
        while self._mic and frame.t_ns - self._mic[0][0] > keep_ns:
            self._mic.popleft()
        while self._ref and frame.t_ns - self._ref[0][0] > keep_ns:
            self._ref.popleft()
        if self._next_ns is None:
            self._next_ns = frame.t_ns + self.interval_ns
        elif frame.t_ns >= self._next_ns:
            self._next_ns = frame.t_ns + self.interval_ns
            self._estimate(frame.t_ns + frame.fmt.frame_ns, rate)
        if self.delay_ms is not None:
            ctx.info["echo_delay_ms"] = self.delay_ms
        return frame

    def _place(
        self, blocks: Sequence[tuple[int, npt.NDArray[np.float32]]], t0: int, n: int, rate: int
    ) -> Signal:
        """Blocks laid out on a common time axis starting at ``t0`` (gaps stay silent)."""
        out = np.zeros(n)
        for t, data in blocks:
            i = round((t - t0) * rate / 1e9)
            j0, j1 = max(i, 0), min(i + len(data), n)
            if j1 > j0:
                out[j0:j1] = data[j0 - i : j1 - i]
        return out

    def _estimate(self, t_end: int, rate: int) -> None:
        window = int(self.window_s * rate)
        max_lag = self.max_delay_ms * rate // 1000
        t_mic0 = t_end - int(self.window_s * 1e9)
        t_ref0 = t_mic0 - self.max_delay_ms * 1_000_000
        mic = self._place(self._mic, t_mic0, window, rate)
        ref = self._place(self._ref, t_ref0, window + 2 * max_lag, rate)
        if np.mean(ref**2) < self.min_level or np.mean(mic**2) < self.min_level:
            return  # the robot is silent: nothing to measure
        lag, prominence = gcc_phat(mic, ref, window + 2 * max_lag)
        # mic[m] matches ref[m - lag]; the ref segment starts max_delay before the mic's,
        # so the echo comes max_lag + lag samples after its reference's DAC time.
        delay_samples = lag + max_lag
        if prominence < self.min_prominence or abs(delay_samples) > max_lag:
            self.rejected += 1
            return
        self.estimates += 1
        self._estimates.append(1000 * delay_samples / rate)
        self.delay_ms = float(np.median(self._estimates))
        needed = int(np.ceil(-self.delay_ms + self.margin_ms))
        if needed > self.lead.ms:
            self.lead.ms = min(self.max_lead_ms, needed)
