"""What ``ral probe`` concludes from its recordings (DESIGN section 6, "What ral probe does").

Inputs are plain arrays, so the analysis runs the same on real devices and in tests:

1. **Noise floor** from a recording of silence: level per channel, dead channels (digital
   silence) and clipped ones.
2. **Impulse responses** from sweeps (:mod:`ral.probe.signals`), with index 0 at the DAC
   time of the sweep's first sample, so the peak position is the playback-to-capture delay.
3. **Channel roles.** A loopback of the speaker feed is a clean, near-instant copy: a
   strong peak with almost no energy after it (high direct-to-reverberant ratio). A
   microphone hears the room: a weaker peak followed by reverberation. A channel that is
   alive in silence but hears almost nothing of a sweep that the mics hear loudly is an
   echo-cancelled (``processed``) output. Anything else is ``unused``.
4. **Delay and echo gain** per microphone, and **drift** from the slope of the delay over
   repeated sweeps (parts per million).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ral.dsp.levels import SILENCE_DB
from ral.probe.signals import deconvolve

__all__ = ["ChannelReport", "ProbeResult", "analyze", "estimate_drift_ppm"]

Signal = npt.NDArray[np.float64]

DEAD_DB = -110.0
CLIP = 0.999
REF_DRR_DB = 20.0
SNR_DB = 15.0
PROCESSED_BELOW_DB = 20.0


@dataclass
class ChannelReport:
    """One input channel."""

    index: int
    role: str
    noise_db: float
    clipped_share: float
    gain_db: float | None = None
    delay_ms: float | None = None
    snr_db: float | None = None
    drr_db: float | None = None
    note: str = ""


@dataclass
class ProbeResult:
    """Everything the probe measured."""

    rate: int
    channels: list[ChannelReport]
    irs: Signal  # (samples, channels), first sweep, index 0 = DAC time of the sweep
    drift_ppm: float | None = None
    suggested_mode: str = "software_ref"
    reasons: list[str] = field(default_factory=list)

    @property
    def roles(self) -> list[str]:
        return [c.role for c in self.channels]

    @property
    def mic_delay_ms(self) -> float | None:
        """Shortest playback-to-capture delay over the microphones."""
        delays = [c.delay_ms for c in self.channels if c.role == "mic" and c.delay_ms is not None]
        return min(delays) if delays else None


def _db(x: float) -> float:
    return 10 * float(np.log10(max(x, 1e-30)))


def _peak(h: Signal) -> float:
    """Peak position with parabolic interpolation (sub-sample)."""
    i = int(np.argmax(np.abs(h)))
    if 0 < i < len(h) - 1:
        a, b, c = np.abs(h[i - 1 : i + 2])
        denom = a - 2 * b + c
        if denom != 0:
            return float(i + 0.5 * (a - c) / denom)
    return float(i)


def _ir_metrics(h: Signal, rate: int) -> tuple[float, float, float, float]:
    """(gain_db, delay_samples, snr_db, drr_db) of one impulse response."""
    p = _peak(h)
    i = round(p)
    gain = float(abs(h[i]))
    tail_start = i + int(0.5 * rate)
    tail = h[tail_start:] if tail_start < len(h) - rate // 10 else h[: max(1, i - rate // 100)]
    noise = float(np.sqrt(np.mean(tail**2))) if tail.size else 0.0
    direct = h[max(0, i - rate // 1000) : i + 2 * rate // 1000 + 1]
    reverb = h[i + 2 * rate // 1000 + 1 : i + int(0.15 * rate)]
    snr = 20 * np.log10(max(gain, 1e-30) / max(noise, 1e-30))
    drr = _db(float(np.sum(direct**2))) - _db(float(np.sum(reverb**2)))
    return 20 * float(np.log10(max(gain, 1e-30))), p, float(snr), drr


def estimate_drift_ppm(delays_s: Sequence[float], times_s: Sequence[float]) -> float | None:
    """Slope of delay over time, in parts per million (None with fewer than two points).

    A positive value means the delay grows: capture runs slower than playback.
    """
    if len(delays_s) < 2 or max(times_s) - min(times_s) <= 0:
        return None
    slope = float(np.polyfit(np.asarray(times_s), np.asarray(delays_s), 1)[0])
    return slope * 1e6


def analyze(
    silence: Signal,
    sweeps: Sequence[Signal],
    sweep_times_s: Sequence[float],
    inv: Signal,
    rate: int,
    *,
    hardware_aec_hint: bool = False,
    same_device: bool = True,
) -> ProbeResult:
    """Classify channels and measure delay, gain and drift.

    Args:
        silence: ``(samples, channels)`` recorded with nothing playing.
        sweeps: Recordings of each sweep, each starting at the sweep's DAC time.
        sweep_times_s: When each sweep was played, relative to the first.
        inv: The sweep's inverse filter.
        rate: Sample rate of all recordings.
        hardware_aec_hint: The device database says the device cancels echo on board.
        same_device: Playback and capture are the same device (one clock).
    """
    channels = silence.shape[1]
    reports: list[ChannelReport] = []
    for c in range(channels):
        x = silence[:, c]
        noise = max(SILENCE_DB, _db(float(np.mean(x**2))))
        clipped = float(np.mean(np.abs(x) >= CLIP))
        reports.append(ChannelReport(c, "unused", noise, clipped))

    irs = np.column_stack([deconvolve(sweeps[0][:, c], inv) for c in range(channels)])
    for r in reports:
        gain, p, snr, drr = _ir_metrics(irs[:, r.index], rate)
        r.gain_db, r.delay_ms, r.snr_db, r.drr_db = gain, 1000 * p / rate, snr, drr

    def is_reference(r: ChannelReport) -> bool:
        assert r.snr_db is not None
        assert r.drr_db is not None
        assert r.delay_ms is not None
        return r.snr_db > SNR_DB and r.drr_db > REF_DRR_DB and r.delay_ms < 5

    best_mic_gain = max(
        (
            r.gain_db
            for r in reports
            if r.gain_db is not None
            and r.snr_db is not None
            and r.snr_db > SNR_DB
            and not is_reference(r)
        ),
        default=None,
    )
    for r in reports:
        assert r.snr_db is not None
        assert r.gain_db is not None
        if r.noise_db < DEAD_DB and r.snr_db < SNR_DB:
            r.role, r.note = "unused", "dead (digital silence)"
        elif is_reference(r):
            r.role, r.note = "reference", "clean copy of the speaker feed"
        elif best_mic_gain is not None and r.gain_db < best_mic_gain - PROCESSED_BELOW_DB:
            # The long sweep's processing gain lifts even a weak residual above the noise,
            # so judge by level against the loudest mic, not by SNR alone.
            r.role, r.note = "processed", "alive, but the sweep is cancelled: on-board AEC"
        elif r.snr_db > SNR_DB:
            r.role = "mic"
        else:
            r.role, r.note = "unused", "no response to the sweep"
        if r.clipped_share > 0.001:
            r.note = (r.note + "; " if r.note else "") + f"clipped {100 * r.clipped_share:.1f} %"

    delays = []
    for rec in sweeps:
        mics = [r.index for r in reports if r.role == "mic"]
        if not mics:
            break
        h = deconvolve(rec[:, mics[0]], inv)
        delays.append(_peak(h) / rate)
    drift = estimate_drift_ppm(delays, sweep_times_s[: len(delays)]) if len(delays) > 1 else None

    result = ProbeResult(rate, reports, irs, drift)
    _suggest(result, hardware_aec_hint, same_device)
    return result


def _suggest(result: ProbeResult, hardware_aec_hint: bool, same_device: bool) -> None:
    roles = result.roles
    if "processed" in roles or (hardware_aec_hint and same_device):
        result.suggested_mode = "hardware"
        result.reasons.append("the device cancels echo itself; play through it")
    elif "reference" in roles:
        result.suggested_mode = "loopback"
        result.reasons.append("a reference channel arrives with the mics on one clock")
    elif "mic" in roles:
        result.suggested_mode = "software_ref"
        delay = result.mic_delay_ms
        result.reasons.append(
            f"no loopback; RAL's own reference leads its echo (measured delay {delay:.1f} ms)"
            if delay is not None
            else "no loopback; RAL's own reference"
        )
        if not same_device:
            result.reasons.append("capture and playback are different devices: watch drift")
    else:
        result.suggested_mode = "gate"
        result.reasons.append("no channel hears the speaker; only gating is possible")
