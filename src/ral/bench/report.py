"""``ral bench``: one recorded session through several configurations, side by side.

Metrics per configuration (DESIGN section 8, "Metrics"):

- ``erle_db``: echo return loss enhancement over frames where the robot was speaking and
  its reference was audible. A recording has no ground truth for who spoke when, so
  double talk counts too: this is a lower bound of the true ERLE.
- ``starts_per_min``: speech starts while the robot was speaking, per minute of robot
  speech. Without labels it bounds the self-trigger rate from above (it also counts real
  interruptions).
- ``barge_ins`` and ``echo_leaks``: counts.
- ``us_per_frame``: processing cost per 10 ms frame on this machine.

Turn latency depends on the agent, which replay cannot rerun; it is reported once, from the
recording.
"""

from __future__ import annotations

import json
import tempfile
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ral.bench.synthetic import EchoScene
from ral.core.events import (
    BargeIn,
    EchoLeak,
    PlaybackEnded,
    PlaybackStarted,
    SpeechStarted,
    TurnLatency,
)
from ral.core.types import AudioFormat, AudioFrame, ChannelMap, ChannelRole
from ral.record.replay import ReplayResult, SessionInputs

__all__ = ["BenchRow", "bench_rows", "recorded_latency_ms", "rows_as_json", "write_scene_recording"]

_REF_ACTIVE_DB = -50.0


@dataclass
class BenchRow:
    """One configuration's numbers on one session."""

    name: str
    erle_db: float | None
    starts_per_min: float
    speech_starts: int
    barge_ins: int
    echo_leaks: int
    us_per_frame: float

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def _speaking(t: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(a <= t < b for a, b in spans)


def _db(power: float) -> float:
    return 10 * float(np.log10(max(power, 1e-20)))


def bench_rows(inputs: SessionInputs, results: Sequence[ReplayResult]) -> list[BenchRow]:
    """Compute the comparison rows for each replay of the same session."""
    spans = inputs.playback
    minutes = sum(b - a for a, b in spans) / 60e9
    mics = inputs.mic[0].channel_map.indices(ChannelRole.MIC) or (0,)
    mic_index = mics[0]
    ref_power = {
        f.t_ns: float(np.mean(f.data[:, 0].astype(np.float64) ** 2)) for f in inputs.reference
    }
    ref_times = sorted(ref_power)
    rows = []
    for res in results:
        n = inputs.fmt.frame_samples
        num = den = 0.0
        for k, frame in enumerate(inputs.mic[: len(res.clean_t)]):
            t = frame.t_ns
            if not _speaking(t, spans):
                continue
            # Reference audible around this frame (its DAC time within the last 100 ms).
            i = bisect_right(ref_times, t)
            window = ref_times[max(0, i - 10) : i + 1]
            if not window or _db(max(ref_power[w] for w in window)) < _REF_ACTIVE_DB:
                continue
            mic = frame.data[:, mic_index].astype(np.float64)
            out = res.clean[k * n : (k + 1) * n].astype(np.float64)
            num += float(np.sum(mic**2))
            den += float(np.sum(out**2))
        erle = None if num == 0 else _db(num) - _db(den)
        starts = [e for e in res.events if isinstance(e, SpeechStarted)]
        during = [e for e in starts if _speaking(e.t_ns, spans)]
        rows.append(
            BenchRow(
                name=res.name,
                erle_db=erle,
                starts_per_min=len(during) / minutes if minutes > 0 else 0.0,
                speech_starts=len(starts),
                barge_ins=sum(isinstance(e, BargeIn) for e in res.events),
                echo_leaks=sum(isinstance(e, EchoLeak) for e in res.events),
                us_per_frame=res.processing_us_per_frame,
            )
        )
    return rows


def recorded_latency_ms(inputs: SessionInputs) -> list[float]:
    """Turn latencies stored in the recording (the agent's, not replayable)."""
    return [e.total_ms for e in inputs.events if isinstance(e, TurnLatency)]


def write_scene_recording(
    scene: EchoScene,
    path: str | Path | None = None,
    *,
    playback_s: Sequence[tuple[float, float]] | None = None,
    capture: tuple[npt.NDArray[np.float64], Sequence[str]] | None = None,
    t0: int = 1_000_000_000_000,
) -> Path:
    """Write a synthetic scene as if RAL had recorded it (for tests and ``ral bench --demo``).

    The far end is the reference (DAC time = its sample time); the mic channels are the
    capture; playback events mark ``playback_s`` (default: the whole scene).
    """
    from ral.record.mcap import McapRecorder

    out = Path(path) if path is not None else Path(tempfile.mkdtemp()) / "scene.mcap"
    rate = scene.rate
    signals, roles = capture if capture is not None else (scene.mic[:, None], ["mic"])
    signals = np.asarray(signals, np.float64)
    n = rate // 100
    fmt = AudioFormat(rate, len(roles), "float32")
    mono = AudioFormat(rate, 1, "float32")
    cmap = ChannelMap.from_names(list(roles))
    ref_map = ChannelMap((ChannelRole.REFERENCE,))
    rec = McapRecorder(out, session_id="synthetic")
    rec.start()
    frames = len(signals) // n
    spans = playback_s if playback_s is not None else [(0.0, frames / 100)]
    for h, (a, b) in enumerate(spans, start=1):
        rec.write_event(
            PlaybackStarted(
                t0 + int(a * 1e9),
                "synthetic",
                handle_id=h,
                tag="synthetic",
                t_request_ns=t0 + int(a * 1e9),
            )
        )
        rec.write_event(
            PlaybackEnded(
                t0 + int(b * 1e9), "synthetic", handle_id=h, tag="synthetic", interrupted=False
            )
        )
    far = scene.far.astype(np.float32)
    for k in range(frames):
        t = t0 + k * 10_000_000
        raw = AudioFrame(signals[k * n : (k + 1) * n].astype(np.float32), fmt, cmap, t, k, "mic")
        ref = AudioFrame(far[k * n : (k + 1) * n, None].copy(), mono, ref_map, t, k, "dac")
        clean = AudioFrame(raw.data[:, :1].copy(), mono, ChannelMap.uniform(1), t, k, "mic")
        rec.write_result(raw, clean, [ref])
    rec.stop()
    return out


def rows_as_json(rows: Sequence[BenchRow], latency_ms: Sequence[float]) -> str:
    return json.dumps(
        {"configs": [r.to_json() for r in rows], "recorded_turn_latency_ms": list(latency_ms)},
        indent=2,
    )
