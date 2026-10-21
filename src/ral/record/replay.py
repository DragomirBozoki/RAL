"""Offline replay of a recorded session through another configuration (DESIGN section 8).

A recording made with ``--record`` holds everything the capture path saw: the normalized
mic frames (``/ral/mic``), the reference frames echo control was fed with their DAC times
(``/ral/reference``) and the playback events. Replay rebuilds those inputs and runs them
through :class:`~ral.engine.capture.CaptureGraph` with any echo, pipeline, VAD and barge-in
settings, so two configurations can be compared on exactly the same audio.

What cannot be replayed: the agent. Playback happened when it happened, so turn latency
is reported from the recording, not recomputed.
"""

from __future__ import annotations

import time
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ral.core.convert import int16_to_float32
from ral.core.errors import FormatError
from ral.core.events import Event, PlaybackEnded, PlaybackStarted
from ral.core.processing import Processor
from ral.core.types import AudioFormat, AudioFrame, ChannelMap, ChannelRole
from ral.engine.capture import CaptureGraph
from ral.record.mcap import Recording, read_recording

__all__ = ["RecordedReference", "ReplayResult", "SessionInputs", "load_session", "replay"]


class RecordedReference:
    """A :class:`~ral.core.ReferenceProvider` that hands out recorded reference frames."""

    def __init__(self, frames: Sequence[AudioFrame]) -> None:
        self._frames = sorted(frames, key=lambda f: f.t_ns)
        self._times = [f.t_ns for f in self._frames]
        self._next = 0

    def push(self, samples: npt.NDArray[np.float32], t_dac_ns: int) -> None:
        """Replay never renders anything."""

    def frames_until(self, t_ns: int) -> list[AudioFrame]:
        end = bisect_left(self._times, t_ns, lo=self._next)
        out = self._frames[self._next : end]
        self._next = end
        return out


@dataclass
class SessionInputs:
    """What the capture path saw during a recorded session."""

    mic: list[AudioFrame]
    reference: list[AudioFrame]
    playback: list[tuple[int, int]]
    events: list[Event]
    metadata: dict[str, dict[str, str]]

    @property
    def fmt(self) -> AudioFormat:
        return self.mic[0].fmt

    @property
    def seconds(self) -> float:
        return len(self.mic) * self.fmt.frame_ms / 1000


def _frames(
    rec: Recording, topic: str, roles: Sequence[ChannelRole] | None, rate: int
) -> list[AudioFrame]:
    blocks = rec.audio.get(topic, [])
    out = []
    for seq, (t_ns, pcm) in enumerate(blocks):
        data = int16_to_float32(pcm)
        channels = data.shape[1]
        fmt = AudioFormat(rate, channels, "float32")
        cmap = ChannelMap(tuple(roles)) if roles else ChannelMap.uniform(channels)
        if data.shape[0] != fmt.frame_samples:
            raise FormatError(f"{topic} has a {data.shape[0]}-sample block; expected 10 ms frames.")
        out.append(AudioFrame(data, fmt, cmap, t_ns, seq, topic))
    return out


def _intervals(events: Sequence[Event]) -> list[tuple[int, int]]:
    starts: dict[int, int] = {}
    spans = []
    for e in sorted(events, key=lambda e: e.t_ns):
        if isinstance(e, PlaybackStarted):
            starts[e.handle_id] = e.t_ns
        elif isinstance(e, PlaybackEnded) and e.handle_id in starts:
            spans.append((starts.pop(e.handle_id), e.t_ns))
    return sorted(spans)


def load_session(path: str | Path) -> SessionInputs:
    """Read the inputs of a recorded session.

    Raises:
        FormatError: If the recording has no raw mic audio (recorded with ``raw: false``).
    """
    rec = read_recording(path)
    if not rec.audio.get("/ral/mic"):
        raise FormatError(
            f"{path} has no /ral/mic audio; replay needs a recording made with recording.raw "
            f"enabled (the default)."
        )
    roles_text = rec.roles.get("/ral/mic", "")
    roles = [ChannelRole(r) for r in roles_text.split(",")] if roles_text else None
    mic = _frames(rec, "/ral/mic", roles, rec.rates["/ral/mic"])
    reference = _frames(
        rec, "/ral/reference", [ChannelRole.REFERENCE], rec.rates.get("/ral/reference", 16000)
    )
    return SessionInputs(mic, reference, _intervals(rec.events), rec.events, rec.metadata)


@dataclass
class ReplayResult:
    """The clean stream and events one configuration produced from a session."""

    name: str
    clean: npt.NDArray[np.float32]
    clean_t: list[int]
    events: list[Event] = field(default_factory=list)
    processing_us_per_frame: float = 0.0


def replay(
    inputs: SessionInputs, stages: Sequence[Processor], *, name: str = "", lead_ms: int = 20
) -> ReplayResult:
    """Run recorded inputs through ``stages``."""
    playback = inputs.playback

    def speaking(t: int) -> bool:
        return any(a <= t < b for a, b in playback)

    first = inputs.mic[0]
    graph = CaptureGraph(
        first.fmt,
        first.channel_map,
        stages,
        reference=RecordedReference(inputs.reference),
        lead_ms=lead_ms,
        speaking=speaking,
        session_id="replay",
        engine_rate=first.fmt.sample_rate,
    )
    out: list[npt.NDArray[np.float32]] = []
    times: list[int] = []
    events: list[Event] = []
    start = time.perf_counter()
    for frame in [*inputs.mic]:
        for result in graph.process(frame):
            out.append(result.clean.data[:, 0])
            times.append(result.raw.t_ns)
            events.extend(result.ctx.events)
    for result in graph.flush():
        out.append(result.clean.data[:, 0])
        times.append(result.raw.t_ns)
        events.extend(result.ctx.events)
    spent = time.perf_counter() - start
    clean = np.concatenate(out) if out else np.zeros(0, np.float32)
    per_frame = spent / max(1, len(inputs.mic)) * 1e6
    return ReplayResult(name, clean, times, events, per_frame)
