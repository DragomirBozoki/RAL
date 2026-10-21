"""Run a synthetic scene through the capture graph, as the engine would, but offline.

The far end is "rendered" 100 ms ahead of its DAC time into a reference buffer (like a
real output buffer), the mic signal is cut into native frames captured at the same
clock, and :class:`~ral.engine.capture.CaptureGraph` processes them with the playback
intervals as the robot's speaking state. Everything is deterministic, so tests and
``ral bench`` can compare configurations on identical input.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ral.bench.synthetic import EchoScene
from ral.core.events import BargeIn, EchoLeak, Event, SpeechStarted
from ral.core.processing import Processor
from ral.core.types import AudioFormat, AudioFrame, ChannelMap
from ral.engine.capture import CaptureGraph
from ral.metrics.selftrigger import self_trigger_rate
from ral.playback.reference import ReferenceBuffer

__all__ = ["SceneRun", "run_scene"]

T0 = 1_000_000_000_000
FRAME_NS = 10_000_000


@dataclass
class SceneRun:
    """What came out of one offline run."""

    clean: npt.NDArray[np.float32]
    events: list[Event] = field(default_factory=list)
    playback: list[tuple[int, int]] = field(default_factory=list)
    user_speech: list[tuple[int, int]] = field(default_factory=list)
    t0: int = T0

    def of_type(self, kind: type[Event]) -> list[Event]:
        return [e for e in self.events if isinstance(e, kind)]

    def seconds(self, kind: type[Event]) -> list[float]:
        """Times of events of one type, in seconds from the start of the scene."""
        return [(e.t_ns - self.t0) / 1e9 for e in self.of_type(kind)]

    @property
    def self_trigger_rate(self) -> float:
        starts = [e.t_ns for e in self.of_type(SpeechStarted)]
        return self_trigger_rate(starts, self.playback, self.user_speech)

    @property
    def barge_ins(self) -> list[float]:
        return self.seconds(BargeIn)

    @property
    def echo_leaks(self) -> int:
        return len(self.of_type(EchoLeak))


def run_scene(
    scene: EchoScene,
    stages: Sequence[Processor],
    *,
    software_ref: bool,
    playback_s: Sequence[tuple[float, float]] | None = None,
    user_speech_s: Sequence[tuple[float, float]] = (),
    lead_ms: int = 20,
    render_ahead_ms: int = 100,
    on_frame: Callable[[AudioFrame], None] | None = None,
    loopback_ms: float | None = None,
    capture: tuple[npt.NDArray[np.float64], Sequence[str]] | None = None,
) -> SceneRun:
    """Process ``scene`` and collect the clean stream and the events.

    Args:
        scene: The synthetic scene.
        stages: Capture stages (echo control first).
        software_ref: Feed the reference buffer to the stages.
        playback_s: When the robot was speaking, in seconds; default the whole scene.
        user_speech_s: When the user was speaking (ground truth for the self-trigger rate).
        lead_ms: Reference lead.
        render_ahead_ms: How far ahead of its DAC time the far end is rendered.
        on_frame: Called with every clean frame.
        loopback_ms: Add a ``reference`` channel to the capture: the far end as a loopback
            on the mics' clock, ``loopback_ms`` late (for the ``loopback`` mode).
        capture: Use these capture channels ``(samples, channels)`` and roles instead of
            the scene's single mic (array devices, hardware-processed channels).
    """
    rate = scene.rate
    if capture is not None:
        signals, roles = np.asarray(capture[0], np.float64), list(capture[1])
    elif loopback_ms is not None:
        d = round(loopback_ms * rate / 1000)
        loop = np.concatenate([np.zeros(d), scene.far])[: len(scene.mic)]
        signals, roles = np.stack([scene.mic, loop], axis=1), ["mic", "reference"]
    else:
        signals, roles = scene.mic[:, None], ["mic"]
    cmap = ChannelMap.from_names(roles)
    cap_fmt = AudioFormat(rate, len(roles), "float32")
    fmt = AudioFormat(rate, 1, "float32")
    n = fmt.frame_samples
    frames = len(scene.mic) // n
    to_ns = [
        (T0 + int(a * 1e9), T0 + int(b * 1e9)) for a, b in (playback_s or [(0.0, frames / 100)])
    ]
    user_ns = [(T0 + int(a * 1e9), T0 + int(b * 1e9)) for a, b in user_speech_s]

    def speaking(t: int) -> bool:
        return any(a <= t < b for a, b in to_ns)

    ref = ReferenceBuffer(fmt, rate) if software_ref else None
    graph = CaptureGraph(
        cap_fmt,
        cmap,
        stages,
        reference=ref,
        lead_ms=lead_ms,
        speaking=speaking,
        session_id="bench",
        engine_rate=rate,
    )
    far = scene.far.astype(np.float32)
    mic = signals.astype(np.float32)
    ahead = render_ahead_ms // 10
    rendered = 0
    out: list[npt.NDArray[np.float32]] = []
    events: list[Event] = []
    for k in range(frames):
        while ref is not None and rendered < min(k + ahead, frames):
            ref.push(far[rendered * n : (rendered + 1) * n, None], T0 + rendered * FRAME_NS)
            rendered += 1
        frame = AudioFrame(
            mic[k * n : (k + 1) * n].copy(), cap_fmt, cmap, T0 + k * FRAME_NS, k, "mic"
        )
        for result in graph.process(frame):
            out.append(result.clean.data[:, 0])
            events.extend(result.ctx.events)
            if on_frame is not None:
                on_frame(result.clean)
    clean = np.concatenate(out) if out else np.zeros(0, np.float32)
    return SceneRun(clean, events, to_ns, user_ns)
