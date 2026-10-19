"""Per-turn latency (DESIGN section 8, "Turn latency").

A turn starts when the user stops speaking and ends when the robot's reply is first
heard. :class:`LatencyTracker` listens to the event bus:

1. ``SpeechEnded`` gives the end of the user's speech (``t_ns``, before the hangover)
   and, from the moment it is published, when RAL decided the user had finished.
2. The next ``PlaybackStarted`` whose audio arrived after that decision is the reply. Its
   ``t_request_ns`` is when the reply's first audio reached RAL and its ``t_ns`` when that
   audio was estimated at the DAC.

The breakdown (``hangover``, ``agent``, ``playback``) adds up to the total, so the slow
stage is visible. RAL cannot see inside the agent; bridges that report STT, model and
TTS timings will split ``agent`` further. TODO(M2): take those timings from bridges.

Replies started by the robot itself (earcons, ``--play``) have no preceding user speech
and are not turns. A barge-in cancels a pending turn: the robot was speaking, and the
user's next speech end starts a new one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ral.core.clock import now_ns
from ral.core.events import BargeIn, Event, PlaybackStarted, SpeechEnded, TurnLatency

__all__ = ["LatencyTracker", "StageStats"]


@dataclass
class StageStats:
    """Median and 90th percentile of one stage, in milliseconds."""

    name: str
    median_ms: float
    p90_ms: float


@dataclass
class _Pending:
    speech_end_ns: int
    decided_ns: int
    session_id: str


@dataclass
class LatencyTracker:
    """Builds :class:`TurnLatency` events from speech and playback events.

    Args:
        publish: Called with every completed turn (the engine publishes it on the bus).
        clock: Time source for "when was this decided"; the RAL clock by default.
        tag_prefixes: Only playback whose tag starts with one of these counts as a reply
            (default: anything except RAL's own ``play`` items).
    """

    publish: Callable[[Event], None] | None = None
    clock: Callable[[], int] = now_ns
    tag_prefixes: tuple[str, ...] | None = None
    turns: list[TurnLatency] = field(default_factory=list)
    _pending: _Pending | None = None

    def on_event(self, event: Event) -> None:
        """Feed every bus event; call on the loop as events are published."""
        if isinstance(event, SpeechEnded):
            self._pending = _Pending(event.t_ns, self.clock(), event.session_id)
        elif isinstance(event, BargeIn):
            self._pending = None
        elif isinstance(event, PlaybackStarted) and self._pending is not None:
            if not self._is_reply(event.tag) or event.t_request_ns < self._pending.decided_ns:
                return
            p = self._pending
            self._pending = None
            turn = TurnLatency(
                p.speech_end_ns,
                p.session_id,
                total_ms=(event.t_ns - p.speech_end_ns) / 1e6,
                hangover_ms=(p.decided_ns - p.speech_end_ns) / 1e6,
                agent_ms=(event.t_request_ns - p.decided_ns) / 1e6,
                playback_ms=(event.t_ns - event.t_request_ns) / 1e6,
                tag=event.tag,
            )
            self.turns.append(turn)
            if self.publish is not None:
                self.publish(turn)

    def _is_reply(self, tag: str) -> bool:
        if self.tag_prefixes is None:
            return tag != "play"
        return tag.startswith(self.tag_prefixes)

    def summary(self) -> list[StageStats]:
        """Median and p90 per stage over all turns so far (empty if there were none)."""
        if not self.turns:
            return []
        out = []
        for name in ("total_ms", "hangover_ms", "agent_ms", "playback_ms"):
            values = np.array([getattr(t, name) for t in self.turns], dtype=np.float64)
            out.append(
                StageStats(
                    name.removesuffix("_ms"),
                    float(np.median(values)),
                    float(np.percentile(values, 90)),
                )
            )
        return out
