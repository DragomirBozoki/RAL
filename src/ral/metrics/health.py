"""Health monitor: catch hardware and timing problems before users do (DESIGN section 8).

Two kinds of checks:

- **From the audio** (:meth:`HealthMonitor.observe`, every processed frame, timed by the
  frames themselves so offline runs behave like live ones): clipping per channel
  (share of samples at full scale over one second, ``Clipping``), dead channels (below
  ``dead_db`` for ``dead_s``, ``DeadChannel``), and each channel's noise floor (the lowest
  one-second level of the last 30 s).
- **From counters** (:meth:`HealthMonitor.poll`, once a second on the event loop): xruns
  of every stream (capture overruns and lost frames, PortAudio input overflows and output
  underflows, playback underruns, frames the engine dropped), reported as ``Xrun`` with
  the increase, and a capture that stopped delivering (``DeviceLost``).

:meth:`HealthMonitor.snapshot` collects everything with the engine's drift, delay, lead
and processing cost, for the metrics exporters and the Studio.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from ral.core.clock import now_ns
from ral.core.events import Clipping, DeviceLost, Event, Xrun

__all__ = ["DeadChannel", "HealthMonitor"]

_CLIP = 0.999


@dataclass(frozen=True)
class DeadChannel(Event):
    """A capture channel has carried (almost) nothing for ``seconds``."""

    channel: int
    seconds: float


@dataclass
class _Channel:
    clipped: int = 0
    samples: int = 0
    level_sum: float = 0.0
    frames: int = 0
    quiet_since: int | None = None
    dead_reported: bool = False
    second_levels: deque[float] = field(default_factory=lambda: deque(maxlen=30))


class HealthMonitor:
    """Watches one engine.

    Args:
        session_id: Stamped on the events.
        publish: Called with every health event.
        counters: Returns the current value of each xrun counter, by name.
        clip_share: Clipped share of a second that raises ``Clipping``.
        dead_db, dead_s: A channel below ``dead_db`` for ``dead_s`` is dead.
        stall_s: No capture frame for this long raises ``DeviceLost``.
    """

    def __init__(
        self,
        session_id: str = "",
        publish: Callable[[Event], None] | None = None,
        counters: Callable[[], dict[str, int]] | None = None,
        *,
        clip_share: float = 0.001,
        dead_db: float = -100.0,
        dead_s: float = 5.0,
        stall_s: float = 2.0,
        device_id: str = "capture",
        extra: Callable[[], dict[str, float | None]] | None = None,
    ) -> None:
        self.session_id = session_id
        self.publish = publish or (lambda _e: None)
        self.counters = counters or dict
        self.extra = extra or dict
        self.clip_share = clip_share
        self.dead_level = 10 ** (dead_db / 10)
        self.dead_ns = int(dead_s * 1e9)
        self.stall_ns = int(stall_s * 1e9)
        self.device_id = device_id
        self._channels: list[_Channel] = []
        self._second_start: int | None = None
        self._last_counts: dict[str, int] = {}
        self.xruns: dict[str, int] = {}
        self._last_frame_wall: int | None = None
        self._lost_reported = False
        self.frames = 0
        self.clipping_events = 0

    # Per frame (the loop thread; numpy on 160 samples)

    def observe(self, data: npt.NDArray[Any], t_ns: int, frame_ns: int = 10_000_000) -> None:
        """One raw engine frame ``(samples, channels)`` captured at ``t_ns``."""
        self.frames += 1
        self._last_frame_wall = now_ns()
        self._lost_reported = False
        channels = data.shape[1]
        while len(self._channels) < channels:
            self._channels.append(_Channel())
        x = np.abs(data)
        clipped = np.count_nonzero(x >= _CLIP, axis=0)
        power = np.mean(np.square(data, dtype=np.float64), axis=0)
        for c in range(channels):
            ch = self._channels[c]
            ch.clipped += int(clipped[c])
            ch.samples += data.shape[0]
            ch.level_sum += float(power[c])
            ch.frames += 1
            if power[c] < self.dead_level:
                ch.quiet_since = t_ns if ch.quiet_since is None else ch.quiet_since
                if not ch.dead_reported and t_ns + frame_ns - ch.quiet_since >= self.dead_ns:
                    ch.dead_reported = True
                    seconds = (t_ns + frame_ns - ch.quiet_since) / 1e9
                    self.publish(DeadChannel(t_ns, self.session_id, channel=c, seconds=seconds))
            else:
                ch.quiet_since, ch.dead_reported = None, False
        if self._second_start is None:
            self._second_start = t_ns
        if t_ns + frame_ns - self._second_start >= 1_000_000_000:
            self._close_second(t_ns)
            self._second_start = t_ns + frame_ns

    def _close_second(self, t_ns: int) -> None:
        for c, ch in enumerate(self._channels):
            if ch.samples:
                share = ch.clipped / ch.samples
                if share > self.clip_share:
                    self.clipping_events += 1
                    self.publish(Clipping(t_ns, self.session_id, channel=c, share=share))
            if ch.frames:
                ch.second_levels.append(10 * float(np.log10(max(ch.level_sum / ch.frames, 1e-12))))
            ch.clipped = ch.samples = ch.frames = 0
            ch.level_sum = 0.0

    # Once a second (the loop)

    def poll(self, wall_ns: int | None = None) -> None:
        """Compare counters with the last poll and check that capture is alive."""
        now = now_ns() if wall_ns is None else wall_ns
        for name, value in self.counters().items():
            last = self._last_counts.get(name, value)
            self._last_counts[name] = value
            if value > last:
                self.xruns[name] = self.xruns.get(name, 0) + value - last
                stream, _, kind = name.partition(".")
                self.publish(
                    Xrun(now, self.session_id, stream=stream, kind=kind, count=value - last)
                )
        if (
            self._last_frame_wall is not None
            and not self._lost_reported
            and now - self._last_frame_wall > self.stall_ns
        ):
            self._lost_reported = True
            seconds = (now - self._last_frame_wall) / 1e9
            reason = f"no audio for {seconds:.1f} s"
            self.publish(DeviceLost(now, self.session_id, device_id=self.device_id, reason=reason))

    # Read side

    def noise_floor_db(self) -> list[float | None]:
        return [min(ch.second_levels) if ch.second_levels else None for ch in self._channels]

    def dead_channels(self) -> list[int]:
        return [c for c, ch in enumerate(self._channels) if ch.dead_reported]

    def snapshot(self) -> dict[str, Any]:
        """Every health number, for exporters and the Studio."""
        return {
            "frames": self.frames,
            "xruns": dict(self.xruns),
            "clipping_events": self.clipping_events,
            "dead_channels": self.dead_channels(),
            "noise_floor_db": self.noise_floor_db(),
            "capture_alive": not self._lost_reported,
            **self.extra(),
        }
