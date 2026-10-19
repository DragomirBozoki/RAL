"""Typed events the engine publishes for bridges, the recorder and robot behaviour.

See ``docs/DESIGN.md`` section 4 ("Events"). Every event is a frozen dataclass deriving
from :class:`Event`, carries ``t_ns`` (RAL clock) and the session id, and serialises to
JSON as ``{"type": "<ClassName>", ...}``. Subclasses defined elsewhere (plugins, later
milestones) are found automatically by :func:`event_from_json`.

What ``t_ns`` means differs per event and is documented on each class; for audio events
it is always the time the sound was at the ADC or DAC, not when the event was emitted.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from ral.core.errors import FormatError

__all__ = [
    "BargeIn",
    "Clipping",
    "DeviceLost",
    "DirectionOfArrival",
    "EchoLeak",
    "Event",
    "PlaybackEnded",
    "PlaybackStarted",
    "ProfileReloaded",
    "RecordingStarted",
    "RecordingStopped",
    "SpeechEnded",
    "SpeechStarted",
    "TurnLatency",
    "WordSpoken",
    "Xrun",
    "event_from_json",
    "event_to_json",
]


@dataclass(frozen=True)
class Event:
    """Base class: when it happened on the RAL clock, and in which session."""

    t_ns: int
    session_id: str


@dataclass(frozen=True)
class SpeechStarted(Event):
    """Speech began in the clean capture stream; ``t_ns`` is the first speech sample."""


@dataclass(frozen=True)
class SpeechEnded(Event):
    """Speech ended; ``t_ns`` is the last speech sample, before any VAD hangover."""

    t_start_ns: int


@dataclass(frozen=True)
class BargeIn(Event):
    """A user interrupted the robot; ``t_ns`` is when the interrupting speech began."""

    interrupted_tag: str | None


@dataclass(frozen=True)
class PlaybackStarted(Event):
    """A playback item's first sample is estimated at the DAC at ``t_ns``."""

    handle_id: int
    tag: str
    t_request_ns: int  # when the first audio of the item reached the playback service


@dataclass(frozen=True)
class PlaybackEnded(Event):
    """A playback item's last sample (after any fade) is estimated at the DAC at ``t_ns``."""

    handle_id: int
    tag: str
    interrupted: bool


@dataclass(frozen=True)
class WordSpoken(Event):
    """A word reported by TTS, re-stamped to DAC time; ``t_ns`` is its start.

    Named ``WordSpoken`` rather than ``WordTiming`` so it does not clash with the TTS
    port's ``WordTiming`` record.
    """

    word: str
    t_end_ns: int
    handle_id: int


@dataclass(frozen=True)
class DirectionOfArrival(Event):
    """Direction of the dominant talker, from an array that reports it."""

    azimuth_deg: float
    elevation_deg: float | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class EchoLeak(Event):
    """Robot speech was detected in the cleaned signal (it correlates with the reference)."""

    correlation: float


@dataclass(frozen=True)
class DeviceLost(Event):
    """A capture or playback device stopped delivering audio."""

    device_id: str
    reason: str


@dataclass(frozen=True)
class Xrun(Event):
    """Audio was lost (``overrun``) or silence was inserted (``underrun``) on a stream."""

    stream: str
    kind: str
    count: int


@dataclass(frozen=True)
class Clipping(Event):
    """A channel hit full scale on a share of the samples in the last interval."""

    channel: int
    share: float


@dataclass(frozen=True)
class ProfileReloaded(Event):
    """The engine switched to a new profile without restarting audio."""

    profile: str


@dataclass(frozen=True)
class RecordingStarted(Event):
    """Recording began (recording is opt-in; DESIGN section 8, "Privacy")."""

    path: str


@dataclass(frozen=True)
class RecordingStopped(Event):
    """Recording ended."""

    path: str


@dataclass(frozen=True)
class TurnLatency(Event):
    """How long one turn took, from the end of user speech to robot audio at the DAC.

    ``t_ns`` is the end of the user's speech. The stages add up to ``total_ms``:
    ``hangover_ms`` (VAD hangover plus processing until RAL decided the user had finished),
    ``agent_ms`` (from that decision until the reply's first audio reached RAL) and
    ``playback_ms`` (from there until that audio was at the DAC).
    """

    total_ms: float
    hangover_ms: float
    agent_ms: float
    playback_ms: float
    tag: str


def event_to_json(event: Event) -> dict[str, Any]:
    """``{"type": "<ClassName>", **fields}``, with only JSON types as values."""
    out: dict[str, Any] = {"type": type(event).__name__}
    for f in dataclasses.fields(event):
        value = getattr(event, f.name)
        out[f.name] = list(value) if isinstance(value, tuple) else value
    return out


def _all_event_types() -> dict[str, type[Event]]:
    found: dict[str, type[Event]] = {}
    stack: list[type[Event]] = [Event]
    while stack:
        cls = stack.pop()
        found[cls.__name__] = cls
        stack.extend(cls.__subclasses__())
    return found


def event_from_json(data: dict[str, Any]) -> Event:
    """Rebuild an event from :func:`event_to_json` output.

    Raises:
        FormatError: If the type is unknown or fields are missing or unexpected.
    """
    payload = dict(data)
    name = payload.pop("type", None)
    cls = _all_event_types().get(str(name))
    if cls is None or cls is Event:
        raise FormatError(f"Unknown event type {name!r}; is the plugin that defines it loaded?")
    names = {f.name for f in dataclasses.fields(cls)}
    unexpected = set(payload) - names
    if unexpected:
        raise FormatError(f"{name} has no fields {sorted(unexpected)}.")
    try:
        return cls(**payload)
    except TypeError as exc:
        raise FormatError(f"Cannot build {name} from {data!r}: {exc}") from exc
