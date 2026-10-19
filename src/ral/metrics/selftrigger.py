"""Self-trigger rate: how often the robot "hears itself" (DESIGN section 8).

A self-trigger is a ``SpeechStarted`` during the robot's own playback that no real user
speech explains. The rate is self-triggers per minute of playback, so sessions with more
robot speech are comparable with sessions with less.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

__all__ = ["self_trigger_rate", "self_triggers"]

Interval = tuple[int, int]


def _inside(t: int, spans: Sequence[Interval], tolerance_ns: int = 0) -> bool:
    return any(a - tolerance_ns <= t < b + tolerance_ns for a, b in spans)


def self_triggers(
    speech_starts: Iterable[int],
    playback: Sequence[Interval],
    user_speech: Sequence[Interval] = (),
    *,
    tolerance_ns: int = 300_000_000,
) -> list[int]:
    """Speech starts inside playback that do not fall within ``tolerance_ns`` of user speech."""
    return [
        t
        for t in speech_starts
        if _inside(t, playback) and not _inside(t, user_speech, tolerance_ns)
    ]


def self_trigger_rate(
    speech_starts: Iterable[int],
    playback: Sequence[Interval],
    user_speech: Sequence[Interval] = (),
    *,
    tolerance_ns: int = 300_000_000,
) -> float:
    """Self-triggers per minute of playback (0.0 when nothing played)."""
    minutes = sum(b - a for a, b in playback) / 60e9
    if minutes <= 0:
        return 0.0
    count = len(self_triggers(speech_starts, playback, user_speech, tolerance_ns=tolerance_ns))
    return count / minutes
