"""The capture graph's interface: one :class:`Processor` per stage, sharing a context.

See ``docs/DESIGN.md`` section 4. The engine builds a :class:`FrameContext` for every
engine frame (the reference frames due for it, whether the robot was speaking when it
was captured) and runs the stages in order; each stage returns the frame for the next.

``info`` carries per-frame measurements between stages, for example the echo controller's
input and output levels, which the barge-in detector compares. Keys are documented by the
stage that writes them; values are floats, levels in dBFS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ral.core.events import Event
from ral.core.types import AudioFrame

__all__ = ["FrameContext", "Processor"]


@dataclass
class FrameContext:
    """Per-frame state shared by the stages of the capture graph."""

    session_id: str
    reference: list[AudioFrame] = field(default_factory=list)
    """Reference frames to feed echo control before this mic frame, oldest first."""
    robot_speaking: bool = False
    """Whether RAL's (or, in ``gate`` mode, any known) playback was audible at capture time."""
    events: list[Event] = field(default_factory=list)
    """Stages append events here; the engine publishes them."""
    info: dict[str, float] = field(default_factory=dict)
    """Measurements stages share, such as ``"mic_db"`` and ``"out_db"``."""


@runtime_checkable
class Processor(Protocol):
    """One stage of the capture graph."""

    def process(self, frame: AudioFrame, ctx: FrameContext) -> AudioFrame: ...

    def reset(self) -> None: ...
