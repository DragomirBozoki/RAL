"""CaptureGraph: the synchronous heart of the capture path.

One native frame in; for every engine frame the Normalizer produces, the graph builds a
:class:`~ral.core.processing.FrameContext` and runs the stages:

- ``reference``: every reference frame whose DAC time is before ``t + lead`` (DESIGN 5,
  "Reference alignment"), when a reference provider is attached (``software_ref``);
- ``robot_speaking``: whether playback was audible when the frame was captured.

It holds no threads and no asyncio, so the engine's DSP worker, offline replay and tests
all run exactly the same code.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ral.core.processing import FrameContext, Processor
from ral.core.protocols import ReferenceProvider
from ral.core.types import AudioFormat, AudioFrame, ChannelMap
from ral.dsp.normalizer import Normalizer

__all__ = ["CaptureGraph", "CaptureResult"]


@dataclass
class CaptureResult:
    """One engine frame after the stages: the raw normalized frame, the clean frame and
    the context the stages filled in."""

    raw: AudioFrame
    clean: AudioFrame
    ctx: FrameContext


class CaptureGraph:
    """Normalizer plus stages, with reference and speaking state per frame.

    Args:
        in_fmt: Native format of the capture frames.
        channel_map: Roles of the capture channels.
        stages: Processors in order (echo control first).
        reference: Reference provider for ``software_ref``, or None.
        lead_ms: Reference lead.
        speaking: ``speaking(t_ns)``, whether playback was audible at capture time ``t``.
        session_id: Stamped on every context.
        engine_rate: Rate of the engine frames.
    """

    def __init__(
        self,
        in_fmt: AudioFormat,
        channel_map: ChannelMap,
        stages: Sequence[Processor],
        *,
        reference: ReferenceProvider | None = None,
        lead_ms: int = 20,
        speaking: Callable[[int], bool] | None = None,
        session_id: str = "",
        engine_rate: int = 16_000,
    ) -> None:
        self.normalizer = Normalizer(in_fmt, channel_map, engine_rate)
        self.stages = list(stages)
        self.reference = reference
        self.lead_ns = lead_ms * 1_000_000
        self.speaking = speaking or (lambda _t: False)
        self.session_id = session_id
        self.frames = 0

    @property
    def engine_fmt(self) -> AudioFormat:
        return self.normalizer.out_fmt

    def process(self, frame: AudioFrame) -> list[CaptureResult]:
        """Run one native frame through; returns zero or more engine results."""
        return [self._run(f) for f in self.normalizer.process(frame)]

    def flush(self) -> list[CaptureResult]:
        """Emit what the normalizer still holds (end of a stream)."""
        return [self._run(f) for f in self.normalizer.flush()]

    def reset(self) -> None:
        self.normalizer.reset()
        for stage in self.stages:
            stage.reset()

    def _run(self, raw: AudioFrame) -> CaptureResult:
        ctx = FrameContext(
            self.session_id,
            reference=(
                self.reference.frames_until(raw.t_ns + self.lead_ns)
                if self.reference is not None
                else []
            ),
            robot_speaking=self.speaking(raw.t_ns),
        )
        clean = raw
        for stage in self.stages:
            clean = stage.process(clean, ctx)
        self.frames += 1
        return CaptureResult(raw, clean, ctx)
