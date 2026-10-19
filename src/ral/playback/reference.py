"""ReferenceBuffer: everything RAL played, on the mic's rate, for the echo canceller.

The sink's render hook calls :meth:`ReferenceBuffer.push` with every block handed to the
device, silence included, and its estimated DAC time (DESIGN section 5, "Reference
alignment"). The push runs on the output thread, so it only copies the block into a
:class:`~ral.core.Framer`'s ring buffer, which is real-time safe.

On the DSP thread, :meth:`frames_until` turns those blocks into 10 ms frames, downmixes
them to mono (the speaker plays one signal; a stereo sink plays it on both sides), and
resamples them to the engine rate with the same :class:`~ral.dsp.normalizer.Normalizer` the
mics use. Frames carry their DAC time as ``t_ns``, stamped from the sample count like
capture frames, so render-callback jitter does not reach the echo canceller.

Frames are handed out by DAC time: the ``software_ref`` echo controller asks for every
frame earlier than ``t_mic + lead``. A frame that is not yet due stays buffered.

TODO(M3): drift compensation. When the sink and the mics run on different clocks, the
reference has to be resampled onto the mic clock with a slowly varying ratio; until then
the nominal ratio is used.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import numpy.typing as npt

from ral.core.convert import int16_to_float32
from ral.core.framer import Framer
from ral.core.types import AudioFormat, AudioFrame, ChannelMap, ChannelRole
from ral.dsp.normalizer import Normalizer

__all__ = ["REFERENCE_CLOCK_ID", "ReferenceBuffer"]

REFERENCE_CLOCK_ID = "dac"


class ReferenceBuffer:
    """Implements :class:`~ral.core.ReferenceProvider` for one sink.

    Args:
        sink_fmt: Format of the blocks the sink renders (its rate, channels and dtype).
        engine_rate: Rate of the frames handed to echo control.
        capacity_s: Rendered audio the buffer holds before the oldest is dropped.
    """

    def __init__(
        self, sink_fmt: AudioFormat, engine_rate: int = 16_000, *, capacity_s: float = 2.0
    ) -> None:
        self.sink_fmt = sink_fmt
        self._framer = Framer(
            sink_fmt,
            ChannelMap.uniform(sink_fmt.channels, ChannelRole.REFERENCE),
            REFERENCE_CLOCK_ID,
            capacity_s=capacity_s,
            overrun="drop_oldest",
        )
        mono = AudioFormat(sink_fmt.sample_rate, 1, "float32", sink_fmt.frame_ms)
        self._mono_fmt = mono
        self._mono_map = ChannelMap((ChannelRole.REFERENCE,))
        self._normalizer = Normalizer(mono, self._mono_map, engine_rate)
        self.fmt = self._normalizer.out_fmt
        """Format of the frames :meth:`frames_until` returns (mono float32)."""
        self._ready: deque[AudioFrame] = deque()

    @property
    def overruns(self) -> int:
        """Pushes that overflowed the buffer because nobody read it."""
        return self._framer.ring.overruns

    @property
    def discontinuities(self) -> int:
        """Gaps in the rendered timeline (lost or skipped audio)."""
        return self._framer.discontinuities

    def push(self, samples: npt.NDArray[Any], t_dac_ns: int) -> None:
        """Render hook: copy a rendered block and its DAC time. Real-time safe."""
        self._framer.push(samples, t_dac_ns)

    def frames_until(self, t_ns: int) -> list[AudioFrame]:
        """Every frame whose DAC time is earlier than ``t_ns``, oldest first."""
        self._pull()
        out: list[AudioFrame] = []
        while self._ready and self._ready[0].t_ns < t_ns:
            out.append(self._ready.popleft())
        return out

    def drain(self) -> list[AudioFrame]:
        """Every frame available now, whatever its DAC time."""
        self._pull()
        out = list(self._ready)
        self._ready.clear()
        return out

    def _pull(self) -> None:
        while (frame := self._framer.read_frame()) is not None:
            mono = AudioFrame(
                self._downmix(frame.data),
                self._mono_fmt,
                self._mono_map,
                frame.t_ns,
                frame.seq,
                frame.clock_id,
            )
            self._ready.extend(self._normalizer.process(mono))

    @staticmethod
    def _downmix(data: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        x = int16_to_float32(data) if data.dtype == np.int16 else data.astype(np.float32)
        if x.shape[1] == 1:
            return x
        mean: npt.NDArray[np.float32] = x.mean(axis=1, keepdims=True, dtype=np.float32)
        return mean
