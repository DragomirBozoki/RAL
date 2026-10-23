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

Drift (``drift=True``): when the sink and the mics run on different clocks, the reference
gains or loses samples against the echo. The buffer measures the sink's real rate from
the DAC times it is given, compares it with the mics' (``capture_estimator``) and
resamples the reference onto the mic clock with :class:`~ral.dsp.drift.DriftCorrector`,
starting from ``initial_ppm`` (the probe's measurement) until the estimate is ready.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import numpy.typing as npt

from ral.core.convert import int16_to_float32
from ral.core.framer import Framer
from ral.core.types import AudioFormat, AudioFrame, ChannelMap, ChannelRole
from ral.dsp.drift import DriftCorrector, DriftEstimator, drift_ppm
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
        self,
        sink_fmt: AudioFormat,
        engine_rate: int = 16_000,
        *,
        capacity_s: float = 2.0,
        drift: bool = False,
        initial_ppm: float = 0.0,
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
        self.estimator = DriftEstimator(sink_fmt.sample_rate)
        """The sink's real sample rate, from the DAC times the render hook reports."""
        self._framer.observer = self.estimator.observe
        self._framer.rate_source = self.estimator.rate_hz
        self.capture_estimator: DriftEstimator | None = None
        """The mics' real sample rate; the engine attaches it when the source reports times."""
        self.initial_ppm = initial_ppm
        self._drift = _DriftStage(self.fmt) if drift else None

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
            for out in self._normalizer.process(mono):
                if self._drift is None:
                    self._ready.append(out)
                else:
                    self._drift.ratio = self._correction_ratio()
                    self._ready.extend(self._drift.process(out))

    @property
    def drift_ppm(self) -> float:
        """How much faster the sink's clock runs than the mics', in ppm (measured once both
        estimators have enough history, ``initial_ppm`` before that)."""
        measured = (
            drift_ppm(self.estimator, self.capture_estimator)
            if self.capture_estimator is not None
            else None
        )
        return self.initial_ppm if measured is None else measured

    def _correction_ratio(self) -> float:
        return 1.0 / (1.0 + self.drift_ppm * 1e-6)

    @staticmethod
    def _downmix(data: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        x = int16_to_float32(data) if data.dtype == np.int16 else data.astype(np.float32)
        if x.shape[1] == 1:
            return x
        mean: npt.NDArray[np.float32] = x.mean(axis=1, keepdims=True, dtype=np.float32)
        return mean


class _DriftStage:
    """Re-cuts engine-rate reference frames after a drift correction, keeping DAC times."""

    def __init__(self, fmt: AudioFormat) -> None:
        self.fmt = fmt
        self.corrector = DriftCorrector()
        self.reset()

    @property
    def ratio(self) -> float:
        return self.corrector.ratio

    @ratio.setter
    def ratio(self, value: float) -> None:
        # Small steps only: the estimate is smooth, and a jump would click the reference.
        current = self.corrector.ratio
        step = 5e-6
        self.corrector.ratio = min(current + step, max(current - step, value))

    def reset(self) -> None:
        self.corrector.reset()
        self._inputs: deque[tuple[int, int]] = deque()  # (absolute start sample, t_ns)
        self._consumed = 0
        self._fifo: npt.NDArray[np.float32] = np.zeros(0, np.float32)
        self._fifo_pos: npt.NDArray[np.float64] = np.zeros(0, np.float64)
        self._next_seq: int | None = None
        self._out_seq = 0

    def process(self, frame: AudioFrame) -> list[AudioFrame]:
        if self._next_seq is not None and frame.seq != self._next_seq:
            self.reset()  # a gap: start a new run, the old positions no longer apply
        self._next_seq = frame.seq + 1
        self._inputs.append((self._consumed, frame.t_ns))
        self._consumed += frame.fmt.frame_samples
        out = self.corrector.process(frame.data[:, 0])
        self._fifo = np.concatenate([self._fifo, out])
        self._fifo_pos = np.concatenate([self._fifo_pos, self.corrector.positions])
        frames = []
        n = self.fmt.frame_samples
        while self._fifo.shape[0] >= n:
            data = self._fifo[:n].reshape(n, 1).copy()
            t_ns = self._time_of(float(self._fifo_pos[0]))
            self._fifo, self._fifo_pos = self._fifo[n:], self._fifo_pos[n:]
            frames.append(
                AudioFrame(data, self.fmt, frame.channel_map, t_ns, self._out_seq, frame.clock_id)
            )
            self._out_seq += 1
        return frames

    def _time_of(self, position: float) -> int:
        """DAC time of an input position, from the input frame that contains it."""
        while len(self._inputs) > 1 and self._inputs[1][0] <= position:
            self._inputs.popleft()
        start, t_ns = self._inputs[0]
        return t_ns + round((position - start) * 1e9 / self.fmt.sample_rate)
