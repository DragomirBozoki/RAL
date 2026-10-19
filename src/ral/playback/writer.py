"""SinkWriter: engine frames in, sink frames out, with pacing and a DAC-time cursor.

The playback service produces mono float32 frames at the engine rate. The writer
resamples them to the sink's rate, copies the mono signal to every sink channel, converts
to the sink's dtype and writes whole sink frames.

Pacing. ``AudioSink.write`` never blocks and drops frames when the sink's buffer is full,
so the writer must not run ahead. In realtime mode it keeps the audio it has handed over
at most ``ahead_ms`` past what the device is about to play, waiting in
:meth:`wait_turn` otherwise. That bounds the delay between a stop request and silence
(an interrupt cannot recall audio already queued) while leaving room for scheduling
jitter. Where asyncio's clock is coarse (Windows before Python 3.13) the lead grows to
four timer ticks, about 62 ms. In offline mode (a non-realtime ``FileSink``) it never waits.

DAC time. The writer keeps a cursor: the estimated DAC time of the next sample it
writes. When audio is written after the sink ran dry, it plays one output latency from
now, so the cursor jumps to ``now + output_latency``; while audio flows it advances by
10 ms per frame. ``PlaybackStarted`` and the speaking state come from this cursor.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import numpy.typing as npt

from ral.core.clock import now_ns
from ral.core.convert import float32_to_int16
from ral.core.errors import FormatError
from ral.core.protocols import AudioSink
from ral.core.types import AudioFormat, AudioFrame, ChannelMap
from ral.dsp.resample import Resampler

__all__ = ["SinkWriter"]


class SinkWriter:
    """Converts and paces frames for one sink.

    Args:
        sink: The started (or soon started) playback device.
        in_fmt: Format of the frames the playback service writes (mono float32).
        realtime: Pace writes by the clock; False for offline sinks.
        ahead_ms: How far past the device's latency the writer may queue audio.

    Raises:
        FormatError: If ``in_fmt`` is not mono float32 or frame lengths differ.
    """

    def __init__(
        self, sink: AudioSink, in_fmt: AudioFormat, *, realtime: bool, ahead_ms: int = 40
    ) -> None:
        if in_fmt.channels != 1 or in_fmt.dtype != "float32":
            raise FormatError(f"The playback service writes mono float32 frames, not {in_fmt}.")
        if in_fmt.frame_ms != sink.fmt.frame_ms:
            raise FormatError("Engine and sink frames must have the same length.")
        self.sink = sink
        self.in_fmt = in_fmt
        self.out_fmt = sink.fmt
        self.realtime = realtime
        # asyncio sleeps on time.monotonic; where that clock is coarse (15.6 ms on Windows
        # before Python 3.13) a short sleep can overshoot by several ticks, so the lead
        # must cover a few of them or the device runs dry between writes.
        tick_ns = round(time.get_clock_info("monotonic").resolution * 1e9)
        self._ahead_ns = max(ahead_ms * 1_000_000, 4 * tick_ns)
        self._resampler = Resampler(in_fmt.sample_rate, sink.fmt.sample_rate, 1)
        self._map = ChannelMap.uniform(sink.fmt.channels)
        self._fifo: npt.NDArray[np.float32] = np.zeros((0, 1), dtype=np.float32)
        self._cursor: int | None = None
        self._seq = 0
        self.frames_written = 0

    def start(self) -> None:
        """Start the DAC-time cursor; call right after the sink starts."""
        self._cursor = now_ns() + self.sink.output_latency_ns()

    @property
    def cursor_ns(self) -> int:
        """Estimated DAC time of the next sample written (after any idle gap)."""
        if self._cursor is None:
            self.start()
        assert self._cursor is not None
        if self.realtime:
            return max(self._cursor, now_ns() + self.sink.output_latency_ns())
        return self._cursor

    async def wait_turn(self) -> None:
        """Wait until another frame may be written without running too far ahead."""
        if not self.realtime or self._cursor is None:
            return
        while (lead := self._cursor - (now_ns() + self.sink.output_latency_ns())) > (
            self._ahead_ns
        ):
            await asyncio.sleep((lead - self._ahead_ns) / 1e9 + 0.001)

    def write(self, samples: npt.NDArray[np.float32]) -> int:
        """Write one engine frame of ``(frame_samples, 1)``; returns its DAC time."""
        if samples.shape != (self.in_fmt.frame_samples, 1):
            raise FormatError(
                f"Expected ({self.in_fmt.frame_samples}, 1) samples, got {samples.shape}."
            )
        t_dac = self.cursor_ns
        self._cursor = t_dac + self.in_fmt.frame_ns
        self._push(self._resampler.process(samples))
        return t_dac

    def idle(self) -> None:
        """Flush the resampler's tail when playback pauses, so nothing is held back."""
        tail = self._resampler.flush()
        if tail.shape[0] == 0 and self._fifo.shape[0] == 0:
            return
        self._push(tail)
        if self._fifo.shape[0]:
            pad = self.out_fmt.frame_samples - self._fifo.shape[0]
            self._push(np.zeros((pad, 1), np.float32))

    def _push(self, samples: npt.NDArray[np.float32]) -> None:
        if samples.shape[0]:
            self._fifo = np.concatenate([self._fifo, samples]) if self._fifo.size else samples
        n = self.out_fmt.frame_samples
        while self._fifo.shape[0] >= n:
            mono = self._fifo[:n]
            self._fifo = self._fifo[n:]
            data = np.repeat(mono, self.out_fmt.channels, axis=1)
            if self.out_fmt.dtype == "int16":
                out = float32_to_int16(data)
            else:
                out = np.ascontiguousarray(data, dtype=np.float32)
            t_ns = self._cursor or 0
            self.sink.write(AudioFrame(out, self.out_fmt, self._map, t_ns, self._seq, "playback"))
            self._seq += 1
            self.frames_written += 1
