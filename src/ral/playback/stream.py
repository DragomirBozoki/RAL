"""PcmStream: audio pushed in chunks (from a socket or an agent) as a playback item.

Bridges receive agent audio in pieces of any length. A :class:`PcmStream` collects them,
cuts exact engine frames and yields them to the playback service as one item, so an
utterance streamed over the network is interruptible and fades like any other. The
producer calls :meth:`push` and finally :meth:`end`; the last partial frame is padded
with silence.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import numpy.typing as npt

from ral.core.convert import int16_to_float32
from ral.core.errors import FormatError
from ral.core.types import AudioFormat, AudioFrame, ChannelMap
from ral.dsp.resample import Resampler

__all__ = ["PcmStream"]


class PcmStream:
    """An async iterator of engine frames fed by :meth:`push`.

    Args:
        rate: Rate of the pushed audio; it is resampled to the engine rate if different.
        fmt: The engine's playback format (mono float32).
    """

    def __init__(self, rate: int = 16_000, fmt: AudioFormat | None = None) -> None:
        self.fmt = fmt or AudioFormat(16_000, 1, "float32")
        self._resampler = (
            None if rate == self.fmt.sample_rate else Resampler(rate, self.fmt.sample_rate, 1)
        )
        self._buf: npt.NDArray[np.float32] = np.zeros(0, np.float32)
        self._frames: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._map = ChannelMap.uniform(1)
        self._seq = 0
        self.ended = False
        self.samples_in = 0

    def push(self, samples: npt.NDArray[np.int16] | npt.NDArray[np.float32] | bytes) -> None:
        """Add mono audio: int16 PCM bytes (little-endian), an int16 array or a float32 array."""
        if self.ended:
            raise FormatError("This stream has ended; start a new one.")
        if isinstance(samples, bytes | bytearray | memoryview):
            if len(samples) % 2:
                raise FormatError("PCM bytes must hold whole 16-bit samples.")
            samples = np.frombuffer(samples, "<i2").astype(np.int16)
        x = np.asarray(samples).reshape(-1)
        f = int16_to_float32(x) if x.dtype == np.int16 else x.astype(np.float32)
        self.samples_in += f.shape[0]
        if self._resampler is not None:
            f = self._resampler.process(f[:, None])[:, 0]
        self._buf = np.concatenate([self._buf, f])
        self._emit()

    def end(self) -> None:
        """No more audio: flush the last partial frame and finish the item."""
        if self.ended:
            return
        if self._resampler is not None:
            self._buf = np.concatenate([self._buf, self._resampler.flush()[:, 0]])
        n = self.fmt.frame_samples
        if self._buf.shape[0]:
            pad = (-self._buf.shape[0]) % n
            self._buf = np.concatenate([self._buf, np.zeros(pad, np.float32)])
            self._emit()
        self.ended = True
        self._frames.put_nowait(None)

    def _emit(self) -> None:
        n = self.fmt.frame_samples
        while self._buf.shape[0] >= n:
            block = self._buf[:n].reshape(n, 1).copy()
            self._buf = self._buf[n:]
            self._frames.put_nowait(AudioFrame(block, self.fmt, self._map, 0, self._seq, "stream"))
            self._seq += 1

    def __aiter__(self) -> AsyncIterator[AudioFrame]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[AudioFrame]:
        while (frame := await self._frames.get()) is not None:
            yield frame
