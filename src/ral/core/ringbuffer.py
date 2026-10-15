"""Single-producer, single-consumer ring buffer for audio samples.

One thread (an audio callback) pushes blocks of any size; one other thread reads them.
Positions are absolute sample indices: ``write_index`` counts every sample ever pushed
into the stream's timeline and ``read_index`` how far the consumer has got. Each push
also records an *anchor*, the index and timestamp of the block's first sample, so the
consumer can work out when any sample was captured.

Why a lock instead of a lock-free design: with the ``drop_oldest`` overrun policy the
producer moves the consumer's read index, so both threads write shared state. A
``threading.Lock`` held only around a few slice copies keeps that correct. At 100 frames
a second per stream the lock is uncontended almost always and held for microseconds,
which is well inside what an audio callback can afford. Everything is allocated up front;
a push makes no buffer allocation, only ``np.copyto`` into one or two slices (numpy still
creates small view objects for the slices).
"""

from __future__ import annotations

import threading
from typing import Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from ral.core.errors import FormatError

__all__ = ["Anchor", "OverrunPolicy", "ReadResult", "RingBuffer"]

OverrunPolicy = Literal["drop_oldest", "drop_newest"]


class Anchor(NamedTuple):
    """The absolute index of a pushed block's first sample and its timestamp."""

    sample_index: int
    t_ns: int


class ReadResult(NamedTuple):
    """Where a read started, and the newest anchor at or before its last sample."""

    start: int
    anchor: Anchor | None


class RingBuffer:
    """Fixed-capacity sample buffer shared by one producer thread and one consumer thread.

    Args:
        channels: Channels per sample.
        dtype: Sample dtype, for example ``"int16"`` or ``"float32"``.
        capacity: Samples per channel the buffer holds.
        overrun: What a push does when the buffer is full. ``"drop_oldest"`` (capture)
            discards the oldest unread samples so latency stays bounded and the read index
            jumps forward. ``"drop_newest"`` (playback) rejects the whole new block.
        max_anchors: How many block anchors to keep; when full, the oldest is dropped.

    Raises:
        FormatError: If an argument is out of range.
    """

    def __init__(
        self,
        channels: int,
        dtype: str | np.dtype[Any],
        capacity: int,
        *,
        overrun: OverrunPolicy = "drop_oldest",
        max_anchors: int = 1024,
    ) -> None:
        if channels <= 0:
            raise FormatError(f"channels must be at least 1, got {channels}.")
        if capacity <= 0:
            raise FormatError(f"capacity must be at least 1 sample, got {capacity}.")
        if overrun not in ("drop_oldest", "drop_newest"):
            raise FormatError(f"overrun must be 'drop_oldest' or 'drop_newest', got {overrun!r}.")
        if max_anchors <= 0:
            raise FormatError(f"max_anchors must be at least 1, got {max_anchors}.")
        self._buf: npt.NDArray[Any] = np.zeros((capacity, channels), dtype=dtype)
        self._capacity = capacity
        self._channels = channels
        self._overrun = overrun
        self._lock = threading.Lock()
        self._write = 0
        self._read = 0
        # Anchors live in a fixed FIFO of plain lists, so a push stores two ints and
        # allocates nothing.
        self._anchor_index = [0] * max_anchors
        self._anchor_t = [0] * max_anchors
        self._anchor_head = 0
        self._anchor_count = 0
        self.overruns = 0
        """Pushes that did not fit (data was dropped)."""
        self.dropped_samples = 0
        """Samples lost to overruns, per channel."""
        self.underruns = 0
        """Partial reads that had to be padded with silence."""

    @property
    def capacity(self) -> int:
        """Samples per channel the buffer holds."""
        return self._capacity

    @property
    def dtype(self) -> np.dtype[Any]:
        """Sample dtype."""
        return self._buf.dtype

    @property
    def write_index(self) -> int:
        """Absolute index one past the newest sample."""
        return self._write

    @property
    def read_index(self) -> int:
        """Absolute index of the oldest unread sample."""
        return self._read

    @property
    def available(self) -> int:
        """Unread samples per channel."""
        with self._lock:
            return self._write - self._read

    # Producer side

    def push(self, samples: npt.NDArray[Any], t_ns: int) -> bool:
        """Append a block; called from the producer thread.

        Args:
            samples: Shape ``(n, channels)`` with the buffer's dtype.
            t_ns: Monotonic timestamp of the block's first sample.

        Returns:
            False if the block was rejected (``drop_newest`` and full), else True.

        Raises:
            FormatError: If ``samples`` has the wrong shape or dtype.
        """
        if samples.ndim != 2 or samples.shape[1] != self._channels:
            raise FormatError(f"Block has shape {samples.shape}, expected (n, {self._channels}).")
        if samples.dtype != self._buf.dtype:
            raise FormatError(f"Block has dtype {samples.dtype}, expected {self._buf.dtype}.")
        n: int = samples.shape[0]
        if n == 0:
            return True
        cap = self._capacity
        with self._lock:
            block_index = self._write
            free = cap - (self._write - self._read)
            if n > free:
                if self._overrun == "drop_newest":
                    self.overruns += 1
                    self.dropped_samples += n
                    return False
                self.overruns += 1
                if n > cap:
                    # Only the newest `cap` samples can survive: all unread data and the
                    # head of the block are lost, and reading resumes at the kept part.
                    skip = n - cap
                    samples = samples[skip:]
                    n = cap
                    self.dropped_samples += (self._write - self._read) + skip
                    self._write += skip
                    self._read = self._write
                else:
                    lost = n - free
                    self._read += lost
                    self.dropped_samples += lost
            self._copy_in(self._write, samples)
            self._write += n
            self._add_anchor(block_index, t_ns)
        return True

    # Consumer side

    def read(self, out: npt.NDArray[Any], *, align: int = 1) -> ReadResult | None:
        """Read exactly ``len(out)`` samples into ``out``, or nothing.

        Args:
            out: Destination of shape ``(n, channels)``.
            align: The read starts at the first multiple of ``align`` at or after the read
                index; samples skipped to get there are discarded. The Framer passes the
                frame length so frames stay on a fixed grid after an overrun.

        Returns:
            The absolute start index and newest anchor at or before the last sample read,
            or None (and nothing consumed) if not enough samples are buffered yet.
        """
        n: int = out.shape[0]
        with self._lock:
            start = -(-self._read // align) * align
            if self._write - start < n:
                return None
            self._copy_out(start, out)
            self._read = start + n
            anchor = self._pop_anchors_before(start + n)
        return ReadResult(start, anchor)

    def read_available(self, out: npt.NDArray[Any]) -> int:
        """Fill ``out`` with what is buffered and pad the rest with zeros.

        For a playback callback that must hand the device a full block every time. A
        short read counts as an underrun.

        Returns:
            How many samples per channel came from the buffer.
        """
        n: int = out.shape[0]
        with self._lock:
            k = min(n, self._write - self._read)
            self._copy_out(self._read, out[:k])
            self._read += k
            self._pop_anchors_before(self._read)
            if k < n:
                out[k:] = 0
                self.underruns += 1
        return k

    def peek_anchor(self) -> Anchor | None:
        """The oldest anchor still stored, without removing it."""
        with self._lock:
            if self._anchor_count == 0:
                return None
            h = self._anchor_head
            return Anchor(self._anchor_index[h], self._anchor_t[h])

    # Internals; callers hold the lock.

    def _copy_in(self, index: int, samples: npt.NDArray[Any]) -> None:
        n = samples.shape[0]
        pos = index % self._capacity
        first = min(n, self._capacity - pos)
        np.copyto(self._buf[pos : pos + first], samples[:first])
        if first < n:
            np.copyto(self._buf[: n - first], samples[first:])

    def _copy_out(self, index: int, out: npt.NDArray[Any]) -> None:
        n: int = out.shape[0]
        pos = index % self._capacity
        first = min(n, self._capacity - pos)
        np.copyto(out[:first], self._buf[pos : pos + first])
        if first < n:
            np.copyto(out[first:], self._buf[: n - first])

    def _add_anchor(self, index: int, t_ns: int) -> None:
        size = len(self._anchor_index)
        if self._anchor_count == size:
            self._anchor_head = (self._anchor_head + 1) % size
            self._anchor_count -= 1
        pos = (self._anchor_head + self._anchor_count) % size
        self._anchor_index[pos] = index
        self._anchor_t[pos] = t_ns
        self._anchor_count += 1

    def _pop_anchors_before(self, end: int) -> Anchor | None:
        size = len(self._anchor_index)
        latest: Anchor | None = None
        while self._anchor_count and self._anchor_index[self._anchor_head] < end:
            h = self._anchor_head
            latest = Anchor(self._anchor_index[h], self._anchor_t[h])
            self._anchor_head = (h + 1) % size
            self._anchor_count -= 1
        return latest
