"""Framer: device blocks of any size in, exact timestamped frames out.

Timestamps come from the sample count, not from each callback. Callback timestamps
jitter by milliseconds (thread scheduling, or a host API that only reports callback time
minus latency), while the sample clock is steady. So the Framer keeps a timeline anchor
``(index, t_ns)`` and stamps the frame starting at sample ``i`` with
``t_anchor + (i - i_anchor) * 1e9 / rate``, in integer nanoseconds so the result stays
exact for hours of audio.

Each push still gives an *observation* of when its first sample was captured. The
timeline moves to the observation (re-anchors) only when the two disagree by more than a
threshold for several observations in a row, so a single late callback cannot shift it.
After a gap the next disagreeing observation re-anchors at once.

Frames sit on a fixed grid of the absolute sample index: frame ``k`` covers samples
``[k * F, (k + 1) * F)`` and gets ``seq = k`` (plus any frames skipped in time, below).
When an overrun drops samples, reading resumes at the next grid line, so no frame mixes
audio from both sides of a gap and ``seq`` jumps by exactly the frames lost. When the
observed time jumps forward by a frame or more while the sample count did not (the
driver lost audio before RAL saw it), ``seq`` skips the same number of frames. Either
way, a hole in ``seq`` means missing audio and ``seq`` stays in step with ``t_ns``.

TODO(M1): replace threshold re-anchoring with a delay-locked loop that tracks clock
drift smoothly; with about 50 ppm of drift the timeline now steps by the threshold every
40 s or so.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ral.core.errors import FormatError
from ral.core.ringbuffer import Anchor, OverrunPolicy, RingBuffer
from ral.core.types import AudioFormat, AudioFrame, ChannelMap

__all__ = ["Framer"]


class Framer:
    """Turns pushed blocks into :class:`AudioFrame` objects with exact ``t_ns`` and ``seq``.

    The producer thread calls :meth:`push`; the consumer thread calls :meth:`read_frame`.

    Args:
        fmt: Format of the pushed samples and of the frames.
        channel_map: Roles of the channels, attached to every frame.
        clock_id: Which device clock produced the samples.
        capacity_s: Buffer length in seconds.
        overrun: Ring buffer overrun policy; capture uses ``"drop_oldest"``.
        reanchor_threshold_ns: How far an observation may disagree with the timeline
            before it counts as an outlier.
        reanchor_after: How many outliers in a row move the timeline.
        max_anchors: Anchors the ring buffer keeps.

    Raises:
        FormatError: If an argument is out of range or does not match ``fmt``.
    """

    def __init__(
        self,
        fmt: AudioFormat,
        channel_map: ChannelMap,
        clock_id: str,
        *,
        capacity_s: float = 2.0,
        overrun: OverrunPolicy = "drop_oldest",
        reanchor_threshold_ns: int = 2_000_000,
        reanchor_after: int = 3,
        max_anchors: int = 1024,
    ) -> None:
        if channel_map.channels != fmt.channels:
            raise FormatError(
                f"Channel map has {channel_map.channels} roles but the format has "
                f"{fmt.channels} channels."
            )
        capacity = round(capacity_s * fmt.sample_rate)
        if capacity < fmt.frame_samples:
            raise FormatError(
                f"capacity_s={capacity_s} holds less than one {fmt.frame_ms} ms frame."
            )
        if reanchor_threshold_ns <= 0:
            raise FormatError(
                f"reanchor_threshold_ns must be positive, got {reanchor_threshold_ns}."
            )
        if reanchor_after < 1:
            raise FormatError(f"reanchor_after must be at least 1, got {reanchor_after}.")
        self.fmt = fmt
        self.channel_map = channel_map
        self.clock_id = clock_id
        self.ring = RingBuffer(
            fmt.channels, fmt.dtype, capacity, overrun=overrun, max_anchors=max_anchors
        )
        self._threshold_ns = reanchor_threshold_ns
        self._reanchor_after = reanchor_after
        # Timeline anchor in "virtual" index space: the sample index plus `_offset`, the
        # samples' worth of frames skipped because time jumped forward.
        self._base_index: int | None = None
        self._base_t = 0
        self._offset = 0
        self._next_start: int | None = None
        self._outliers = 0
        self._resync = False
        self.discontinuities = 0
        """Gaps seen, from overruns or from forward time jumps."""
        self.lost_frames = 0
        """Frames skipped in ``seq`` because of those gaps."""
        self.reanchors = 0
        """Times the timeline moved to the observed time."""

    def push(self, samples: npt.NDArray[Any], t_ns: int) -> bool:
        """Producer side: append a block whose first sample was captured at ``t_ns``."""
        return self.ring.push(samples, t_ns)

    def read_frame(self) -> AudioFrame | None:
        """Consumer side: the next frame, or None if a full frame is not buffered yet."""
        fsamples = self.fmt.frame_samples
        out = np.empty((fsamples, self.fmt.channels), dtype=self.fmt.numpy_dtype)
        result = self.ring.read(out, align=fsamples)
        if result is None:
            return None
        start, anchor = result

        if self._next_start is not None and start != self._next_start:
            # Samples were dropped. seq follows the index, so it jumps by itself.
            self.discontinuities += 1
            self.lost_frames += (start - self._next_start) // fsamples
            self._resync = True
        self._next_start = start + fsamples

        if anchor is None and self._base_index is None:
            # Can only happen if the anchor FIFO overflowed before the first read; the
            # oldest remaining anchor lies after this frame but still pins the timeline.
            anchor = self.ring.peek_anchor()
        if anchor is not None:
            self._observe(start, anchor)
        # A frame is only readable after a push, and every push stores an anchor.
        assert self._base_index is not None

        index = start + self._offset
        t_ns = self._base_t + self._ns(index - self._base_index)
        return AudioFrame(out, self.fmt, self.channel_map, t_ns, index // fsamples, self.clock_id)

    def _ns(self, samples: int) -> int:
        return samples * 1_000_000_000 // self.fmt.sample_rate

    def _observe(self, start: int, anchor: Anchor) -> None:
        """Compare the anchor's view of this frame's start time with the timeline."""
        observed = anchor.t_ns + self._ns(start - anchor.sample_index)
        index = start + self._offset
        if self._base_index is None:
            self._base_index, self._base_t = index, observed
            return
        error = observed - (self._base_t + self._ns(index - self._base_index))
        if abs(error) <= self._threshold_ns:
            self._outliers = 0
            self._resync = False
            return
        self._outliers += 1
        if not self._resync and self._outliers < self._reanchor_after:
            return

        frame_ns = self.fmt.frame_ns
        if error >= frame_ns:
            # Time ran ahead of the samples by at least a frame: audio was lost upstream.
            # Skip the nearest whole number of frames (ties round down).
            skip = (error + frame_ns // 2 - 1) // frame_ns
            if skip > 0:
                self._offset += skip * self.fmt.frame_samples
                index += skip * self.fmt.frame_samples
                self.discontinuities += 1
                self.lost_frames += skip
        self._base_index, self._base_t = index, observed
        self.reanchors += 1
        self._outliers = 0
        self._resync = False
