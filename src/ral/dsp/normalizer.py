"""Normalizer: native device frames in, engine frames out.

Step 3 of the capture chain (DESIGN section 5): convert to float32, resample to the
engine rate and apply the channel map. Channels marked ``unused`` are dropped; mic,
reference and processed channels keep their order and roles, and all of them go through
the same resampler, so reference channels stay sample-aligned with the mics.

Timing. Input and output frames are both 10 ms long, and the resampler is aligned so that
output sample ``k`` estimates the signal at input time ``k * M / L`` (see
:mod:`ral.dsp.resample`). Output frame ``i`` of a run therefore covers exactly the time of
input frame ``i`` and inherits its ``t_ns`` and ``seq``. The resampler's lookahead only
delays *when* an output frame becomes available (by one input frame at most for the
default filter), never its timestamp.

Gaps. When the input ``seq`` jumps, the run ends: the frames still inside the resampler
are flushed (padded with silence) and a new run starts at the next frame, so no output
frame mixes audio from both sides of a gap and ``seq`` keeps its hole.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import numpy.typing as npt

from ral.core.convert import int16_to_float32
from ral.core.errors import FormatError
from ral.core.types import AudioFormat, AudioFrame, ChannelMap, ChannelRole
from ral.dsp.resample import Resampler

__all__ = ["Normalizer"]


class Normalizer:
    """Converts frames of one input format into float32 frames at the engine rate.

    Args:
        in_fmt: Format of the incoming frames (the device's native format).
        channel_map: Roles of the incoming channels.
        engine_rate: Output sample rate, 16 kHz by default.

    Raises:
        FormatError: If the channel map does not match ``in_fmt``, keeps no channel, or the
            frame lengths differ.
    """

    def __init__(
        self, in_fmt: AudioFormat, channel_map: ChannelMap, engine_rate: int = 16_000
    ) -> None:
        if channel_map.channels != in_fmt.channels:
            raise FormatError(
                f"Channel map has {channel_map.channels} roles but the format has "
                f"{in_fmt.channels} channels."
            )
        keep = tuple(i for i, r in enumerate(channel_map.roles) if r is not ChannelRole.UNUSED)
        if not keep:
            raise FormatError("Every channel is 'unused'; the channel map must keep at least one.")
        self.in_fmt = in_fmt
        self.in_map = channel_map
        self.out_fmt = AudioFormat(engine_rate, len(keep), "float32", in_fmt.frame_ms)
        self.out_map = ChannelMap(
            tuple(channel_map.roles[i] for i in keep),
            None
            if channel_map.positions_m is None
            else tuple(channel_map.positions_m[i] for i in keep),
        )
        self._keep: list[int] | None = None if len(keep) == in_fmt.channels else list(keep)
        self._resampler = Resampler(in_fmt.sample_rate, engine_rate, len(keep))
        self._pending: deque[tuple[int, int]] = deque()  # (seq, t_ns) awaiting output
        self._out = np.zeros((0, len(keep)), dtype=np.float32)
        self._next_seq: int | None = None
        self._clock_id = ""
        self.gaps = 0
        """Input discontinuities seen (each ends a resampler run)."""

    def process(self, frame: AudioFrame) -> list[AudioFrame]:
        """Feed one native frame; returns the engine frames that are now complete.

        Raises:
            FormatError: If ``frame`` does not have the configured input format.
        """
        if frame.fmt != self.in_fmt:
            raise FormatError(f"Frame format {frame.fmt} does not match {self.in_fmt}.")
        out: list[AudioFrame] = []
        if self._next_seq is not None and frame.seq != self._next_seq:
            self.gaps += 1
            out.extend(self.flush())
        self._next_seq = frame.seq + 1
        self._clock_id = frame.clock_id
        self._pending.append((frame.seq, frame.t_ns))
        samples = self._convert(frame.data)
        self._emit(self._resampler.process(samples), out)
        return out

    def flush(self) -> list[AudioFrame]:
        """End the current run: emit what the resampler still holds, padded with silence."""
        out: list[AudioFrame] = []
        self._emit(self._resampler.flush(), out)
        # A partial last frame (only possible with upsampling) is padded with zeros.
        if self._pending and self._out.shape[0] > 0:
            pad = self.out_fmt.frame_samples - self._out.shape[0]
            self._out = np.concatenate(
                [self._out, np.zeros((pad, self.out_fmt.channels), np.float32)]
            )
            self._emit(np.zeros((0, self.out_fmt.channels), np.float32), out)
        self._pending.clear()
        self._out = np.zeros((0, self.out_fmt.channels), dtype=np.float32)
        self._next_seq = None
        return out

    def reset(self) -> None:
        """Drop all state without emitting anything."""
        self._resampler.reset()
        self._pending.clear()
        self._out = np.zeros((0, self.out_fmt.channels), dtype=np.float32)
        self._next_seq = None

    def _convert(self, data: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        selected = data if self._keep is None else data[:, self._keep]
        if selected.dtype == np.int16:
            return int16_to_float32(selected)
        return selected.astype(np.float32, copy=False)

    def _emit(self, samples: npt.NDArray[np.float32], out: list[AudioFrame]) -> None:
        if samples.shape[0]:
            self._out = np.concatenate([self._out, samples]) if self._out.size else samples
        n = self.out_fmt.frame_samples
        while self._out.shape[0] >= n and self._pending:
            seq, t_ns = self._pending.popleft()
            data = self._out[:n].copy()
            self._out = self._out[n:]
            out.append(AudioFrame(data, self.out_fmt, self.out_map, t_ns, seq, self._clock_id))
