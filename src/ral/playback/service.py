"""PlaybackQueue: the playback service (DESIGN section 5, "Playback").

Everything RAL plays goes through here, then through a :class:`SinkWriter` into the sink,
whose render hook feeds the reference buffer (CLAUDE.md, reference invariant).

Model (v0.1): one item plays at a time. Items wait in a priority queue (higher number
first, first come first served within a priority). A new item with a higher priority
than the playing one preempts it if the playing one is interruptible: the old item fades
out and is dropped, not resumed. Suggested priorities: alerts 90, speech 50, earcons 30,
music 10. TODO(M2): mix concurrent items (music under speech) instead of preempting.

Each item's async iterator is read by its own pump task into a short queue, so a slow
producer (streaming TTS) never stalls the output loop: if no frame is ready when the
device needs one, nothing is written and the sink renders silence, which also reaches the
reference buffer, so the echo canceller's timeline stays continuous.

Fades. Stopping an item fades it out over ``fade_ms`` (default 40 ms) with a raised-cosine
ramp, which avoids the click of cutting a waveform mid-cycle. Ducking ramps the output
gain to ``gain_db`` over 30 ms, holds it for ``duration_ms`` and ramps back.

Timing. ``PlaybackStarted`` and ``PlaybackEnded`` carry the estimated DAC time of the
item's first and last sample and are published when that time arrives (immediately in
offline mode). :meth:`speaking_at` answers "was the robot speaking at time t?" from the
same DAC-time intervals; echo control and barge-in use it.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
import logging
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ral.core.bus import EventBus
from ral.core.clock import now_ns
from ral.core.errors import FormatError
from ral.core.events import Event, PlaybackEnded, PlaybackStarted
from ral.core.protocols import PlaybackHandle
from ral.core.types import AudioFormat, AudioFrame, ChannelMap
from ral.playback.writer import SinkWriter

__all__ = ["PlaybackQueue", "audio_frames"]

log = logging.getLogger(__name__)

_PUMP_FRAMES = 50  # 500 ms of audio buffered per item ahead of the output loop
_DUCK_RAMP_NS = 30_000_000
_MAX_INTERVALS = 256


async def audio_frames(
    samples: npt.NDArray[np.float32], fmt: AudioFormat | None = None
) -> AsyncIterator[AudioFrame]:
    """Yield ``samples`` (1-D or ``(n, 1)``, float32) as engine frames, zero-padding the end.

    A convenience for tests, examples and bridges that already hold a whole clip.
    """
    fmt = fmt or AudioFormat(16_000, 1, "float32")
    x = np.asarray(samples, dtype=np.float32).reshape(-1, 1)
    n = fmt.frame_samples
    cmap = ChannelMap.uniform(1)
    for seq, start in enumerate(range(0, x.shape[0], n)):
        block = np.zeros((n, 1), np.float32)
        chunk = x[start : start + n]
        block[: chunk.shape[0]] = chunk
        yield AudioFrame(block, fmt, cmap, 0, seq, "playback")


@dataclass(order=True)
class _Item:
    sort_key: tuple[int, int]
    handle: PlaybackHandle = field(compare=False)
    priority: int = field(compare=False)
    interruptible: bool = field(compare=False)
    audio: AsyncIterator[AudioFrame] = field(compare=False)
    frames: asyncio.Queue[AudioFrame] = field(compare=False)
    done: asyncio.Future[bool] = field(compare=False)
    pump: asyncio.Task[None] | None = field(default=None, compare=False)
    exhausted: bool = field(default=False, compare=False)
    t_request_ns: int | None = field(default=None, compare=False)
    t_start_dac: int | None = field(default=None, compare=False)
    fade_total: int = field(default=0, compare=False)  # samples; 0 = not fading
    fade_left: int = field(default=0, compare=False)
    interrupted: bool = field(default=False, compare=False)


class PlaybackQueue:
    """Implements :class:`~ral.core.protocols.PlaybackService` on one :class:`SinkWriter`.

    Args:
        writer: Where frames go; its sink must be started before :meth:`start`.
        bus: Where ``PlaybackStarted`` and ``PlaybackEnded`` are published, if given.
        session_id: Session id stamped on events.
    """

    def __init__(
        self, writer: SinkWriter, bus: EventBus | None = None, session_id: str = ""
    ) -> None:
        self.writer = writer
        self.fmt = writer.in_fmt
        self.bus = bus
        self.session_id = session_id
        self._ids = itertools.count(1)
        self._order = itertools.count()
        self._queue: list[_Item] = []
        self._items: dict[int, _Item] = {}
        self._active: _Item | None = None
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._gain = 1.0
        self._duck_gain = 1.0
        self._duck_until: int | None = None
        self._intervals: deque[tuple[int, int | None]] = deque(maxlen=_MAX_INTERVALS)
        self._timers: set[asyncio.TimerHandle] = set()
        self._results: dict[int, bool] = {}  # handle id -> interrupted, for wait()

    # Lifecycle

    async def start(self) -> None:
        """Start the output loop."""
        self.writer.start()
        self._task = asyncio.create_task(self._run(), name="ral-playback")

    async def close(self) -> None:
        """Stop everything at once (no fade) and end the output loop."""
        for item in list(self._items.values()):
            self._finish(item, interrupted=True, t_end=self.writer.cursor_ns)
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()

    # PlaybackService

    def play(
        self,
        audio: AsyncIterator[AudioFrame],
        *,
        priority: int = 50,
        interruptible: bool = True,
        tag: str = "",
    ) -> PlaybackHandle:
        """Queue ``audio`` (mono float32 frames at the engine rate)."""
        handle = PlaybackHandle(next(self._ids), tag)
        loop = asyncio.get_running_loop()
        item = _Item(
            sort_key=(-priority, next(self._order)),
            handle=handle,
            priority=priority,
            interruptible=interruptible,
            audio=audio,
            frames=asyncio.Queue(_PUMP_FRAMES),
            done=loop.create_future(),
        )
        item.pump = asyncio.create_task(self._pump(item), name=f"ral-playback-pump-{handle.id}")
        self._items[handle.id] = item
        heapq.heappush(self._queue, item)
        self._wake.set()
        return handle

    async def stop(self, handle: PlaybackHandle, fade_ms: int = 40) -> None:
        """Stop ``handle``: fade it out if playing, drop it if queued; returns once ended."""
        item = self._items.get(handle.id)
        if item is None:
            return
        if item is self._active:
            self._begin_fade(item, fade_ms)
            self._wake.set()
        else:
            self._queue.remove(item)
            heapq.heapify(self._queue)
            self._finish(item, interrupted=True, t_end=None)
        await asyncio.shield(item.done)

    def duck(self, gain_db: float, duration_ms: int) -> None:
        """Lower the output by ``gain_db`` (negative) for ``duration_ms``; 0 ms cancels."""
        if duration_ms <= 0:
            self._duck_until = None
            self._duck_gain = 1.0
            return
        self._duck_gain = float(10 ** (min(gain_db, 0.0) / 20))
        self._duck_until = self.writer.cursor_ns + duration_ms * 1_000_000

    # Extras used by the engine and bridges

    async def wait(self, handle: PlaybackHandle) -> bool:
        """Wait until ``handle`` ends; True if it was interrupted."""
        item = self._items.get(handle.id)
        if item is None:
            return self._results.get(handle.id, False)
        return await asyncio.shield(item.done)

    def interrupt(self, fade_ms: int = 40) -> PlaybackHandle | None:
        """Barge-in: fade out the playing item and drop queued items, if interruptible.

        Returns:
            The handle that was playing and is now fading, or None.
        """
        for item in [i for i in self._queue if i.interruptible]:
            self._queue.remove(item)
            self._finish(item, interrupted=True, t_end=None)
        heapq.heapify(self._queue)
        active = self._active
        if active is not None and active.interruptible and not active.fade_total:
            self._begin_fade(active, fade_ms)
            self._wake.set()
            return active.handle
        return None

    @property
    def current(self) -> PlaybackHandle | None:
        """The item playing now, if any."""
        return self._active.handle if self._active else None

    @property
    def busy(self) -> bool:
        """True while an item plays or waits."""
        return self._active is not None or bool(self._queue)

    def speaking_at(self, t_ns: int) -> bool:
        """Whether RAL's audio was at the DAC at time ``t_ns``."""
        for start, end in reversed(self._intervals):
            if start <= t_ns and (end is None or t_ns < end):
                return True
            if end is not None and end <= t_ns:
                return False
        return False

    # Output loop

    async def _run(self) -> None:
        while True:
            await self.writer.wait_turn()
            frame = self._next_frame()
            if frame is not None:
                self._write(frame)
                continue
            # Nothing ready: let the sink render silence until something arrives.
            if self._active is None and not self._queue:
                self.writer.idle()
            self._wake.clear()
            if self.writer.realtime:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), 0.005)
            else:
                await self._wake.wait()

    def _next_frame(self) -> npt.NDArray[np.float32] | None:
        while True:
            self._maybe_preempt()
            if self._active is None:
                if not self._queue:
                    return None
                self._active = heapq.heappop(self._queue)
            item = self._active
            try:
                frame = item.frames.get_nowait()
            except asyncio.QueueEmpty:
                if not item.exhausted:
                    return None  # the producer is slow; the sink renders silence meanwhile
                self._finish(item, interrupted=item.interrupted, t_end=self.writer.cursor_ns)
                continue
            data: npt.NDArray[np.float32] = frame.data.astype(np.float32, copy=True)
            if item.fade_total:
                data = self._apply_fade(item, data)
            return data

    def _write(self, data: npt.NDArray[np.float32]) -> None:
        item = self._active
        assert item is not None
        t_dac = self.writer.write(self._apply_duck(data))
        if item.t_start_dac is None:
            item.t_start_dac = t_dac
            self._intervals.append((t_dac, None))
            self._publish_at(
                PlaybackStarted(
                    t_dac,
                    self.session_id,
                    handle_id=item.handle.id,
                    tag=item.handle.tag,
                    t_request_ns=item.t_request_ns or t_dac,
                )
            )
        if item.fade_total and item.fade_left == 0:
            # The fade ended inside this frame: the item ends with it.
            self._finish(item, interrupted=True, t_end=t_dac + self.fmt.frame_ns)

    def _maybe_preempt(self) -> None:
        active = self._active
        if active is None or not self._queue or active.fade_total:
            return
        if self._queue[0].priority > active.priority and active.interruptible:
            self._begin_fade(active, 40)

    # Item helpers

    async def _pump(self, item: _Item) -> None:
        # An item stopped before its pump ran never reads its producer.
        try:
            async for frame in item.audio:
                if frame.fmt != self.fmt:
                    raise FormatError(
                        f"Playback item {item.handle} has frames in {frame.fmt}; "
                        f"the playback service needs {self.fmt}."
                    )
                if item.t_request_ns is None:
                    item.t_request_ns = now_ns()
                await item.frames.put(frame)
                self._wake.set()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Playback item %s failed; ending it", item.handle)
        finally:
            item.exhausted = True
            # Close the producer from its own task, so it can release resources.
            aclose = getattr(item.audio, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()
            self._wake.set()

    def _begin_fade(self, item: _Item, fade_ms: int) -> None:
        item.interrupted = True
        samples = max(1, fade_ms * self.fmt.sample_rate // 1000)
        item.fade_total = samples
        item.fade_left = samples
        if item.t_start_dac is None:
            # Nothing has played yet: end at once instead of fading silence.
            self._active = None if self._active is item else self._active
            self._finish(item, interrupted=True, t_end=None)

    def _apply_fade(self, item: _Item, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        n = data.shape[0]
        done = item.fade_total - item.fade_left
        pos = np.arange(done, done + n, dtype=np.float64)
        # Raised cosine from 1 to 0 over fade_total samples, then 0.
        ramp = 0.5 * (1.0 + np.cos(np.pi * np.minimum(pos / item.fade_total, 1.0)))
        item.fade_left = max(0, item.fade_left - n)
        return (data * ramp[:, None]).astype(np.float32)

    def _apply_duck(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        t = self.writer.cursor_ns
        target = self._duck_gain if self._duck_until is not None and t < self._duck_until else 1.0
        if self._duck_until is not None and t >= self._duck_until:
            self._duck_until = None
            self._duck_gain = 1.0
        if target == self._gain == 1.0:
            return data
        # Move linearly toward the target, at most one ramp's worth per frame.
        step = (self.fmt.frame_ns / _DUCK_RAMP_NS) * 1.0
        new = (
            min(target, self._gain + step)
            if target > self._gain
            else max(target, self._gain - step)
        )
        ramp = np.linspace(self._gain, new, data.shape[0], endpoint=False, dtype=np.float32)
        self._gain = new
        return data * ramp[:, None]

    def _finish(self, item: _Item, *, interrupted: bool, t_end: int | None) -> None:
        if item.done.done():
            return
        if self._active is item:
            self._active = None
        self._items.pop(item.handle.id, None)
        if item.pump is not None and not item.pump.done():
            item.pump.cancel()
        if item.t_start_dac is not None and t_end is not None:
            for i in range(len(self._intervals) - 1, -1, -1):
                if self._intervals[i] == (item.t_start_dac, None):
                    self._intervals[i] = (item.t_start_dac, t_end)
                    break
            self._publish_at(
                PlaybackEnded(
                    t_end,
                    self.session_id,
                    handle_id=item.handle.id,
                    tag=item.handle.tag,
                    interrupted=interrupted,
                )
            )
        item.done.set_result(interrupted)
        self._results[item.handle.id] = interrupted
        if len(self._results) > _MAX_INTERVALS:
            del self._results[next(iter(self._results))]
        self._wake.set()

    def _publish_at(self, event: Event) -> None:
        if self.bus is None:
            return
        delay = (event.t_ns - now_ns()) / 1e9
        if not self.writer.realtime or delay <= 0:
            self.bus.publish(event)
            return
        loop = asyncio.get_running_loop()
        timer: asyncio.TimerHandle

        def fire() -> None:
            self._timers.discard(timer)
            assert self.bus is not None
            self.bus.publish(event)

        timer = loop.call_later(delay, fire)
        self._timers.add(timer)
