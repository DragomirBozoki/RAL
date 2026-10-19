from __future__ import annotations

import asyncio
import wave
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ral.core import AudioFormat, AudioFrame, ChannelMap, EventBus
from ral.core import events as ev
from ral.devices import FileBackend
from ral.playback.reference import ReferenceBuffer
from ral.playback.service import PlaybackQueue
from ral.playback.writer import SinkWriter

ENGINE = AudioFormat(16000, 1, "float32")


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
        ch = w.getnchannels()
    return np.frombuffer(raw, "<i2").reshape(-1, ch).astype(np.float32) / 32768


class Rig:
    """A playback queue on a FileSink, with its reference buffer and an event log."""

    def __init__(self, path: Path, sink_fmt: AudioFormat, realtime: bool) -> None:
        backend = FileBackend(sinks={"spk": path}, realtime=realtime)
        (dev,) = backend.discover()
        self.path = path
        self.sink = backend.open_sink(dev, sink_fmt)
        self.ref = ReferenceBuffer(sink_fmt)
        self.sink.on_rendered(self.ref.push)
        self.bus = EventBus()
        self.events = self.bus.subscribe()
        self.writer = SinkWriter(self.sink, ENGINE, realtime=realtime)
        self.queue = PlaybackQueue(self.writer, self.bus, "test")

    async def __aenter__(self) -> Rig:
        await self.sink.start()
        await self.queue.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.queue.close()
        await self.sink.stop()

    def drain_events(self) -> list[ev.Event]:
        out = []
        while (e := self.events.get_nowait()) is not None:
            out.append(e)
        return out

    def output(self) -> np.ndarray:
        return read_wav(self.path)


RigFactory = Callable[..., Rig]


@pytest.fixture
def rig_factory(tmp_path: Path) -> RigFactory:
    def make(sink_fmt: AudioFormat = ENGINE, realtime: bool = False, name: str = "out") -> Rig:
        return Rig(tmp_path / f"{name}.wav", sink_fmt, realtime)

    return make


async def gated(
    values: list[float], gate: asyncio.Event | None = None, after: int = 0
) -> AsyncIterator[AudioFrame]:
    """Frames of constant value; waits on `gate` after `after` frames (a slow producer)."""
    cmap = ChannelMap.uniform(1)
    for i, v in enumerate(values):
        if gate is not None and i == after:
            await gate.wait()
        yield AudioFrame(np.full((160, 1), v, np.float32), ENGINE, cmap, 0, i, "src")


async def settle(n: int = 20) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


def tag_of(e: Any) -> str:
    return str(getattr(e, "tag", ""))
