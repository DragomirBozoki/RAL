"""RalTransport against a real RAL engine and WebSocket bridge, inside one process."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from ral_pipecat.transport import RalTransport, RalTransportParams

from ral.bridges import start_bridges, stop_bridges
from ral.core import events as ev
from ral.engine.engine import Engine
from ral.profiles import parse_profile

DATA = Path(__file__).resolve().parents[3] / "tests" / "data"


class FakeBot(FrameProcessor):
    """Listens for 200 ms, then says a 310 ms tone at 24 kHz the way a TTS service does.

    310 ms is not a whole number of Pipecat's 40 ms output chunks, so the last chunk is
    partial and is flushed by TTSStoppedFrame, as with a real TTS.
    """

    def __init__(self) -> None:
        super().__init__()
        self.heard = 0
        self.spoke = False
        self.interruptions = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self.heard += 1
            if self.heard == 20 and not self.spoke:
                self.spoke = True
                t = np.arange(7440) / 24000
                pcm = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2").tobytes()
                await self.push_frame(TTSStartedFrame())
                for i in range(0, len(pcm), 960):  # 20 ms frames
                    await self.push_frame(TTSAudioRawFrame(pcm[i : i + 960], 24000, 1))
                await self.push_frame(TTSStoppedFrame())
            return
        if isinstance(frame, InterruptionFrame):
            self.interruptions += 1
        await self.push_frame(frame, direction)


@pytest.fixture
async def ral(tmp_path: Path) -> AsyncIterator[tuple[Engine, int]]:
    mic = (DATA / "speech_16k.wav").as_posix()
    out = (tmp_path / "out.wav").as_posix()
    prof = parse_profile(
        "ral: 1\n"
        f"capture: {{backend: file, path: '{mic}', realtime: true}}\n"
        f"playback: {{backend: file, path: '{out}', realtime: true}}\n"
        "echo: {mode: off}\npipeline: [aec, vad]\nvad: {engine: energy}\n"
        "barge_in: {policy: off}\nbridges: {websocket: {port: 0}}\n"
    )
    engine = Engine(prof)
    async with engine:
        bridges = await start_bridges(engine, prof.bridges)
        try:
            yield engine, bridges[0].port  # type: ignore[attr-defined]
        finally:
            await stop_bridges(bridges)


async def run_pipeline(port: int, bot: FakeBot, until: Any, timeout: float = 6.0) -> RalTransport:
    transport = RalTransport(RalTransportParams(url=f"ws://127.0.0.1:{port}", tag="bot"))
    task = PipelineTask(Pipeline([transport.input(), bot, transport.output()]))
    runner = PipelineRunner(handle_sigint=False)
    run = asyncio.create_task(runner.run(task))
    try:
        await asyncio.wait_for(until(), timeout)
    finally:
        await task.cancel()
        await asyncio.wait_for(run, 5)
    return transport


async def test_bot_hears_the_robot_and_speaks_through_ral(ral: tuple[Engine, int]) -> None:
    engine, port = ral
    played = engine.bus.subscribe(ev.PlaybackStarted, ev.PlaybackEnded)
    bot = FakeBot()
    seen: list[dict[str, Any]] = []

    async def until() -> None:
        started = await played.get()
        ended = await played.get()
        seen.extend([{"t": started.t_ns, "tag": started.tag}, {"t": ended.t_ns}])

    await run_pipeline(port, bot, until)
    assert bot.heard >= 20  # clean 10 ms frames from RAL
    assert seen[0]["tag"] == "bot"
    # 310 ms at 24 kHz plus the silence Pipecat pads the last chunk with (up to 40 ms),
    # resampled by RAL and played without a gap before the last frame.
    assert 310_000_000 <= seen[1]["t"] - seen[0]["t"] <= 360_000_000


async def test_ral_barge_in_interrupts_the_bot(ral: tuple[Engine, int]) -> None:
    engine, port = ral
    bot = FakeBot()

    async def until() -> None:
        while bot.heard < 5:
            await asyncio.sleep(0.01)
        engine.bus.publish(ev.BargeIn(0, engine.session_id, "bot"))
        while bot.interruptions == 0:
            await asyncio.sleep(0.01)

    await run_pipeline(port, bot, until)
    assert bot.interruptions >= 1


async def test_events_reach_the_handler(ral: tuple[Engine, int]) -> None:
    engine, port = ral
    transport = RalTransport(RalTransportParams(url=f"ws://127.0.0.1:{port}"))
    got: list[dict[str, Any]] = []

    @transport.event_handler("on_ral_event")
    async def on_event(_: Any, event: dict[str, Any]) -> None:
        got.append(event)

    task = PipelineTask(Pipeline([transport.input(), FakeBot(), transport.output()]))
    run = asyncio.create_task(PipelineRunner(handle_sigint=False).run(task))
    try:
        await asyncio.sleep(0.3)
        engine.bus.publish(ev.SpeechStarted(1, engine.session_id))
        for _ in range(100):
            if any(e["type"] == "SpeechStarted" for e in got):
                break
            await asyncio.sleep(0.02)
    finally:
        await task.cancel()
        await asyncio.wait_for(run, 5)
    assert any(e["type"] == "SpeechStarted" for e in got)
