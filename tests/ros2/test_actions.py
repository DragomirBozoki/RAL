"""ROS 2 actions and services against the fake rclpy (tests/fakes/rclpy_fake.py)."""

from __future__ import annotations

import array
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pytest

from fakes import rclpy_fake as rf
from ral.core import AudioFormat, AudioFrame, ChannelMap, MissingDependencyError
from ral.core import events as ev

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def ros(monkeypatch: pytest.MonkeyPatch) -> rf.Graph:
    return rf.install(monkeypatch)


@asynccontextmanager
async def running(tmp_path: Path, **bridge_kw: object) -> AsyncIterator[object]:
    from ral.bridges.ros2 import Ros2Bridge
    from ral.engine.engine import Engine
    from ral.profiles import parse_profile

    prof = parse_profile(
        "ral: 1\n"
        f"capture: {{backend: file, path: '{(DATA / 'speech_16k.wav').as_posix()}', "
        "realtime: true}\n"
        f"playback: {{backend: file, path: '{(tmp_path / 'o.wav').as_posix()}', "
        "realtime: true}\n"
        "echo: {mode: off}\npipeline: [aec]\n"
    )
    engine = Engine(prof)
    async with engine:
        bridge = Ros2Bridge("/ral", **bridge_kw)  # type: ignore[arg-type]
        await bridge.attach(engine)
        try:
            yield engine
        finally:
            await bridge.detach()


def tone(seconds: float, rate: int = 16000) -> array.array[int]:
    x = (np.sin(2 * np.pi * 440 * np.arange(int(seconds * rate)) / rate) * 8000).astype("<i2")
    return array.array("B", x.tobytes())


async def finished(handle: rf.GoalHandle) -> rf._ActionResult:
    assert await asyncio.to_thread(handle.done.wait, 10)
    result: rf._ActionResult = handle.result
    return result


async def test_play_audio_runs_to_the_end_with_feedback(ros: rf.Graph, tmp_path: Path) -> None:
    async with running(tmp_path) as engine:
        started = engine.bus.subscribe(ev.PlaybackStarted)  # type: ignore[attr-defined]
        handle = rf.send_goal("/ral/play_audio", rf.PlayAudio.Goal(tone(0.5, 8000), 8000, "beep"))
        assert handle is not None
        result = await finished(handle)
        assert (await asyncio.wait_for(started.get(), 1)).tag == "beep"
    assert handle.status == "succeeded"
    assert result.reason == "finished"
    assert not result.interrupted
    assert result.played_s == pytest.approx(0.5, abs=0.02)  # resampled from 8 kHz
    assert handle.feedback
    assert handle.feedback[-1].played_s <= result.played_s


async def test_cancel_fades_the_goal_out(ros: rf.Graph, tmp_path: Path) -> None:
    async with running(tmp_path) as engine:
        started = engine.bus.subscribe(ev.PlaybackStarted)  # type: ignore[attr-defined]
        handle = rf.send_goal("/ral/play_audio", rf.PlayAudio.Goal(tone(3.0)))
        assert handle is not None
        await asyncio.wait_for(started.get(), 2)
        assert handle.cancel()
        result = await finished(handle)
    assert handle.status == "canceled"
    assert result.reason == "cancelled"
    assert result.interrupted
    assert result.played_s < 2.0


async def test_bad_goals_are_rejected(ros: rf.Graph, tmp_path: Path) -> None:
    async with running(tmp_path):
        assert rf.send_goal("/ral/play_audio", rf.PlayAudio.Goal(array.array("B", b"\x00"))) is None
        assert rf.send_goal("/ral/play_audio", rf.PlayAudio.Goal(array.array("B"))) is None
        assert rf.send_goal("/ral/say", rf.Say.Goal("hello")) is None  # no TTS adapter


async def test_say_speaks_with_the_given_tts(ros: rf.Graph, tmp_path: Path) -> None:
    said: list[str] = []
    fmt = AudioFormat(16000, 1, "float32")

    async def tts(text: str) -> AsyncIterator[AudioFrame]:
        said.append(text)
        for k in range(30):
            data = np.full((160, 1), 0.1, np.float32)
            yield AudioFrame(data, fmt, ChannelMap.uniform(1), 0, k, "tts")

    async with running(tmp_path, tts=tts):
        assert rf.send_goal("/ral/say", rf.Say.Goal("   ")) is None
        handle = rf.send_goal("/ral/say", rf.Say.Goal("hello robot", tag="greet"))
        assert handle is not None
        result = await finished(handle)
    assert said == ["hello robot"]
    assert result.reason == "finished"
    assert result.played_s == pytest.approx(0.3)


async def test_stop_and_duck_services(ros: rf.Graph, tmp_path: Path) -> None:
    async with running(tmp_path) as engine:
        started = engine.bus.subscribe(ev.PlaybackStarted)  # type: ignore[attr-defined]
        handle = rf.send_goal("/ral/play_audio", rf.PlayAudio.Goal(tone(3.0)))
        assert handle is not None
        await asyncio.wait_for(started.get(), 2)
        duck = await asyncio.to_thread(rf.call_service, "/ral/duck", rf.Duck.Request(-20, 500))
        assert duck.ok
        stop = await asyncio.to_thread(rf.call_service, "/ral/stop", rf.Stop.Request(20))
        assert stop.stopped
        result = await finished(handle)
        idle = await asyncio.to_thread(rf.call_service, "/ral/stop", rf.Stop.Request())
    assert result.reason == "interrupted"
    assert handle.status == "succeeded"
    assert not idle.stopped
    assert not ros.actions  # destroyed on detach
    assert not ros.services


async def test_without_the_interface_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph = rf.install(monkeypatch, interfaces=False)
    async with running(tmp_path):
        assert not graph.actions  # auto: topics only
    with pytest.raises(MissingDependencyError, match="colcon"):
        async with running(tmp_path, actions=True):
            pass
