import asyncio
import wave
from pathlib import Path

import numpy as np
import pytest

from ral.core import events as ev
from ral.engine.engine import Engine
from ral.playback.clip import load_wav
from ral.playback.service import audio_frames
from ral.profiles import parse_profile

DATA = Path(__file__).resolve().parents[1] / "data"


def file_profile(tmp_path: Path, *, realtime: bool, mode: str = "off", vad: str = "energy") -> str:
    mic = (DATA / "speech_16k.wav").as_posix()
    out = (tmp_path / "out.wav").as_posix()
    rt = str(realtime).lower()
    return f"""
ral: 1
capture: {{backend: file, path: '{mic}', realtime: {rt}}}
playback: {{backend: file, path: '{out}', realtime: {rt}}}
echo: {{mode: {mode}}}
pipeline: [aec, vad]
vad: {{engine: {vad}}}
"""


def wav_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as w:
        return w.getnframes()


async def test_offline_file_run_processes_every_frame_and_finds_speech(tmp_path: Path) -> None:
    engine = Engine(parse_profile(file_profile(tmp_path, realtime=False)), session_id="t1")
    events = engine.bus.subscribe(ev.SpeechStarted, ev.SpeechEnded)
    frames = engine.subscribe(maxsize=2000)  # before start: offline runs flat out
    async with engine:
        await asyncio.wait_for(engine.capture_done.wait(), 30)
        await asyncio.sleep(0)
        clean = []
        while not frames._queue.empty():
            item = frames._queue.get_nowait()
            if item is not None:
                clean.append(item.clean)
    assert engine.worker_error is None
    assert (engine.input_overruns, frames.dropped) == (0, 0)
    assert [f.seq for f in clean] == list(range(1000))  # 10 s of 10 ms frames, none lost
    seen = []
    while (e := events.get_nowait()) is not None:
        seen.append(e)
    assert any(isinstance(e, ev.SpeechStarted) for e in seen)
    assert all(e.session_id == "t1" for e in seen)


async def test_playback_goes_through_the_sink_and_the_reference(tmp_path: Path) -> None:
    pytest.importorskip("livekit.rtc")
    engine = Engine(parse_profile(file_profile(tmp_path, realtime=False, mode="software_ref")))
    started = engine.bus.subscribe(ev.PlaybackStarted, ev.PlaybackEnded)
    clip = load_wav(DATA / "speech_16k.wav")[:16000]
    async with engine:
        assert engine.playback is not None
        handle = engine.playback.play(audio_frames(clip), tag="hello")
        await engine.playback.wait(handle)
        await asyncio.wait_for(engine.capture_done.wait(), 30)
    assert wav_frames(tmp_path / "out.wav") >= 16000
    assert isinstance(await started.get(), ev.PlaybackStarted)
    assert engine.graph is not None
    from ral.dsp.delay import DelayTracker
    from ral.dsp.echo import ApmProcessor

    kinds = [type(s) for s in engine.graph.stages]
    assert kinds[:2] == [DelayTracker, ApmProcessor]  # delay tracking is on by default
    apm = engine.graph.stages[1]
    assert isinstance(apm, ApmProcessor)
    assert apm.reference_frames > 0
    assert apm.lead is engine.lead  # one lead shared by graph, tracker and APM


async def test_realtime_run_paces_and_stops_cleanly(tmp_path: Path) -> None:
    engine = Engine(parse_profile(file_profile(tmp_path, realtime=True)))
    loop = asyncio.get_running_loop()
    async with engine:
        sub = engine.subscribe()
        t0 = loop.time()
        got = 0
        async for _ in sub.clean():
            got += 1
            if got == 50:
                break
        elapsed = loop.time() - t0
    assert 0.35 < elapsed < 1.5  # 50 frames of 10 ms, paced by the clock
    assert engine.worker_error is None


async def test_barge_in_interrupts_playback(tmp_path: Path) -> None:
    engine = Engine(parse_profile(file_profile(tmp_path, realtime=False)))
    async with engine:
        assert engine.playback is not None
        gate = asyncio.Event()

        async def endless():  # type: ignore[no-untyped-def]
            async for f in audio_frames(np.full(16000, 0.1, np.float32)):
                yield f
            await gate.wait()

        handle = engine.playback.play(endless(), tag="answer")
        await asyncio.sleep(0.05)
        engine.bus.publish(ev.BargeIn(0, engine.session_id, "answer"))
        assert await asyncio.wait_for(engine.playback.wait(handle), 5) is True


def test_bad_channel_list_is_a_config_error(tmp_path: Path) -> None:
    from ral.core import ConfigError

    text = file_profile(tmp_path, realtime=False).replace(
        "realtime: false}", "realtime: false, channels: [mic, mic]}", 1
    )
    engine = Engine(parse_profile(text))

    async def go() -> None:
        await engine.start()

    with pytest.raises(ConfigError, match="one role per channel"):
        asyncio.run(go())


def test_hardware_mode_needs_a_processed_channel(tmp_path: Path) -> None:
    from ral.core import ConfigError

    text = file_profile(tmp_path, realtime=False).replace(
        "echo: {mode: off}", "echo: {mode: hardware}"
    )
    engine = Engine(parse_profile(text))

    async def go() -> None:
        await engine.start()

    with pytest.raises(ConfigError, match="processed"):
        asyncio.run(go())
