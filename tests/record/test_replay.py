import asyncio
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcap")

from ral.bench.report import write_scene_recording
from ral.bench.synthetic import echo_scene
from ral.core import AudioFormat, AudioFrame, ChannelMap, FormatError
from ral.core import events as ev
from ral.dsp.echo import EchoOff
from ral.dsp.vad import EnergyVad
from ral.record.mcap import McapRecorder
from ral.record.replay import RecordedReference, load_session, replay

UTTERANCES = [(0.5, 3.5), (5.0, 8.0)]
T0 = 1_000_000_000_000


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory) -> Path:
    scene = echo_scene(seconds=9, delay_ms=20, far_spans=UTTERANCES, seed=5)
    return write_scene_recording(
        scene, tmp_path_factory.mktemp("rec") / "s.mcap", playback_s=UTTERANCES
    )


def test_session_inputs_round_trip(session: Path) -> None:
    inputs = load_session(session)
    assert len(inputs.mic) == 900
    assert len(inputs.reference) == 900
    assert inputs.mic[0].fmt == AudioFormat(16000, 1, "float32")
    assert inputs.playback == [
        (T0 + 500_000_000, T0 + 3_500_000_000),
        (T0 + 5_000_000_000, T0 + 8_000_000_000),
    ]
    assert [f.seq for f in inputs.mic[:3]] == [0, 1, 2]


def test_recorded_reference_hands_out_frames_by_time() -> None:
    fmt = AudioFormat(16000, 1, "float32")
    frames = [
        AudioFrame(np.zeros((160, 1), np.float32), fmt, ChannelMap.uniform(1), t, i, "dac")
        for i, t in enumerate([30, 10, 20])
    ]
    ref = RecordedReference(frames)
    assert [f.t_ns for f in ref.frames_until(25)] == [10, 20]
    assert ref.frames_until(25) == []
    assert [f.t_ns for f in ref.frames_until(100)] == [30]


def test_replay_reproduces_the_echo_and_its_removal(session: Path) -> None:
    pytest.importorskip("livekit.rtc")
    from ral.bench.report import bench_rows
    from ral.dsp.echo import ApmProcessor

    inputs = load_session(session)
    off = replay(inputs, [EchoOff(), EnergyVad()], name="off")
    aec = replay(inputs, [ApmProcessor(), EnergyVad()], name="aec")
    rows = {r.name: r for r in bench_rows(inputs, [off, aec])}
    assert rows["off"].starts_per_min > 10
    assert rows["aec"].starts_per_min == 0
    assert rows["off"].erle_db is not None
    assert abs(rows["off"].erle_db) < 1
    assert rows["aec"].erle_db is not None
    assert rows["aec"].erle_db > 30
    assert len(off.clean) == 900 * 160


async def test_an_engine_recording_replays(tmp_path: Path) -> None:
    from ral.engine.engine import Engine
    from ral.profiles import parse_profile

    data = Path(__file__).resolve().parents[1] / "data"
    mic = (data / "speech_16k.wav").as_posix()
    out = (tmp_path / "o.wav").as_posix()
    rec = (tmp_path / "rec").as_posix()
    prof = parse_profile(
        "ral: 1\n"
        f"capture: {{backend: file, path: '{mic}', realtime: false, channels: [mic]}}\n"
        f"playback: {{backend: file, path: '{out}', realtime: false}}\n"
        "echo: {mode: off}\npipeline: [aec, vad]\nvad: {engine: energy}\n"
        f"recording: {{enabled: true, path: '{rec}'}}\n"
    )
    async with Engine(prof) as engine:
        await asyncio.wait_for(engine.capture_done.wait(), 30)
    (path,) = (tmp_path / "rec").glob("*.mcap")
    inputs = load_session(path)
    assert len(inputs.mic) == 1000
    result = replay(inputs, [EchoOff(), EnergyVad()])
    assert any(isinstance(e, ev.SpeechStarted) for e in result.events)


def test_clean_only_recordings_cannot_be_replayed(tmp_path: Path) -> None:
    rec = McapRecorder(tmp_path / "c.mcap", raw=False)
    rec.start()
    f = AudioFrame(
        np.zeros((160, 1), np.float32),
        AudioFormat(16000, 1, "float32"),
        ChannelMap.uniform(1),
        0,
        0,
        "m",
    )
    rec.write_result(f, f, [])
    rec.stop()
    with pytest.raises(FormatError, match="raw"):
        load_session(tmp_path / "c.mcap")
