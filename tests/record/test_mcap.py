import asyncio
import json
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcap")

from ral.core import AudioFormat, AudioFrame, ChannelMap
from ral.core import events as ev
from ral.engine.engine import Engine
from ral.profiles import parse_profile
from ral.record.mcap import McapRecorder, read_recording

DATA = Path(__file__).resolve().parents[1] / "data"


def frame(values: np.ndarray, roles: list[str], t: int, seq: int) -> AudioFrame:
    fmt = AudioFormat(16000, len(roles), "float32")
    return AudioFrame(values.astype(np.float32), fmt, ChannelMap.from_names(roles), t, seq, "c")


def test_round_trip_of_audio_events_and_metadata(tmp_path: Path) -> None:
    rec = McapRecorder(tmp_path / "s.mcap", session_id="abc", profile={"echo": {"mode": "off"}})
    rec.start()
    rng = np.random.default_rng(0)
    raw_blocks = []
    for k in range(20):
        pcm = rng.integers(-32768, 32767, (160, 2), dtype=np.int16)
        raw_blocks.append(pcm)
        raw = frame(pcm / 32768.0, ["mic", "reference"], k * 10_000_000, k)
        clean = frame(pcm[:, :1] / 32768.0, ["processed"], k * 10_000_000, k)
        rec.write_result(raw, clean, [])
    rec.write_event(ev.SpeechStarted(5, "abc"))
    rec.stop()
    back = read_recording(tmp_path / "s.mcap")
    np.testing.assert_array_equal(back.signal("/ral/mic"), np.concatenate(raw_blocks))
    assert back.roles["/ral/mic"] == "mic,reference"
    assert back.signal("/ral/clean").shape == (3200, 1)
    assert back.events == [ev.SpeechStarted(5, "abc")]
    assert back.metadata["ral.session"]["session_id"] == "abc"
    assert json.loads(back.metadata["ral.profile"]["yaml_json"]) == {"echo": {"mode": "off"}}
    assert rec.dropped == 0


def test_clean_only_when_raw_is_off(tmp_path: Path) -> None:
    rec = McapRecorder(tmp_path / "s.mcap", raw=False)
    rec.start()
    f = frame(np.zeros((160, 1)), ["mic"], 0, 0)
    rec.write_result(f, f, [f])
    rec.stop()
    back = read_recording(tmp_path / "s.mcap")
    assert set(back.audio) == {"/ral/clean"}


def profile(tmp_path: Path, extra: str = "") -> str:
    mic = (DATA / "speech_16k.wav").as_posix()
    out = (tmp_path / "out.wav").as_posix()
    return (
        "ral: 1\n"
        f"capture: {{backend: file, path: '{mic}', realtime: false}}\n"
        f"playback: {{backend: file, path: '{out}', realtime: false}}\n"
        "echo: {mode: off}\npipeline: [aec, vad]\nvad: {engine: energy}\n" + extra
    )


async def test_engine_records_a_session(tmp_path: Path) -> None:
    rec_dir = (tmp_path / "rec").as_posix()
    engine = Engine(
        parse_profile(profile(tmp_path, f"recording: {{enabled: true, path: '{rec_dir}'}}\n"))
    )
    async with engine:
        await asyncio.wait_for(engine.capture_done.wait(), 30)
    (path,) = list((tmp_path / "rec").glob("*.mcap"))
    back = read_recording(path)
    with wave.open(str(DATA / "speech_16k.wav"), "rb") as w:
        source = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    np.testing.assert_array_equal(back.signal("/ral/mic")[:, 0], source)  # raw is bit-exact
    # Offline, frames are stamped faster than real time, so order by log time is not the
    # order of writing; check what is there.
    kinds = {type(e).__name__ for e in back.events}
    assert {"RecordingStarted", "RecordingStopped", "SpeechStarted"} <= kinds


async def test_recording_is_off_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    engine = Engine(parse_profile(profile(tmp_path)))
    async with engine:
        await asyncio.wait_for(engine.capture_done.wait(), 30)
    assert engine.recorder is None
    assert not list(tmp_path.rglob("*.mcap"))
