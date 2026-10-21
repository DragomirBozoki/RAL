"""ROS 2 backend and bridge against a fake rclpy (tests/fakes/rclpy_fake.py).

Run against a real ROS 2 installation on Ubuntu with the hardware-free commands in
docs/PROGRESS.md (maintainer tasks); these tests only prove RAL's side of the contract.
"""

from __future__ import annotations

import array
import asyncio
import json
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from fakes import rclpy_fake as rf
from ral.core import AudioFormat, AudioFrame, ChannelMap, MissingDependencyError, now_ns
from ral.core import events as ev

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def ros(monkeypatch: pytest.MonkeyPatch) -> rf.Graph:
    return rf.install(monkeypatch)


async def until(pred: object, timeout: float = 3.0) -> bool:
    """Poll without blocking the event loop (the engine runs on it)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():  # type: ignore[operator]
            return True
        await asyncio.sleep(0.01)
    return False


def stamped(pcm: np.ndarray, ros_ns: int) -> rf.AudioDataStamped:
    msg = rf.AudioDataStamped()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ros_ns, 1_000_000_000)
    msg.audio.data = array.array("B", pcm.astype("<i2").tobytes())
    return msg


def speech() -> np.ndarray:
    with wave.open(str(DATA / "speech_16k.wav"), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").copy()


def test_without_ros2_the_error_says_to_source_it(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "rclpy", None)
    from ral.ros2 import import_rclpy

    with pytest.raises(MissingDependencyError, match="source your ROS 2 setup"):
        import_rclpy()


async def test_source_frames_carry_header_time_on_the_ral_clock(ros: rf.Graph) -> None:
    from ral.devices.ros2_topic import Ros2TopicBackend
    from ral.ros2 import Ros2Runtime

    runtime = Ros2Runtime("t")
    backend = Ros2TopicBackend(runtime)
    fmt = AudioFormat(16000, 2, "int16")
    source = backend.open_source(
        backend.device("/hal/audio/capture", "input"), fmt, ChannelMap.uniform(2)
    )
    await source.start()
    _, pub, ex, _ = rf.hal()
    # 20 ms chunks stamped 50 ms in the past (ROS time), as a HAL behind a network would.
    ramp = (np.arange(16000 * 2) % 30000).astype(np.int16).reshape(-1, 2)
    t_ral_before = now_ns()
    for k in range(25):
        chunk = ramp[k * 320 : (k + 1) * 320]
        pub.publish(stamped(chunk, time.time_ns() - 50_000_000))
    frames = []
    async for frame in source.frames():
        frames.append(frame)
        if len(frames) == 50:
            break
    await source.stop()
    ex.shutdown()
    runtime.close()
    np.testing.assert_array_equal(np.concatenate([f.data for f in frames]), ramp[:8000])
    assert [f.seq for f in frames] == list(range(50))
    assert abs(frames[0].t_ns - (t_ral_before - 50_000_000)) < 30_000_000


async def test_unstamped_messages_are_timed_on_arrival(ros: rf.Graph) -> None:
    from ral.devices.ros2_topic import Ros2TopicBackend
    from ral.ros2 import Ros2Runtime

    runtime = Ros2Runtime("t")
    backend = Ros2TopicBackend(runtime, decoder="audio_common_unstamped")
    fmt = AudioFormat(16000, 1, "int16")
    source = backend.open_source(backend.device("/mic", "input"), fmt, ChannelMap.uniform(1))
    await source.start()
    node = rf.Node("hal")
    ex = rf.SingleThreadedExecutor()
    ex.add_node(node)
    pub = node.create_publisher(rf.AudioData, "/mic", rf.QoSProfile(depth=5))
    pub.publish(rf.AudioData(array.array("B", np.ones(160, "<i2").tobytes())))
    frame = await asyncio.wait_for(anext(source.frames()), 2)
    assert abs(frame.t_ns - (now_ns() - 10_000_000)) < 50_000_000
    await source.stop()
    runtime.close()


async def test_sink_publishes_every_10_ms_with_info_and_hooks(ros: rf.Graph) -> None:
    from ral.devices.ros2_topic import Ros2TopicBackend
    from ral.ros2 import Ros2Runtime

    runtime = Ros2Runtime("t")
    backend = Ros2TopicBackend(runtime, latency_ms=60)
    fmt = AudioFormat(16000, 1, "int16")
    sink = backend.open_sink(backend.device("/hal/audio/playback", "output"), fmt)
    hooks: list[tuple[np.ndarray, int]] = []
    sink.on_rendered(lambda s, t: hooks.append((s.copy(), t)))
    await sink.start()
    tone = np.full((160, 1), 1000, np.int16)
    for k in range(5):
        sink.write(AudioFrame(tone, fmt, ChannelMap.uniform(1), 0, k, "x"))
    t0 = now_ns()
    await asyncio.sleep(0.2)
    await sink.stop()
    (pub,) = [p for p in rf.GRAPH.latched["/hal/audio/playback_info"]]
    assert pub.sample_rate == 16000
    assert pub.sample_format == "S16LE"
    sent = [p for p in ros.pubs.get("/hal/audio/playback", [])]
    assert sent == []  # destroyed on stop
    assert len(hooks) >= 15  # one block per 10 ms, silence included
    played = np.concatenate([h[0][:, 0] for h in hooks])
    assert (played[:800] == 1000).all()
    assert (played[800:] == 0).all()
    assert hooks[0][1] - t0 > 40_000_000  # DAC estimate: publish time + 60 ms


async def test_engine_runs_on_topics(ros: rf.Graph, tmp_path: Path) -> None:
    from ral.engine.engine import Engine
    from ral.playback.service import audio_frames
    from ral.profiles import parse_profile

    prof = parse_profile(
        "ral: 1\n"
        "capture: {backend: ros2_topic, topic: /hal/audio/capture, "
        "format: {rate: 16000, channels: 1, dtype: int16}}\n"
        "playback: {backend: ros2_topic, topic: /hal/audio/playback, latency_ms: 60}\n"
        "echo: {mode: off}\npipeline: [aec, vad]\nvad: {engine: energy}\n"
    )
    engine = Engine(prof)
    started = engine.bus.subscribe(ev.SpeechStarted)
    async with engine:
        _, pub, ex, _ = rf.hal()
        pcm = speech()[: 16000 * 3]
        # A HAL stamps each chunk with the capture time of its first sample.
        t0 = time.time_ns()
        for k in range(0, len(pcm), 320):
            pub.publish(stamped(pcm[k : k + 320], t0 + k * 1_000_000_000 // 16000))
            if k % 3200 == 0:
                await asyncio.sleep(0.1)
        assert isinstance(await asyncio.wait_for(started.get(), 5), ev.SpeechStarted)
        assert engine.playback is not None
        await engine.playback.wait(
            engine.playback.play(audio_frames(np.full(1600, 0.3, np.float32)))
        )
        ex.shutdown()
    assert rf.GRAPH.latched["/hal/audio/playback_info"]


async def test_bridge_publishes_clean_audio_and_events_and_plays(
    ros: rf.Graph, tmp_path: Path
) -> None:
    from ral.bridges import start_bridges, stop_bridges
    from ral.engine.engine import Engine
    from ral.profiles import parse_profile

    mic = (DATA / "speech_16k.wav").as_posix()
    out = (tmp_path / "o.wav").as_posix()
    prof = parse_profile(
        "ral: 1\n"
        f"capture: {{backend: file, path: '{mic}', realtime: true}}\n"
        f"playback: {{backend: file, path: '{out}', realtime: true}}\n"
        "echo: {mode: off}\npipeline: [aec, vad]\nvad: {engine: energy}\n"
        "bridges: {ros2: {namespace: /robot1/ral}}\n"
    )
    engine = Engine(prof)
    played = engine.bus.subscribe(ev.PlaybackStarted, ev.PlaybackEnded)
    client = rf.Node("agent")
    cex = rf.SingleThreadedExecutor()
    cex.add_node(client)
    import threading

    threading.Thread(target=cex.spin, daemon=True).start()
    clean: list[rf.AudioDataStamped] = []
    events: list[dict[str, object]] = []
    async with engine:
        bridges = await start_bridges(engine, prof.bridges)
        client.create_subscription(
            rf.AudioDataStamped, "/robot1/ral/clean", clean.append, rf.QoSProfile(depth=10)
        )
        client.create_subscription(
            rf.String,
            "/robot1/ral/events",
            lambda m: events.append(json.loads(m.data)),
            rf.QoSProfile(depth=10),
        )
        info: list[rf.AudioInfo] = []
        client.create_subscription(
            rf.AudioInfo, "/robot1/ral/clean_info", info.append, rf.QoSProfile(depth=1)
        )
        play = client.create_publisher(
            rf.AudioDataStamped, "/robot1/ral/play", rf.QoSProfile(depth=10)
        )
        control = client.create_publisher(rf.String, "/robot1/ral/control", rf.QoSProfile(depth=10))
        assert await until(lambda: len(clean) >= 20)
        assert info
        assert info[0].sample_rate == 16000
        assert len(bytes(clean[0].audio.data)) == 320
        tone = (np.full(16000, 3000)).astype("<i2")
        for k in range(0, len(tone), 1600):
            play.publish(stamped(tone[k : k + 1600], time.time_ns()))
        first = await asyncio.wait_for(played.get(), 3)
        assert isinstance(first, ev.PlaybackStarted)
        assert first.tag == "ros2"
        control.publish(rf.String(json.dumps({"type": "interrupt"})))
        ended = await asyncio.wait_for(played.get(), 3)
        assert isinstance(ended, ev.PlaybackEnded)
        assert ended.interrupted
        await stop_bridges(bridges)
    cex.shutdown()
    assert any(e["type"] == "PlaybackStarted" for e in events)


def test_bridge_and_decoder_config_errors() -> None:
    from ral.bridges import make_bridge
    from ral.core import ConfigError
    from ral.profiles import parse_profile

    with pytest.raises(ConfigError, match="unknown keys"):
        make_bridge("ros2", {"namspace": "/x"})
    with pytest.raises(ConfigError, match="topic"):
        parse_profile("capture: {backend: ros2_topic, topic: /a}\n")  # no format


def test_unknown_decoder(ros: rf.Graph) -> None:
    from ral.core import ConfigError
    from ral.ros2.decoders import make_decoder

    with pytest.raises(ConfigError, match=r"ral.decoders"):
        make_decoder("vendor_x_audio")
