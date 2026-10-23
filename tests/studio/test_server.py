import asyncio
import json
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
import yaml

from ral.bridges import start_bridges, stop_bridges
from ral.core import AudioFormat, AudioFrame, ChannelMap
from ral.core.processing import FrameContext
from ral.dsp.echo import ApmProcessor, DuplexGate
from ral.engine.capture import CaptureResult
from ral.engine.engine import Engine
from ral.profiles import parse_profile
from ral.studio import LiveAnalyser, StudioBridge


def result(k: int, data: np.ndarray) -> CaptureResult:
    fmt = AudioFormat(16000, data.shape[1], "float32")
    frame = AudioFrame(data, fmt, ChannelMap.uniform(data.shape[1]), k * 10_000_000, k, "mic")
    clean = AudioFrame(
        data[:, :1].copy(), AudioFormat(16000, 1, "float32"), ChannelMap.uniform(1), 0, k, "x"
    )
    return CaptureResult(frame, clean, FrameContext("s", robot_speaking=True))


def test_analyser_levels_and_spectrum() -> None:
    an = LiveAnalyser(frames_per_message=10)
    t = np.arange(1600) / 16000
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
    msgs = []
    for k in range(10):
        data = np.zeros((160, 2), np.float32)
        data[:, 0] = tone[k * 160 : (k + 1) * 160]
        msgs.append(an.feed(result(k, data)))
    assert msgs[:9] == [None] * 9
    msg = msgs[9]
    assert msg is not None
    assert msg["raw_db"][0] == pytest.approx(-9.0, abs=0.1)  # sine power 0.125
    assert msg["raw_db"][1] == -100.0
    assert msg["speaking"] is True
    assert len(msg["spectrum"]) == 128
    assert int(np.argmax(msg["spectrum"])) == 16  # 1000 Hz in 62.5 Hz bands


def write_wav(path: Path, seconds: float = 4.0) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((int(16000 * seconds), 2)) * 0.01
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((x * 32767).astype("<i2").tobytes())


@pytest.fixture
async def studio(tmp_path: Path) -> AsyncIterator[tuple[Engine, StudioBridge]]:
    mic = tmp_path / "mic.wav"
    write_wav(mic)
    prof = parse_profile(
        "ral: 1\n"
        f"capture: {{backend: file, path: '{mic.as_posix()}', realtime: true}}\n"
        f"playback: {{backend: file, path: '{(tmp_path / 'o.wav').as_posix()}', "
        "realtime: true}\n"
        "echo: {mode: gate}\npipeline: [aec]\nbridges: {studio: {port: 0}}\n"
    )
    engine = Engine(prof)
    async with engine:
        bridges = await start_bridges(engine, prof.bridges)
        (bridge,) = bridges
        assert isinstance(bridge, StudioBridge)
        try:
            yield engine, bridge
        finally:
            await stop_bridges(bridges)


async def request(
    port: int, method: str, path: str, body: object = None, headers: dict[str, str] | None = None
) -> tuple[int, bytes]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    payload = b"" if body is None else json.dumps(body).encode()
    hdrs = {"Host": f"127.0.0.1:{port}", "Content-Length": str(len(payload))}
    if body is not None:
        hdrs["Content-Type"] = "application/json"
    hdrs.update(headers or {})
    head = f"{method} {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in hdrs.items())
    writer.write(head.encode() + b"\r\n" + payload)
    data = await asyncio.wait_for(reader.read(), 5)
    writer.close()
    status_line, _, rest = data.partition(b"\r\n")
    return int(status_line.split()[1]), rest.partition(b"\r\n\r\n")[2]


async def test_page_and_state(studio: tuple[Engine, StudioBridge]) -> None:
    engine, bridge = studio
    status, body = await request(bridge.port, "GET", "/")
    assert status == 200
    assert b"<title>RAL Studio</title>" in body
    status, body = await request(bridge.port, "GET", "/api/state")
    state = json.loads(body)
    assert state["echo_mode"] == "gate"
    assert state["capture"]["roles"] == ["mic", "mic"]
    assert state["session_id"] == engine.session_id
    assert (await request(bridge.port, "GET", "/nope"))[0] == 404


async def test_sse_streams_state_meters_and_events(studio: tuple[Engine, StudioBridge]) -> None:
    engine, bridge = studio
    reader, writer = await asyncio.open_connection("127.0.0.1", bridge.port)
    writer.write(b"GET /events HTTP/1.1\r\nHost: x\r\n\r\n")
    kinds: set[str] = set()

    async def read() -> None:
        while not {"state", "meters", "event"} <= kinds:
            line = await reader.readline()
            if line.startswith(b"event: "):
                kinds.add(line[7:].strip().decode())
                if kinds == {"state", "meters"}:
                    engine.set_echo_mode("off")  # publishes ProfileReloaded

    await asyncio.wait_for(read(), 10)
    writer.close()


async def test_echo_mode_switches_live(studio: tuple[Engine, StudioBridge]) -> None:
    pytest.importorskip("livekit.rtc")
    engine, bridge = studio
    assert engine.graph is not None
    assert isinstance(engine.graph.stages[0], DuplexGate)
    status, body = await request(bridge.port, "POST", "/api/echo", {"mode": "software_ref"})
    assert status == 200
    assert json.loads(body)["echo_mode"] == "software_ref"
    assert engine.profile.echo.mode == "software_ref"
    assert any(isinstance(s, ApmProcessor) for s in engine.graph.stages)
    status, body = await request(bridge.port, "POST", "/api/echo", {"mode": "hardware"})
    assert status == 400
    assert "processed" in json.loads(body)["error"]  # this device has no AEC channel
    assert engine.profile.echo.mode == "software_ref"


async def test_channel_map_editor_returns_a_valid_profile(
    studio: tuple[Engine, StudioBridge],
) -> None:
    _, bridge = studio
    status, body = await request(
        bridge.port, "POST", "/api/channels", {"channels": ["mic", "reference"]}
    )
    assert status == 200
    profile = yaml.safe_load(json.loads(body)["yaml"])
    assert profile["capture"]["channels"] == ["mic", "reference"]
    parse_profile(yaml.safe_dump(profile))
    status, body = await request(bridge.port, "POST", "/api/channels", {"channels": ["mic"]})
    assert status == 400
    assert "2 channels" in json.loads(body)["error"]
    status, _ = await request(bridge.port, "POST", "/api/channels", {"channels": ["mic", "ear"]})
    assert status == 400


async def test_other_sites_cannot_drive_it(studio: tuple[Engine, StudioBridge]) -> None:
    engine, bridge = studio
    status, _ = await request(
        bridge.port, "POST", "/api/echo", {"mode": "off"}, {"Origin": "http://evil.example"}
    )
    assert status == 403
    status, _ = await request(
        bridge.port, "POST", "/api/echo", {"mode": "off"}, {"Content-Type": "text/plain"}
    )
    assert status == 415
    assert engine.profile.echo.mode == "gate"
