import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("websockets")

from websockets.asyncio.client import connect

from ral.bridges import make_bridge, start_bridges, stop_bridges
from ral.bridges.websocket import WebSocketBridge
from ral.core import ConfigError
from ral.engine.engine import Engine
from ral.profiles import parse_profile

DATA = Path(__file__).resolve().parents[1] / "data"


def profile(tmp_path: Path) -> str:
    mic = (DATA / "speech_16k.wav").as_posix()
    out = (tmp_path / "out.wav").as_posix()
    return (
        "ral: 1\n"
        f"capture: {{backend: file, path: '{mic}', realtime: true}}\n"
        f"playback: {{backend: file, path: '{out}', realtime: true}}\n"
        "echo: {mode: off}\npipeline: [aec, vad]\nvad: {engine: energy}\n"
        "bridges: {websocket: {port: 0}}\n"
    )


@pytest.fixture
async def running(tmp_path: Path) -> AsyncIterator[tuple[Engine, WebSocketBridge]]:
    prof = parse_profile(profile(tmp_path))
    engine = Engine(prof)
    async with engine:
        bridges = await start_bridges(engine, prof.bridges)
        (bridge,) = bridges
        assert isinstance(bridge, WebSocketBridge)
        try:
            yield engine, bridge
        finally:
            await stop_bridges(bridges)


async def next_text(ws: Any, kind: str, timeout: float = 3.0) -> dict[str, Any]:
    async def find() -> dict[str, Any]:
        while True:
            msg = await ws.recv()
            if isinstance(msg, str) and json.loads(msg)["type"] == kind:
                return json.loads(msg)  # type: ignore[no-any-return]

    return await asyncio.wait_for(find(), timeout)


async def test_hello_audio_and_events(running: tuple[Engine, WebSocketBridge]) -> None:
    engine, bridge = running
    async with connect(f"ws://127.0.0.1:{bridge.port}") as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello"
        assert hello["protocol"] == 1
        assert hello["session_id"] == engine.session_id
        assert hello["capture"] == {"format": "pcm-s16", "rate": 16000, "channels": 1}
        binary = 0
        while binary < 20:
            msg = await asyncio.wait_for(ws.recv(), 3)
            if isinstance(msg, bytes):
                assert len(msg) == 320  # 10 ms, int16, mono, 16 kHz
                binary += 1
        await ws.send(json.dumps({"type": "ping"}))
        assert (await next_text(ws, "pong"))["type"] == "pong"


async def test_client_audio_is_played_through_the_engine(
    running: tuple[Engine, WebSocketBridge],
) -> None:
    _, bridge = running
    async with connect(f"ws://127.0.0.1:{bridge.port}?audio=0") as ws:
        await ws.recv()  # hello
        await ws.send(json.dumps({"type": "play", "tag": "answer"}))
        tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(4800) / 16000) * 32767).astype("<i2")
        for chunk in np.array_split(tone, 7):  # odd chunk sizes, as a network delivers them
            await ws.send(chunk.tobytes())
        await ws.send(json.dumps({"type": "end"}))
        started = await next_text(ws, "PlaybackStarted")
        ended = await next_text(ws, "PlaybackEnded")
        assert started["tag"] == "answer"
        assert ended["interrupted"] is False
        assert ended["t_ns"] - started["t_ns"] == 300_000_000  # 4800 samples at 16 kHz


async def test_interrupt_and_errors(running: tuple[Engine, WebSocketBridge]) -> None:
    _, bridge = running
    async with connect(f"ws://127.0.0.1:{bridge.port}?audio=0") as ws:
        await ws.recv()
        await ws.send(np.zeros(16000 * 3, "<i2").tobytes())  # 3 s, no explicit play
        await next_text(ws, "PlaybackStarted")
        await ws.send(json.dumps({"type": "interrupt"}))
        ended = await next_text(ws, "PlaybackEnded")
        assert ended["interrupted"] is True
        await ws.send("not json")
        assert (await next_text(ws, "error"))["type"] == "error"
        await ws.send(json.dumps({"type": "launch"}))
        assert "launch" in (await next_text(ws, "error"))["message"]
        await ws.send(b"\x01")  # half a sample
        assert "whole 16-bit samples" in (await next_text(ws, "error"))["message"]


def test_config_validation() -> None:
    assert isinstance(make_bridge("websocket", {"port": 9000}), WebSocketBridge)
    with pytest.raises(ConfigError, match="unknown keys"):
        make_bridge("websocket", {"prot": 9000})
    with pytest.raises(ConfigError, match="Unknown bridge"):
        make_bridge("carrier-pigeon", {})


def test_listens_on_localhost_by_default() -> None:
    assert WebSocketBridge().host == "127.0.0.1"


async def test_play_at_another_rate_is_resampled(running: tuple[Engine, WebSocketBridge]) -> None:
    _, bridge = running
    async with connect(f"ws://127.0.0.1:{bridge.port}?audio=0") as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "play", "tag": "tts", "rate": 24000}))
        tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(7200) / 24000) * 32767).astype("<i2")
        await ws.send(tone.tobytes())  # 300 ms at 24 kHz
        await ws.send(json.dumps({"type": "end"}))
        started = await next_text(ws, "PlaybackStarted")
        ended = await next_text(ws, "PlaybackEnded")
        assert ended["t_ns"] - started["t_ns"] == 300_000_000
        await ws.send(json.dumps({"type": "play", "rate": -1}))
        assert "rate" in (await next_text(ws, "error"))["message"]
