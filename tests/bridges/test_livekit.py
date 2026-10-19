import asyncio
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ral.bridges import make_bridge
from ral.bridges.livekit import LiveKitBridge, UtteranceSegmenter
from ral.core import ConfigError

DATA = Path(__file__).resolve().parents[1] / "data"


def run(seg: UtteranceSegmenter, levels: list[float]) -> list[str]:
    return [seg.feed(level) for level in levels]


def test_segmenter_starts_on_speech_and_ends_after_silence() -> None:
    seg = UtteranceSegmenter(speech_db=-50, silence_ms=30)
    out = run(seg, [-90, -90, -20, -20, -90, -20, -90, -90, -90, -90])
    assert out == [
        "skip", "skip", "start", "continue", "continue", "continue",
        "continue", "continue", "end", "skip",
    ]  # fmt: skip


def test_segmenter_drops_the_interrupted_turn_until_silence() -> None:
    seg = UtteranceSegmenter(speech_db=-50, silence_ms=20)
    assert run(seg, [-20, -20]) == ["start", "continue"]
    seg.drop_current_turn()
    assert run(seg, [-20, -20, -90, -20, -90, -90, -20]) == [
        "skip", "skip", "skip", "skip", "skip", "skip", "start",
    ]  # fmt: skip


def test_config_validation() -> None:
    bridge = make_bridge("livekit", {"url": "wss://lk.example", "room": "lab", "token": "t"})
    assert isinstance(bridge, LiveKitBridge)
    with pytest.raises(ConfigError, match="needs room"):
        LiveKitBridge.from_config({"url": "wss://lk.example"})
    with pytest.raises(ConfigError, match="unknown keys"):
        LiveKitBridge.from_config({"url": "u", "room": "r", "rooom": "x"})
    with pytest.raises(ConfigError, match="publish"):
        LiveKitBridge("u", "r", publish="everything")


def test_token_needs_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
    with pytest.raises(ConfigError, match="api_key and api_secret"):
        LiveKitBridge("u", "r")._make_token()


def test_token_is_minted_for_room_and_identity() -> None:
    pytest.importorskip("livekit.api")
    import jwt

    secret = "s" * 32
    tok = LiveKitBridge(
        "u", "lab", identity="robot-1", api_key="k", api_secret=secret
    )._make_token()
    claims = jwt.decode(tok, secret, algorithms=["HS256"])
    assert claims["sub"] == "robot-1"
    assert claims["video"]["room"] == "lab"
    assert claims["video"]["roomJoin"] is True


# Integration with a real server: livekit-server --dev --bind 127.0.0.1, then
# RAL_TEST_LIVEKIT_URL=ws://127.0.0.1:7880 uv run pytest tests/bridges/test_livekit.py

LK_URL = os.environ.get("RAL_TEST_LIVEKIT_URL")


@pytest.mark.skipif(not LK_URL, reason="set RAL_TEST_LIVEKIT_URL to a livekit-server --dev")
async def test_agent_hears_the_robot_and_the_robot_plays_the_agent(tmp_path: Path) -> None:
    from livekit import api, rtc

    from ral.bridges import start_bridges, stop_bridges
    from ral.core import events as ev
    from ral.engine.engine import Engine
    from ral.profiles import parse_profile

    assert LK_URL is not None
    room = f"ral-test-{os.getpid()}"
    mic = (DATA / "speech_16k.wav").as_posix()
    out = (tmp_path / "out.wav").as_posix()
    prof = parse_profile(
        "ral: 1\n"
        f"capture: {{backend: file, path: '{mic}', realtime: true}}\n"
        f"playback: {{backend: file, path: '{out}', realtime: true}}\n"
        "echo: {mode: off}\npipeline: [aec, vad]\nvad: {engine: energy}\n"
        f"bridges: {{livekit: {{url: '{LK_URL}', room: '{room}', api_key: devkey, "
        "api_secret: secret}}\n"
    )
    engine = Engine(prof)
    played = engine.bus.subscribe(ev.PlaybackStarted)
    agent = rtc.Room()
    heard: list[int] = []
    events: list[dict[str, Any]] = []
    readers: list[asyncio.Future[None]] = []

    def on_track(track: Any, *_: Any) -> None:
        async def read() -> None:
            async for e in rtc.AudioStream(track, sample_rate=16000, num_channels=1):
                heard.append(int(np.abs(np.frombuffer(e.frame.data, np.int16)).max()))

        readers.append(asyncio.ensure_future(read()))

    agent.on("track_subscribed", on_track)
    agent.on(
        "data_received",
        lambda p: events.append(json.loads(bytes(p.data))) if p.topic == "ral.events" else None,
    )
    grants = api.VideoGrants(room_join=True, room=room)
    token = api.AccessToken("devkey", "secret").with_identity("agent").with_grants(grants).to_jwt()
    async with engine:
        bridges = await start_bridges(engine, prof.bridges)
        try:
            await agent.connect(LK_URL, token)
            source = rtc.AudioSource(16000, 1)
            track = rtc.LocalAudioTrack.create_audio_track("agent", source)
            await agent.local_participant.publish_track(track, rtc.TrackPublishOptions())
            await asyncio.sleep(1.0)
            tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16000) * 32767).astype(
                np.int16
            )
            for k in range(50):
                block = tone[k * 160 : (k + 1) * 160]
                await source.capture_frame(
                    rtc.AudioFrame(bytearray(block.tobytes()), 16000, 1, 160)
                )
            started = await asyncio.wait_for(played.get(), 5)
            assert isinstance(started, ev.PlaybackStarted)
            assert started.tag == "livekit:agent"
            await asyncio.sleep(1.0)
        finally:
            await agent.disconnect()
            await stop_bridges(bridges)
    assert max(heard, default=0) > 1000  # the agent heard the robot's (clean) mic
    assert any(e["type"] == "PlaybackStarted" for e in events)
