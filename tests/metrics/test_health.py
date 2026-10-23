import asyncio
import wave
from pathlib import Path

import numpy as np
import pytest

from ral.core import events as ev
from ral.metrics.health import DeadChannel, HealthMonitor

FRAME_NS = 10_000_000


def frames(
    seconds: float, channels: int, level: float, clip_from: float | None = None
) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    out = []
    for k in range(int(seconds * 100)):
        x = (rng.standard_normal((160, channels)) * level).astype(np.float32)
        if clip_from is not None and k >= clip_from * 100:
            x[:, 0] = np.clip(x[:, 0] * 1000, -1, 1)
        out.append(x)
    return out


def run(mon: HealthMonitor, blocks: list[np.ndarray]) -> None:
    for k, b in enumerate(blocks):
        mon.observe(b, k * FRAME_NS)


def test_clipping_is_reported_per_channel_and_second() -> None:
    events: list[ev.Event] = []
    mon = HealthMonitor("s", events.append)
    run(mon, frames(3, 2, 0.01, clip_from=1.0))
    clips = [e for e in events if isinstance(e, ev.Clipping)]
    assert clips
    assert {e.channel for e in clips} == {0}
    assert all(e.share > 0.5 for e in clips)


def test_dead_channel_after_five_seconds_and_noise_floor() -> None:
    events: list[ev.Event] = []
    mon = HealthMonitor("s", events.append)
    blocks = frames(7, 2, 0.01)
    for b in blocks:
        b[:, 1] = 0.0
    run(mon, blocks)
    dead = [e for e in events if isinstance(e, DeadChannel)]
    assert [e.channel for e in dead] == [1]  # reported once
    assert dead[0].seconds == pytest.approx(5.0, abs=0.02)
    floor = mon.noise_floor_db()
    assert floor[0] == pytest.approx(-40, abs=1)
    assert mon.snapshot()["dead_channels"] == [1]


def test_xruns_are_reported_as_increases() -> None:
    counts = {"capture.overrun": 0, "playback.underflow": 3}
    events: list[ev.Event] = []
    mon = HealthMonitor("s", events.append, lambda: dict(counts))
    mon.poll(0)
    counts["capture.overrun"] = 2
    counts["playback.underflow"] = 3
    mon.poll(1_000_000_000)
    (xrun,) = [e for e in events if isinstance(e, ev.Xrun)]
    assert (xrun.stream, xrun.kind, xrun.count) == ("capture", "overrun", 2)
    assert mon.snapshot()["xruns"] == {"capture.overrun": 2}


def test_a_stalled_capture_is_a_lost_device() -> None:
    events: list[ev.Event] = []
    mon = HealthMonitor("s", events.append, device_id="portaudio:wasapi:3", stall_s=2.0)
    mon.observe(np.zeros((160, 1), np.float32), 0)
    t = mon._last_frame_wall
    assert t is not None
    mon.poll(t + 1_000_000_000)
    assert not events
    mon.poll(t + 3_000_000_000)
    (lost,) = [e for e in events if isinstance(e, ev.DeviceLost)]
    assert lost.device_id == "portaudio:wasapi:3"
    assert "no audio" in lost.reason
    mon.poll(t + 4_000_000_000)
    assert len([e for e in events if isinstance(e, ev.DeviceLost)]) == 1  # once per stall
    assert mon.snapshot()["capture_alive"] is False


async def test_engine_reports_health(tmp_path: Path) -> None:
    from ral.engine.engine import Engine
    from ral.profiles import parse_profile

    mic = tmp_path / "two.wav"
    rng = np.random.default_rng(1)
    x = np.zeros((16000 * 7, 2))
    x[:, 0] = rng.standard_normal(len(x)) * 0.01
    x[16000:32000, 0] = np.clip(x[16000:32000, 0] * 1000, -1, 1)
    with wave.open(str(mic), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    prof = parse_profile(
        "ral: 1\n"
        f"capture: {{backend: file, path: '{mic.as_posix()}', realtime: false}}\n"
        f"playback: {{backend: file, path: '{(tmp_path / 'o.wav').as_posix()}', realtime: false}}\n"
        "echo: {mode: off}\npipeline: [aec]\n"
    )
    engine = Engine(prof)
    events = engine.bus.subscribe(ev.Clipping, DeadChannel)
    async with engine:
        await asyncio.wait_for(engine.capture_done.wait(), 30)
        for _ in range(50):
            await asyncio.sleep(0.01)
        assert engine.health is not None
        snap = engine.health.snapshot()
    got = []
    while (e := events.get_nowait()) is not None:
        got.append(e)
    assert any(isinstance(e, ev.Clipping) and e.channel == 0 for e in got)
    assert any(isinstance(e, DeadChannel) and e.channel == 1 for e in got)
    assert snap["frames"] == 700
    assert snap["dsp_us_per_frame"] is not None
    assert "drift_ppm" in snap
