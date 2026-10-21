import asyncio
from pathlib import Path

import numpy as np
import pytest

from ral.core import events as ev
from ral.core import now_ns
from ral.metrics.latency import LatencyTracker

MS = 1_000_000


def test_turn_breakdown_adds_up() -> None:
    clock = iter([1_300 * MS])  # RAL decided the user finished at 1.3 s
    tracker = LatencyTracker(clock=lambda: next(clock))
    tracker.on_event(ev.SpeechEnded(1_000 * MS, "s", t_start_ns=0))
    tracker.on_event(
        ev.PlaybackStarted(2_050 * MS, "s", handle_id=1, tag="ws", t_request_ns=1_900 * MS)
    )
    (turn,) = tracker.turns
    assert turn.total_ms == pytest.approx(1050)
    assert turn.hangover_ms == pytest.approx(300)
    assert turn.agent_ms == pytest.approx(600)
    assert turn.playback_ms == pytest.approx(150)
    assert turn.hangover_ms + turn.agent_ms + turn.playback_ms == pytest.approx(turn.total_ms)


def test_robot_initiated_playback_and_barge_in_are_not_turns() -> None:
    tracker = LatencyTracker(clock=lambda: 10 * MS)
    tracker.on_event(ev.PlaybackStarted(5 * MS, "s", handle_id=1, tag="ws", t_request_ns=1 * MS))
    tracker.on_event(ev.SpeechEnded(8 * MS, "s", t_start_ns=0))
    tracker.on_event(
        ev.PlaybackStarted(20 * MS, "s", handle_id=2, tag="play", t_request_ns=15 * MS)
    )
    tracker.on_event(ev.PlaybackStarted(21 * MS, "s", handle_id=3, tag="ws", t_request_ns=9 * MS))
    assert tracker.turns == []  # own clip, and audio that arrived before the decision
    tracker.on_event(ev.BargeIn(30 * MS, "s", None))
    tracker.on_event(ev.PlaybackStarted(40 * MS, "s", handle_id=4, tag="ws", t_request_ns=35 * MS))
    assert tracker.turns == []


def test_summary_and_publish() -> None:
    published: list[ev.Event] = []
    t = iter(range(0, 10**12, 100 * MS))
    tracker = LatencyTracker(publish=published.append, clock=lambda: next(t))
    for k in range(10):
        tracker.on_event(ev.SpeechEnded(0, "s", t_start_ns=0))
        start = (k + 1) * 1000 * MS
        tracker.on_event(
            ev.PlaybackStarted(start, "s", handle_id=k, tag="ws", t_request_ns=start - 50 * MS)
        )
    assert len(published) == 10
    names = [s.name for s in tracker.summary()]
    assert names == ["total", "hangover", "agent", "playback"]
    assert tracker.summary()[3].median_ms == pytest.approx(50)
    assert ev.TurnLatency in {type(e) for e in published}


async def test_engine_reports_a_turn(tmp_path: Path) -> None:
    from ral.engine.engine import Engine
    from ral.playback.service import audio_frames
    from ral.profiles import parse_profile

    data = Path(__file__).resolve().parents[1] / "data"
    mic = (data / "speech_16k.wav").as_posix()
    out = (tmp_path / "o.wav").as_posix()
    prof = parse_profile(
        "ral: 1\n"
        f"capture: {{backend: file, path: '{mic}', realtime: true}}\n"
        f"playback: {{backend: file, path: '{out}', realtime: true}}\n"
        "echo: {mode: off}\npipeline: [aec, vad]\nvad: {engine: energy, hangover_ms: 200}\n"
    )
    engine = Engine(prof)
    ended = engine.bus.subscribe(ev.SpeechEnded)
    turns = engine.bus.subscribe(ev.TurnLatency)
    async with engine:
        await asyncio.wait_for(ended.get(), 8)  # the user stopped speaking
        t_heard = now_ns()
        await asyncio.sleep(0.1)  # the "agent" thinks
        assert engine.playback is not None
        thought_ms = (now_ns() - t_heard) / 1e6  # asyncio timers are coarse on Windows
        engine.playback.play(audio_frames(np.zeros(1600, np.float32)), tag="ws")
        turn = await asyncio.wait_for(turns.get(), 3)
    assert isinstance(turn, ev.TurnLatency)
    assert thought_ms - 1 <= turn.agent_ms < thought_ms + 300
    assert turn.hangover_ms >= 150  # the 200 ms VAD hangover, measured
    assert turn.total_ms == pytest.approx(turn.hangover_ms + turn.agent_ms + turn.playback_ms)
