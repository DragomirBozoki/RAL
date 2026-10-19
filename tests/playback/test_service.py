import asyncio

import numpy as np
import pytest

from ral.core import AudioFormat, AudioFrame, ChannelMap, PlaybackService, now_ns
from ral.core import events as ev
from ral.playback.service import audio_frames

from .conftest import RigFactory, gated, settle, tag_of


async def test_implements_the_protocol(rig_factory: RigFactory) -> None:
    async with rig_factory() as rig:
        assert isinstance(rig.queue, PlaybackService)


async def test_clip_plays_whole_with_events_and_reference(rig_factory: RigFactory) -> None:
    clip = (0.5 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16000)).astype(np.float32)
    async with rig_factory() as rig:
        handle = rig.queue.play(audio_frames(clip), tag="hello")
        assert await rig.queue.wait(handle) is False
    out = rig.output()[:, 0]
    np.testing.assert_allclose(out[:8000], clip, atol=1e-4)
    started, ended = rig.drain_events()
    assert isinstance(started, ev.PlaybackStarted)
    assert started.tag == "hello"
    assert started.handle_id == handle.id
    assert isinstance(ended, ev.PlaybackEnded)
    assert not ended.interrupted
    assert ended.t_ns - started.t_ns == 50 * 10_000_000
    ref = np.concatenate([f.data[:, 0] for f in rig.ref.drain()])
    np.testing.assert_allclose(ref[:8000], clip, atol=1e-4)


async def test_48k_stereo_int16_sink(rig_factory: RigFactory) -> None:
    clip = (0.5 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
    async with rig_factory(AudioFormat(48000, 2, "int16")) as rig:
        await rig.queue.wait(rig.queue.play(audio_frames(clip)))
    out = rig.output()
    assert out.shape[1] == 2
    np.testing.assert_array_equal(out[:, 0], out[:, 1])
    t = np.arange(len(out)) / 48000
    ideal = 0.5 * np.sin(2 * np.pi * 440 * t)
    inner = slice(2400, 45600)
    assert np.sqrt(np.mean((out[inner, 0] - ideal[inner]) ** 2)) < 1e-3


async def test_same_priority_plays_in_order(rig_factory: RigFactory) -> None:
    async with rig_factory() as rig:
        a = rig.queue.play(gated([0.1] * 3), tag="a")
        b = rig.queue.play(gated([0.2] * 3), tag="b")
        await rig.queue.wait(b)
        assert not await rig.queue.wait(a)
    out = rig.output()[:, 0]
    np.testing.assert_allclose(out[:480], 0.1, atol=1e-4)
    np.testing.assert_allclose(out[480:960], 0.2, atol=1e-4)


async def test_higher_priority_preempts_with_fade(rig_factory: RigFactory) -> None:
    gate = asyncio.Event()
    async with rig_factory() as rig:
        low = rig.queue.play(gated([0.5] * 100, gate, after=10), priority=10, tag="music")
        await settle()
        high = rig.queue.play(gated([0.25] * 5), priority=90, tag="alert")
        await settle()
        gate.set()
        assert await rig.queue.wait(low) is True
        assert await rig.queue.wait(high) is False
    out = rig.output()[:, 0]
    # 10 full frames of music, then a 40 ms fade, then the alert.
    np.testing.assert_allclose(out[:1600], 0.5, atol=1e-4)
    fade = out[1600:2240]
    assert np.all(np.diff(fade) <= 1e-6)
    assert fade[-1] < 0.01
    np.testing.assert_allclose(out[2240:3040], 0.25, atol=1e-4)
    kinds = [(type(e).__name__, tag_of(e)) for e in rig.drain_events()]
    assert kinds == [
        ("PlaybackStarted", "music"),
        ("PlaybackEnded", "music"),
        ("PlaybackStarted", "alert"),
        ("PlaybackEnded", "alert"),
    ]


async def test_uninterruptible_item_is_not_preempted(rig_factory: RigFactory) -> None:
    gate = asyncio.Event()
    async with rig_factory() as rig:
        low = rig.queue.play(gated([0.5] * 20, gate, 5), priority=10, interruptible=False)
        await settle()
        high = rig.queue.play(gated([0.25] * 5), priority=90)
        await settle()
        gate.set()
        assert await rig.queue.wait(low) is False
        await rig.queue.wait(high)
    out = rig.output()[:, 0]
    np.testing.assert_allclose(out[:3200], 0.5, atol=1e-4)


async def test_stop_fades_and_stops_cleanly(rig_factory: RigFactory) -> None:
    gate = asyncio.Event()
    async with rig_factory() as rig:
        h = rig.queue.play(gated([0.5] * 50, gate, after=5))
        await settle()
        stopping = asyncio.create_task(rig.queue.stop(h, fade_ms=20))
        await settle()
        gate.set()
        await stopping
        assert not rig.queue.busy
    out = rig.output()[:, 0]
    assert len(out) == 5 * 160 + 320
    assert out[-1] < 0.01
    assert np.all(np.diff(out[800:]) <= 1e-6)


async def test_stop_queued_item_drops_it_without_events(rig_factory: RigFactory) -> None:
    gate = asyncio.Event()
    async with rig_factory() as rig:
        a = rig.queue.play(gated([0.5] * 5, gate, after=2))
        b = rig.queue.play(gated([0.2] * 5))
        await settle()
        await rig.queue.stop(b)
        gate.set()
        await rig.queue.wait(a)
    assert [getattr(e, "handle_id", None) for e in rig.drain_events()] == [a.id, a.id]


async def test_interrupt_fades_active_and_drops_queued_interruptible(
    rig_factory: RigFactory,
) -> None:
    gate = asyncio.Event()
    async with rig_factory() as rig:
        a = rig.queue.play(gated([0.5] * 50, gate, after=3), tag="answer")
        b = rig.queue.play(gated([0.5] * 5), tag="answer-2")
        c = rig.queue.play(gated([0.1] * 2), interruptible=False, tag="beep")
        await settle()
        assert rig.queue.interrupt(fade_ms=10) == a
        gate.set()
        assert await rig.queue.wait(a) is True
        assert await rig.queue.wait(b) is True
        assert await rig.queue.wait(c) is False


async def test_duck_lowers_the_level(rig_factory: RigFactory) -> None:
    async with rig_factory() as rig:
        rig.queue.duck(-20.0, duration_ms=300)
        await rig.queue.wait(rig.queue.play(gated([0.5] * 50)))
    out = rig.output()[:, 0]
    assert out[100 * 16 : 250 * 16].max() == pytest.approx(0.05, abs=2e-3)
    assert out[400 * 16 :].min() == pytest.approx(0.5, abs=1e-3)


async def test_speaking_at_follows_dac_intervals(rig_factory: RigFactory) -> None:
    async with rig_factory() as rig:
        await rig.queue.wait(rig.queue.play(gated([0.5] * 10)))
        started, ended = rig.drain_events()
    q = rig.queue
    assert q.speaking_at(started.t_ns)
    assert q.speaking_at(ended.t_ns - 1)
    assert not q.speaking_at(ended.t_ns)
    assert not q.speaking_at(started.t_ns - 1)


async def test_wrong_format_ends_the_item(
    rig_factory: RigFactory, caplog: pytest.LogCaptureFixture
) -> None:
    async def bad():  # type: ignore[no-untyped-def]
        fmt = AudioFormat(48000, 1, "float32")
        yield AudioFrame(np.zeros((480, 1), np.float32), fmt, ChannelMap.uniform(1), 0, 0, "x")

    async with rig_factory() as rig:
        await rig.queue.wait(rig.queue.play(bad()))
    assert "needs" in caplog.text


async def test_realtime_paces_and_publishes_at_dac_time(rig_factory: RigFactory) -> None:
    async with rig_factory(realtime=True) as rig:
        t0 = now_ns()
        h = rig.queue.play(audio_frames(np.full(4800, 0.3, np.float32)))
        started = await asyncio.wait_for(rig.events.get(), 2)
        assert now_ns() >= started.t_ns - 2_000_000  # published when it reached the DAC
        await rig.queue.wait(h)
        elapsed = now_ns() - t0
        ended = await asyncio.wait_for(rig.events.get(), 2)
        assert isinstance(ended, ev.PlaybackEnded)
    # 300 ms of audio; the writer may run at most ahead_ms (40) early.
    assert elapsed > 220_000_000
    assert ended.t_ns - started.t_ns == 300_000_000
