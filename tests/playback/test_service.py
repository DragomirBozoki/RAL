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
    async with rig_factory(realtime=True) as rig:
        low = rig.queue.play(gated([0.5] * 100), priority=10, tag="music")
        await asyncio.sleep(0.15)
        high = rig.queue.play(gated([0.25] * 10), priority=90, tag="alert")
        assert await rig.queue.wait(low) is True
        assert await rig.queue.wait(high) is False
        await asyncio.sleep(0.1)  # the file sink renders on coarse timer ticks
    out = rig.output()[:, 0]
    start = int(np.argmax(out > 0.4))
    fade_start = start + int(np.argmax(out[start:] < 0.49))
    at_alert = (np.abs(out - 0.25) < 1e-3).astype(int)
    alert = np.flatnonzero(np.convolve(at_alert, np.ones(160, int), "valid") == 160)
    assert alert.size > 0  # a run of at least one frame at the alert level
    fade = out[fade_start : alert[0]]
    fade = fade[fade > 1e-4]
    assert np.all(np.diff(fade) <= 1e-6)  # a smooth ramp down, not a cut
    assert 400 <= fade.size <= 660  # most of the 40 ms (640 samples) raised cosine
    assert not np.any(out[alert[0] :] > 0.3)  # the music never comes back
    # Ended(music) and Started(alert) share one DAC time, so check the timeline by t_ns.
    t = {(type(e).__name__, tag_of(e)): e.t_ns for e in rig.drain_events()}
    assert t[("PlaybackStarted", "music")] < t[("PlaybackEnded", "music")]
    assert t[("PlaybackEnded", "music")] <= t[("PlaybackStarted", "alert")]
    assert t[("PlaybackStarted", "alert")] < t[("PlaybackEnded", "alert")]


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
    async with rig_factory(realtime=True) as rig:
        h = rig.queue.play(gated([0.5] * 100))
        await asyncio.sleep(0.15)
        await rig.queue.stop(h, fade_ms=20)
        assert not rig.queue.busy
        await asyncio.sleep(0.1)
    out = rig.output()[:, 0]
    loud = np.flatnonzero(out > 0.45)
    tail = out[loud[-1] :]
    ramp_end = int(np.argmax(tail < 1e-4))
    assert 200 <= ramp_end <= 330  # the lower 80 % of a 20 ms (320-sample) ramp
    assert np.all(np.diff(tail[: ramp_end + 1]) <= 1e-6)
    assert np.all(np.abs(tail[ramp_end:]) < 1e-4)  # silence after the fade


async def test_stop_with_a_stalled_producer_ends_at_once(rig_factory: RigFactory) -> None:
    gate = asyncio.Event()
    async with rig_factory() as rig:
        h = rig.queue.play(gated([0.5] * 50, gate, after=5))
        await settle()
        await asyncio.wait_for(rig.queue.stop(h, fade_ms=20), 1)
    assert len(rig.output()) == 5 * 160


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
    # 300 ms of audio; the writer may run at most its lead (40 to 62 ms) early.
    assert elapsed > 220_000_000
    assert ended.t_ns - started.t_ns == 300_000_000
