import threading

import numpy as np
import pytest

from ral.core import AudioFormat, AudioFrame, ChannelMap, ChannelRole, ReferenceProvider
from ral.devices import FileBackend
from ral.playback.reference import ReferenceBuffer

T0 = 7_000_000_000
FRAME_NS = 10_000_000


def block(fmt: AudioFormat, values: np.ndarray) -> np.ndarray:
    """A (n, channels) block in the sink's dtype holding `values` on every channel."""
    x = np.repeat(values[:, None], fmt.channels, axis=1)
    return (x * 32768).astype(np.int16) if fmt.dtype == "int16" else x.astype(np.float32)


def test_implements_the_protocol() -> None:
    assert isinstance(ReferenceBuffer(AudioFormat(16000, 1, "float32")), ReferenceProvider)


def test_frames_carry_dac_time_and_come_out_by_deadline() -> None:
    fmt = AudioFormat(16000, 1, "float32")
    ref = ReferenceBuffer(fmt)
    for k in range(5):
        ref.push(block(fmt, np.full(160, k / 10)), T0 + k * FRAME_NS)
    early = ref.frames_until(T0 + 2 * FRAME_NS)
    assert [f.t_ns for f in early] == [T0, T0 + FRAME_NS]
    assert ref.frames_until(T0 + 2 * FRAME_NS) == []
    rest = ref.frames_until(T0 + 100 * FRAME_NS)
    assert [f.seq for f in rest] == [2, 3, 4]
    np.testing.assert_allclose(rest[0].data[:, 0], 0.2, atol=1e-6)
    assert rest[0].channel_map.roles == (ChannelRole.REFERENCE,)


def test_blocks_of_any_size_are_reframed() -> None:
    fmt = AudioFormat(16000, 1, "float32")
    ref = ReferenceBuffer(fmt)
    x = np.arange(1600, dtype=np.float32) / 1600
    i = 0
    for n in [7, 333, 160, 1, 99, 1000]:
        ref.push(x[i : i + n, None], T0 + i * 1_000_000_000 // 16000)
        i += n
    frames = ref.drain()
    assert len(frames) == 10
    np.testing.assert_allclose(np.concatenate([f.data[:, 0] for f in frames]), x, atol=1e-7)
    assert [f.t_ns for f in frames] == [T0 + k * FRAME_NS for k in range(10)]


def test_stereo_48k_int16_is_downmixed_and_resampled() -> None:
    fmt = AudioFormat(48000, 2, "int16")
    ref = ReferenceBuffer(fmt)
    t = np.arange(48000) / 48000
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([tone, tone], axis=1)
    stereo[:, 1] = 0.0  # one silent side halves the mono level
    pcm = (stereo * 32767).astype(np.int16)
    for k in range(100):
        ref.push(pcm[k * 480 : (k + 1) * 480], T0 + k * FRAME_NS)
    frames = ref.drain()
    assert len(frames) == 99  # the resampler holds the last frame for its lookahead
    assert frames[0].fmt == AudioFormat(16000, 1, "float32")
    y = np.concatenate([f.data[:, 0] for f in frames])
    ideal = 0.25 * np.sin(2 * np.pi * 440 * np.arange(len(y)) / 16000)
    assert np.sqrt(np.mean((y[800:] - ideal[800:]) ** 2)) < 1e-3


async def test_file_sink_render_hook_feeds_the_buffer(tmp_path) -> None:  # type: ignore[no-untyped-def]
    fmt = AudioFormat(16000, 1, "float32")
    backend = FileBackend(sinks={"spk": tmp_path / "out.wav"})
    (dev,) = backend.discover()
    sink = backend.open_sink(dev, fmt)
    ref = ReferenceBuffer(fmt)
    sink.on_rendered(ref.push)
    await sink.start()
    cmap = ChannelMap.uniform(1)
    for k in range(20):
        sink.write(AudioFrame(np.full((160, 1), k / 100, np.float32), fmt, cmap, 0, k, "x"))
    await sink.stop()
    frames = ref.drain()
    assert [f.seq for f in frames] == list(range(20))
    steps = np.diff([f.t_ns for f in frames])
    assert set(steps.tolist()) == {FRAME_NS}
    assert frames[5].data[0, 0] == pytest.approx(0.05, abs=1e-4)


def test_producer_thread_and_consumer_see_every_sample() -> None:
    fmt = AudioFormat(16000, 1, "float32")
    ref = ReferenceBuffer(fmt, capacity_s=5.0)
    total = 16000 * 3
    x = np.arange(total, dtype=np.float32) / total

    def produce() -> None:
        rng = np.random.default_rng(3)
        i = 0
        while i < total:
            n = int(rng.integers(1, 700))
            ref.push(x[i : i + n, None], T0 + i * 1_000_000_000 // 16000)
            i += n

    th = threading.Thread(target=produce)
    th.start()
    got = []
    while th.is_alive() or len(got) < total // 160:
        got.extend(ref.drain())
        if not th.is_alive() and len(got) >= total // 160:
            break
    th.join()
    got.extend(ref.drain())
    y = np.concatenate([f.data[:, 0] for f in got])
    np.testing.assert_allclose(y, x, atol=1e-7)
    assert ref.overruns == 0
