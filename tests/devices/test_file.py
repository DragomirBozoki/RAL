import asyncio
import wave
from pathlib import Path

import numpy as np
import pytest

from ral.core import (
    AudioFormat,
    AudioFrame,
    AudioSink,
    AudioSource,
    ChannelMap,
    DeviceBackend,
    DeviceError,
    DeviceInfo,
    DeviceNotFoundError,
    FormatError,
    now_ns,
)
from ral.devices import FileBackend, FileSource

MS = 1_000_000


def write_wav(path: Path, samples: np.ndarray, rate: int, width: int = 2) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(samples.shape[1])
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(samples.astype(f"<i{width}").tobytes())


def read_wav(path: Path) -> tuple[bytes, tuple[int, int, int]]:
    with wave.open(str(path), "rb") as wav:
        params = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
        return wav.readframes(wav.getnframes()), params


def noise(n: int, channels: int, seed: int = 0) -> np.ndarray:
    """Full-range int16 noise, so every bit of every sample is exercised."""
    rng = np.random.default_rng(seed)
    return rng.integers(-32768, 32768, size=(n, channels), dtype=np.int16)


def device(backend: FileBackend, device_id: str) -> DeviceInfo:
    return next(d for d in backend.discover() if d.id == device_id)


async def collect(source: FileSource) -> list[AudioFrame]:
    await source.start()
    frames = [frame async for frame in source.frames()]
    await source.stop()
    return frames


async def copy_file(src: Path, dst: Path, dtype: str, rate: int, channels: int) -> list:
    """Source to sink, frame by frame, returning what the render hook saw."""
    backend = FileBackend({"in": src}, {"out": dst})
    fmt = AudioFormat(rate, channels, dtype)  # type: ignore[arg-type]
    source = backend.open_source(device(backend, "file:in:in"), fmt, ChannelMap.uniform(channels))
    sink = backend.open_sink(device(backend, "file:out:out"), fmt)
    rendered: list[tuple[np.ndarray, int]] = []
    sink.on_rendered(lambda samples, t: rendered.append((samples.copy(), t)))
    await sink.start()
    for frame in await collect(source):
        sink.write(frame)
    await sink.stop()
    return rendered


# Round trip


@pytest.mark.parametrize("dtype", ["int16", "float32"])
@pytest.mark.parametrize(("rate", "channels"), [(16000, 1), (48000, 2), (44100, 6)])
async def test_round_trip_is_bit_exact(
    tmp_path: Path, dtype: str, rate: int, channels: int
) -> None:
    samples = noise(37 * rate // 100, channels)  # a whole number of 10 ms frames
    write_wav(tmp_path / "in.wav", samples, rate)
    await copy_file(tmp_path / "in.wav", tmp_path / "out.wav", dtype, rate, channels)
    assert read_wav(tmp_path / "out.wav") == read_wav(tmp_path / "in.wav")


async def test_last_partial_frame_is_zero_padded(tmp_path: Path) -> None:
    samples = noise(1000, 2)  # 6 frames of 160 plus 40 samples
    write_wav(tmp_path / "in.wav", samples, 16000)
    await copy_file(tmp_path / "in.wav", tmp_path / "out.wav", "int16", 16000, 2)
    data, params = read_wav(tmp_path / "out.wav")
    out = np.frombuffer(data, dtype="<i2").reshape(-1, 2)
    assert params == (2, 2, 16000)
    assert out.shape == (1120, 2)
    assert np.array_equal(out[:1000], samples)
    assert not out[1000:].any()


# Source


async def test_source_frames_have_seq_t_ns_and_roles(tmp_path: Path) -> None:
    write_wav(tmp_path / "in.wav", noise(800, 2), 16000)
    backend = FileBackend({"in": tmp_path / "in.wav"})
    cmap = ChannelMap.from_names(["mic", "reference"])
    source = backend.open_source(
        device(backend, "file:in:in"), AudioFormat(16000, 2, "int16"), cmap
    )
    before = now_ns()
    frames = await collect(source)
    t0 = frames[0].t_ns
    assert before <= t0 <= now_ns()
    assert [f.seq for f in frames] == [0, 1, 2, 3, 4]
    assert [f.t_ns - t0 for f in frames] == [k * 10 * MS for k in range(5)]
    assert all(f.channel_map is cmap and f.clock_id == "file:in:in" for f in frames)


async def test_realtime_source_delivers_each_frame_after_its_last_sample(tmp_path: Path) -> None:
    write_wav(tmp_path / "in.wav", noise(3200, 1), 16000)  # 20 frames, 200 ms
    backend = FileBackend({"in": tmp_path / "in.wav"}, realtime=True)
    source = backend.open_source(
        device(backend, "file:in:in"), AudioFormat(16000, 1, "int16"), ChannelMap.uniform(1)
    )
    await source.start()
    start = now_ns()
    async for frame in source.frames():
        assert now_ns() >= frame.t_ns + 10 * MS
    elapsed = now_ns() - start
    await source.stop()
    assert 190 * MS <= elapsed < 1000 * MS


async def test_stop_ends_frames(tmp_path: Path) -> None:
    write_wav(tmp_path / "in.wav", noise(1600, 1), 16000)
    backend = FileBackend({"in": tmp_path / "in.wav"})
    source = backend.open_source(
        device(backend, "file:in:in"), AudioFormat(16000, 1, "int16"), ChannelMap.uniform(1)
    )
    await source.start()
    seen = 0
    async for _ in source.frames():
        seen += 1
        if seen == 3:
            await source.stop()
    assert seen == 3


async def test_frames_before_start_fail(tmp_path: Path) -> None:
    write_wav(tmp_path / "in.wav", noise(160, 1), 16000)
    backend = FileBackend({"in": tmp_path / "in.wav"})
    source = backend.open_source(
        device(backend, "file:in:in"), AudioFormat(16000, 1, "int16"), ChannelMap.uniform(1)
    )
    with pytest.raises(DeviceError, match="start"):
        await anext(source.frames())


# Sink


async def test_render_hook_sees_every_block_with_continuous_dac_time(tmp_path: Path) -> None:
    write_wav(tmp_path / "in.wav", noise(480, 1), 16000)
    rendered = await copy_file(tmp_path / "in.wav", tmp_path / "out.wav", "float32", 16000, 1)
    assert len(rendered) == 3
    assert all(s.dtype == np.float32 and s.shape == (160, 1) for s, _ in rendered)
    t0 = rendered[0][1]
    assert [t - t0 for _, t in rendered] == [0, 10 * MS, 20 * MS]


async def test_realtime_sink_renders_silence_on_underrun(tmp_path: Path) -> None:
    fmt = AudioFormat(16000, 1, "int16")
    backend = FileBackend(sinks={"out": tmp_path / "out.wav"}, realtime=True)
    sink = backend.open_sink(device(backend, "file:out:out"), fmt)
    rendered: list[np.ndarray] = []
    sink.on_rendered(lambda samples, t: rendered.append(samples.copy()))
    await sink.start()
    tone = np.full((160, 1), 1000, dtype=np.int16)
    sink.write(AudioFrame(tone, fmt, ChannelMap.uniform(1), 0, 0, "test"))
    while len(rendered) < 5:  # the render task runs every 10 ms
        await asyncio.sleep(0.005)
    await sink.stop()

    assert np.array_equal(rendered[0], tone)
    assert all(not block.any() for block in rendered[1:5])
    assert sink.buffer.underruns >= 4
    out = np.frombuffer(read_wav(tmp_path / "out.wav")[0], dtype="<i2")
    assert np.array_equal(out[:160], tone[:, 0])
    assert len(out) == 160 * len(rendered)


async def test_realtime_sink_drops_new_frames_when_full(tmp_path: Path) -> None:
    fmt = AudioFormat(16000, 1, "int16")
    backend = FileBackend(sinks={"out": tmp_path / "out.wav"}, realtime=True, sink_capacity_s=0.02)
    sink = backend.open_sink(device(backend, "file:out:out"), fmt)
    await sink.start()
    frame = AudioFrame(np.zeros((160, 1), np.int16), fmt, ChannelMap.uniform(1), 0, 0, "t")
    for _ in range(5):  # nothing renders in between: we never yield to the loop
        sink.write(frame)
    await sink.stop()
    assert sink.buffer.overruns == 3


async def test_write_checks_state_and_format(tmp_path: Path) -> None:
    fmt = AudioFormat(16000, 1, "int16")
    backend = FileBackend(sinks={"out": tmp_path / "out.wav"})
    sink = backend.open_sink(device(backend, "file:out:out"), fmt)
    frame = AudioFrame(np.zeros((160, 1), np.int16), fmt, ChannelMap.uniform(1), 0, 0, "t")
    with pytest.raises(DeviceError, match="start"):
        sink.write(frame)
    await sink.start()
    other = AudioFormat(16000, 1, "float32")
    with pytest.raises(FormatError, match="does not match"):
        sink.write(AudioFrame(np.zeros((160, 1), np.float32), other, frame.channel_map, 0, 0, "t"))
    await sink.stop()
    assert sink.output_latency_ns() == 0


# Backend


def test_discover_reports_file_formats(tmp_path: Path) -> None:
    write_wav(tmp_path / "a.wav", noise(160, 6), 48000)
    backend = FileBackend({"a": tmp_path / "a.wav"}, {"b": tmp_path / "b.wav"})
    a, b = backend.discover()
    assert (a.id, a.backend, a.max_input_channels, a.max_output_channels) == (
        "file:in:a",
        "file",
        6,
        0,
    )
    assert a.default_sample_rate == 48000.0
    assert a.extra == {"path": str(tmp_path / "a.wav")}
    assert (b.id, b.max_input_channels, b.default_sample_rate) == ("file:out:b", 0, None)


def test_backend_source_and_sink_satisfy_the_protocols(tmp_path: Path) -> None:
    write_wav(tmp_path / "a.wav", noise(160, 1), 16000)
    backend = FileBackend({"a": tmp_path / "a.wav"}, {"b": tmp_path / "b.wav"})
    fmt = AudioFormat(16000, 1, "int16")
    assert isinstance(backend, DeviceBackend)
    source = backend.open_source(device(backend, "file:in:a"), fmt, ChannelMap.uniform(1))
    assert isinstance(source, AudioSource)
    assert isinstance(backend.open_sink(device(backend, "file:out:b"), fmt), AudioSink)
    assert isinstance(source, FileSource)


def test_open_source_rejects_format_and_channel_map_mismatch(tmp_path: Path) -> None:
    write_wav(tmp_path / "a.wav", noise(160, 2), 16000)
    backend = FileBackend({"a": tmp_path / "a.wav"})
    dev = device(backend, "file:in:a")
    with pytest.raises(FormatError, match="16000 Hz with 2 channels"):
        backend.open_source(dev, AudioFormat(48000, 2, "int16"), ChannelMap.uniform(2))
    with pytest.raises(FormatError, match="Channel map has 1 roles"):
        backend.open_source(dev, AudioFormat(16000, 2, "int16"), ChannelMap.uniform(1))


def test_non_16_bit_and_broken_files_are_rejected(tmp_path: Path) -> None:
    write_wav(tmp_path / "wide.wav", noise(160, 1).astype(np.int32), 16000, width=4)
    (tmp_path / "junk.wav").write_bytes(b"not a wav file")
    with pytest.raises(FormatError, match="16-bit PCM only"):
        FileBackend({"w": tmp_path / "wide.wav"}).discover()
    with pytest.raises(FormatError, match="not a readable"):
        FileBackend({"j": tmp_path / "junk.wav"}).discover()
    with pytest.raises(DeviceError, match="Cannot open"):
        FileBackend({"m": tmp_path / "missing.wav"}).discover()


def test_unknown_device_is_not_found(tmp_path: Path) -> None:
    backend = FileBackend(sinks={"b": tmp_path / "b.wav"})
    stranger = DeviceInfo("portaudio:wasapi:3", "Mic", "portaudio", "WASAPI", 2, 0, 48000.0)
    with pytest.raises(DeviceNotFoundError, match="configured: file:out:b"):
        backend.open_sink(stranger, AudioFormat(16000, 1, "int16"))
    with pytest.raises(DeviceNotFoundError, match="configured: none"):
        backend.open_source(stranger, AudioFormat(16000, 1, "int16"), ChannelMap.uniform(1))
