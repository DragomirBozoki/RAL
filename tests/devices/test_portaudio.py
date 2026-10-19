"""PortAudio backend tests.

Most tests drive the backend through a fake ``sounddevice`` module and call the stream
callbacks by hand. Tests marked ``hardware`` use the real devices; pick them with
RAL_TEST_INPUT and RAL_TEST_OUTPUT (a part of the device name) or use the defaults.
"""

from __future__ import annotations

import asyncio
import builtins
import itertools
import os
import sys
from types import SimpleNamespace
from typing import Any

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
    MissingDependencyError,
    now_ns,
)
from ral.devices.portaudio import (
    PaClockMapper,
    PortAudioBackend,
    group_by_host_api,
    host_api_slug,
    load_sounddevice,
    preferred_host_apis,
)

MS = 1_000_000

# A fake sounddevice


class FakePortAudioError(Exception):
    pass


class FakeStream:
    def __init__(self, sd: FakeSd, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.latency = 0.020
        self.state = "open"
        self._sd = sd
        sd.streams.append(self)

    def start(self) -> None:
        if self._sd.fail_open:
            raise FakePortAudioError("Invalid sample rate [PaErrorCode -9997]")
        self.state = "started"

    def stop(self) -> None:
        self.state = "stopped"

    def close(self) -> None:
        self.state = "closed"


def raw_device(index: int, api: int, name: str, inputs: int, outputs: int, rate: float) -> dict:
    return {
        "index": index,
        "hostapi": api,
        "name": name,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
        "default_samplerate": rate,
        "default_low_input_latency": 0.003,
        "default_low_output_latency": 0.01,
    }


class FakeSd:
    PortAudioError = FakePortAudioError

    def __init__(self) -> None:
        self.streams: list[FakeStream] = []
        self.fail_open = False
        self.apis = [
            {"name": "MME", "default_input_device": 0, "default_output_device": 1},
            {"name": "Windows WASAPI", "default_input_device": 2, "default_output_device": 3},
        ]
        self.devices = [
            raw_device(0, 0, "Microphone Array (Realtek(R) Au", 2, 0, 44100.0),
            raw_device(1, 0, "Speakers (Realtek(R) Audio)", 0, 2, 44100.0),
            raw_device(2, 1, "Microphone Array (Realtek(R) Audio)", 2, 0, 48000.0),
            raw_device(3, 1, "Speakers (Realtek(R) Audio)", 0, 2, 48000.0),
            raw_device(4, 1, "Odd Card", 1, 1, 22050.0),
        ]

    def query_hostapis(self) -> list[dict]:
        return self.apis

    def query_devices(self) -> list[dict]:
        return self.devices

    def get_portaudio_version(self) -> tuple[int, str]:
        return 1246976, "PortAudio fake"

    def InputStream(self, **kwargs: Any) -> FakeStream:  # sounddevice's name
        return FakeStream(self, **kwargs)

    def OutputStream(self, **kwargs: Any) -> FakeStream:
        return FakeStream(self, **kwargs)


def time_info(current: float, adc: float = 0.0, dac: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(currentTime=current, inputBufferAdcTime=adc, outputBufferDacTime=dac)


def flags(**set_flags: bool) -> SimpleNamespace:
    return SimpleNamespace(
        input_overflow=set_flags.get("input_overflow", False),
        output_underflow=set_flags.get("output_underflow", False),
    )


@pytest.fixture
def sd() -> FakeSd:
    return FakeSd()


@pytest.fixture
def backend(sd: FakeSd, monkeypatch: pytest.MonkeyPatch) -> PortAudioBackend:
    monkeypatch.setattr(sys, "platform", "win32")
    return PortAudioBackend(sd)  # type: ignore[arg-type]


def pa_now() -> float:
    """PortAudio time that equals the RAL clock, as measured on Windows."""
    return now_ns() / 1e9


# Helpers and clock mapping


def test_host_api_slugs() -> None:
    assert host_api_slug("Windows WASAPI") == "wasapi"
    assert host_api_slug("Windows WDM-KS") == "wdm-ks"
    assert host_api_slug("MME") == "mme"
    assert host_api_slug("Core Audio") == "core-audio"


def test_preferred_host_apis_per_platform() -> None:
    assert preferred_host_apis("win32") == ("Windows WASAPI",)
    assert preferred_host_apis("linux")[0] == "ALSA"
    assert preferred_host_apis("darwin") == ("Core Audio",)
    assert preferred_host_apis("sunos") == ()


def test_clock_mapper_keeps_the_lowest_offset() -> None:
    mapper = PaClockMapper()
    mapper.observe(10.0, ral_now_ns=10_000_300_000)  # 0.3 ms callback delay
    mapper.observe(10.01, ral_now_ns=10_010_100_000)  # 0.1 ms: the better estimate
    assert mapper.offset_ns == 100_000
    mapper.observe(10.02, ral_now_ns=10_020_100_000 + 128 * 1000)  # a late callback
    assert mapper.offset_ns == 101_000  # rises by 1/128 of the difference only
    assert mapper.to_ral_ns(11.0) == 11_000_000_000 + 101_000


def test_clock_mapper_follows_a_step_in_the_pa_clock() -> None:
    mapper = PaClockMapper()
    mapper.observe(100.0, ral_now_ns=5 * 10**9)
    for k in range(1, 2000):  # PortAudio's clock steps back by 1 s
        mapper.observe(99.0 + k * 0.01, ral_now_ns=5 * 10**9 + k * 10 * MS)
    assert mapper.offset_ns is not None
    assert abs(mapper.offset_ns - (5 * 10**9 - 99 * 10**9)) < MS


# Discovery


def test_discover_puts_wasapi_first_with_stable_ids(backend: PortAudioBackend) -> None:
    devices = backend.discover()
    assert [d.id for d in devices] == [
        "portaudio:wasapi:2",
        "portaudio:wasapi:3",
        "portaudio:wasapi:4",
        "portaudio:mme:0",
        "portaudio:mme:1",
    ]
    mic = devices[0]
    assert (mic.host_api, mic.max_input_channels, mic.default_sample_rate) == (
        "Windows WASAPI",
        2,
        48000.0,
    )
    assert mic.extra["default_input"] == "true"
    assert mic.extra["low_input_latency_ms"] == "3.0"
    groups = group_by_host_api(devices)
    assert list(groups) == ["Windows WASAPI", "MME"]
    assert len(groups["MME"]) == 2


def test_default_and_find_prefer_wasapi(backend: PortAudioBackend) -> None:
    assert backend.default_device("input").id == "portaudio:wasapi:2"
    assert backend.default_device("output").id == "portaudio:wasapi:3"
    assert backend.find("speakers", "output").id == "portaudio:wasapi:3"
    with pytest.raises(DeviceNotFoundError, match="ral doctor"):
        backend.find("speakers", "input")


def test_default_device_missing(backend: PortAudioBackend, sd: FakeSd) -> None:
    for api in sd.apis:
        api["default_input_device"] = -1
    with pytest.raises(DeviceNotFoundError, match="no default input"):
        backend.default_device("input")


def test_native_format(backend: PortAudioBackend) -> None:
    mic = backend.find("microphone", "input")
    assert backend.native_format(mic, "input") == AudioFormat(48000, 2, "float32")
    assert backend.native_format(mic, "input", "int16", 1) == AudioFormat(48000, 1, "int16")
    with pytest.raises(FormatError, match="no output channels"):
        backend.native_format(mic, "output")
    with pytest.raises(FormatError, match="22050 Hz"):
        backend.native_format(backend.find("odd", "input"), "input")


def test_backend_and_streams_satisfy_protocols(backend: PortAudioBackend) -> None:
    mic, spk = backend.default_device("input"), backend.default_device("output")
    fmt = backend.native_format(mic, "input")
    assert isinstance(backend, DeviceBackend)
    assert isinstance(backend.open_source(mic, fmt, ChannelMap.uniform(2)), AudioSource)
    assert isinstance(backend.open_sink(spk, backend.native_format(spk, "output")), AudioSink)
    assert backend.portaudio_version == "PortAudio fake"


def test_open_checks_device_and_channels(backend: PortAudioBackend) -> None:
    mic = backend.default_device("input")
    with pytest.raises(FormatError, match="2 input channels; 4"):
        backend.open_source(mic, AudioFormat(48000, 4, "float32"), ChannelMap.uniform(4))
    with pytest.raises(FormatError, match="Channel map"):
        backend.open_source(mic, AudioFormat(48000, 2, "float32"), ChannelMap.uniform(1))
    stranger = DeviceInfo("file:in:a", "a", "file", None, 1, 0, 16000.0)
    with pytest.raises(DeviceNotFoundError, match="not a PortAudio device"):
        backend.open_sink(stranger, AudioFormat(16000, 1, "float32"))


def test_missing_sounddevice_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # makes the import fail
    with pytest.raises(MissingDependencyError, match=r"ral-audio\[audio\]"):
        load_sounddevice()


def test_missing_portaudio_library_gives_apt_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(DeviceError, match="libportaudio2"):
        load_sounddevice()


# Source


async def started_source(backend: PortAudioBackend, sd: FakeSd) -> tuple[Any, FakeStream]:
    mic = backend.default_device("input")
    source = backend.open_source(mic, backend.native_format(mic, "input"), ChannelMap.uniform(2))
    await source.start()
    return source, sd.streams[-1]


async def test_source_opens_native_stream_with_host_block_size(
    backend: PortAudioBackend, sd: FakeSd
) -> None:
    source, stream = await started_source(backend, sd)
    assert stream.state == "started"
    assert {k: stream.kwargs[k] for k in ("device", "samplerate", "channels", "dtype")} == {
        "device": 2,
        "samplerate": 48000,
        "channels": 2,
        "dtype": "float32",
    }
    assert stream.kwargs["blocksize"] == 0
    await source.stop()
    assert stream.state == "closed"


async def test_source_frames_carry_adc_time(backend: PortAudioBackend, sd: FakeSd) -> None:
    source, stream = await started_source(backend, sd)
    block = np.ones((256, 2), dtype=np.float32)
    first_adc = pa_now() - 0.004  # captured 4 ms before the first callback
    for k in range(10):  # 2560 samples: five 480-sample frames
        adc = first_adc + k * 256 / 48000
        stream.callback(block, 256, time_info(pa_now(), adc=adc), flags())
    frames = []
    async for frame in source.frames():
        frames.append(frame)
        if len(frames) == 5:
            break
    await source.stop()

    assert [f.seq for f in frames] == [0, 1, 2, 3, 4]
    assert [f.t_ns - frames[0].t_ns for f in frames] == [k * 10 * MS for k in range(5)]
    expected = round(first_adc * 1e9)
    assert abs(frames[0].t_ns - expected) < MS  # within the callback's own delay
    assert frames[0].clock_id == "portaudio:wasapi:2"


@pytest.mark.parametrize("adc_kind", ["zero", "equal_to_current"])
async def test_source_falls_back_to_current_minus_latency(
    backend: PortAudioBackend, sd: FakeSd, adc_kind: str
) -> None:
    source, stream = await started_source(backend, sd)
    now = pa_now()
    adc = 0.0 if adc_kind == "zero" else now
    stream.callback(np.ones((480, 2), np.float32), 480, time_info(now, adc=adc), flags())
    frame = source.framer.read_frame()
    await source.stop()
    assert frame is not None
    assert abs(frame.t_ns - round((now - stream.latency) * 1e9)) < MS


async def test_source_counts_input_overflows(backend: PortAudioBackend, sd: FakeSd) -> None:
    source, stream = await started_source(backend, sd)
    block = np.zeros((10, 2), np.float32)
    stream.callback(block, 10, time_info(pa_now(), adc=1.0), flags(input_overflow=True))
    stream.callback(block, 10, time_info(pa_now(), adc=1.0), flags())
    await source.stop()
    assert source.input_overflows == 1


async def test_open_failure_explains_native_rate(backend: PortAudioBackend, sd: FakeSd) -> None:
    sd.fail_open = True
    mic = backend.default_device("input")
    source = backend.open_source(mic, AudioFormat(16000, 2, "float32"), ChannelMap.uniform(2))
    with pytest.raises(DeviceError, match=r"native rate, here 48000 Hz"):
        await source.start()


async def test_frames_before_start_fail(backend: PortAudioBackend) -> None:
    mic = backend.default_device("input")
    source = backend.open_source(mic, backend.native_format(mic, "input"), ChannelMap.uniform(2))
    with pytest.raises(DeviceError, match="start"):
        await anext(source.frames())


# Sink


async def test_sink_renders_written_frames_then_silence_to_the_hooks(
    backend: PortAudioBackend, sd: FakeSd
) -> None:
    spk = backend.default_device("output")
    fmt = AudioFormat(48000, 2, "float32")
    sink = backend.open_sink(spk, fmt)
    seen: list[tuple[np.ndarray, int]] = []
    sink.on_rendered(lambda samples, t: seen.append((samples.copy(), t)))
    await sink.start()
    stream = sd.streams[-1]
    tone = np.full((480, 2), 0.5, dtype=np.float32)
    sink.write(AudioFrame(tone, fmt, ChannelMap.uniform(2), 0, 0, "agent"))

    now = pa_now()
    for k in range(3):  # three 300-sample device blocks: 480 samples of tone, then silence
        out = np.full((300, 2), 9.0, dtype=np.float32)
        t = now + k * 300 / 48000
        stream.callback(out, 300, time_info(t, dac=t + 0.010), flags())
    await sink.stop()

    played = np.concatenate([s for s, _ in seen])
    assert np.array_equal(played[:480], tone)
    assert not played[480:].any()
    dac_times = [t for _, t in seen]
    assert abs(dac_times[0] - round((now + 0.010) * 1e9)) < MS
    assert all(b > a for a, b in itertools.pairwise(dac_times))
    assert sink.output_latency_ns() == 20 * MS


async def test_sink_dac_fallback_and_underflow_count(backend: PortAudioBackend, sd: FakeSd) -> None:
    spk = backend.default_device("output")
    sink = backend.open_sink(spk, AudioFormat(48000, 2, "float32"))
    times: list[int] = []
    sink.on_rendered(lambda samples, t: times.append(t))
    await sink.start()
    stream = sd.streams[-1]
    now = pa_now()
    out = np.zeros((480, 2), np.float32)
    stream.callback(out, 480, time_info(now, dac=now), flags(output_underflow=True))
    await sink.stop()
    assert abs(times[0] - round((now + stream.latency) * 1e9)) < MS
    assert sink.output_underflows == 1


async def test_sink_write_checks_format_and_drops_when_full(
    backend: PortAudioBackend, sd: FakeSd
) -> None:
    spk = backend.default_device("output")
    fmt = AudioFormat(48000, 2, "float32")
    sink = PortAudioBackend(sd, capacity_s=0.02).open_sink(spk, fmt)  # type: ignore[arg-type]
    frame = AudioFrame(np.zeros((480, 2), np.float32), fmt, ChannelMap.uniform(2), 0, 0, "a")
    for _ in range(3):
        sink.write(frame)
    assert sink.buffer.overruns == 1
    other = AudioFormat(16000, 2, "float32")
    with pytest.raises(FormatError, match="does not match"):
        sink.write(AudioFrame(np.zeros((160, 2), np.float32), other, frame.channel_map, 0, 0, "a"))


# Real hardware: uv run pytest -m hardware -s tests/devices/test_portaudio.py


def _hw_backend() -> PortAudioBackend:
    return PortAudioBackend()


def _hw_device(backend: PortAudioBackend, direction: str) -> DeviceInfo:
    name = os.environ.get("RAL_TEST_INPUT" if direction == "input" else "RAL_TEST_OUTPUT")
    if name:
        return backend.find(name, direction)  # type: ignore[arg-type]
    return backend.default_device(direction)  # type: ignore[arg-type]


@pytest.mark.hardware
def test_hw_discovers_devices() -> None:
    backend = _hw_backend()
    devices = backend.discover()
    print(f"\n{backend.portaudio_version}: {len(devices)} devices")
    for api, group in group_by_host_api(devices).items():
        print(f"  {api}: {len(group)}")
    print(f"  default input:  {_hw_device(backend, 'input').name}")
    print(f"  default output: {_hw_device(backend, 'output').name}")
    assert devices


@pytest.mark.hardware
async def test_hw_records_two_seconds() -> None:
    backend = _hw_backend()
    mic = _hw_device(backend, "input")
    fmt = backend.native_format(mic, "input", channels=min(2, mic.max_input_channels))
    source = backend.open_source(mic, fmt, ChannelMap.uniform(fmt.channels))
    await source.start()
    frames: list[AudioFrame] = []
    async for frame in source.frames():
        frames.append(frame)
        if len(frames) == 200:
            break
    lag_ms = (now_ns() - frames[-1].t_ns) / MS
    await source.stop()

    data = np.concatenate([f.data for f in frames])
    rms_db = 20 * np.log10(np.sqrt(np.mean(data.astype(np.float64) ** 2)) + 1e-12)
    print(
        f"\n{mic.name} [{mic.host_api}] {fmt.sample_rate} Hz x{fmt.channels}: "
        f"{len(frames)} frames, level {rms_db:.1f} dBFS, last frame {lag_ms:.1f} ms old, "
        f"overflows {source.input_overflows}, discontinuities "
        f"{source.framer.discontinuities}, reanchors {source.framer.reanchors}, "
        f"clock offset {source.clock.offset_ns} ns"
    )
    assert [f.seq for f in frames] == list(range(frames[0].seq, frames[0].seq + 200))
    assert all(b.t_ns - a.t_ns == 10 * MS for a, b in itertools.pairwise(frames))
    assert 0 < lag_ms < 200  # stamped at the ADC, delivered shortly after


@pytest.mark.hardware
async def test_hw_plays_tone_and_reports_every_block() -> None:
    backend = _hw_backend()
    spk = _hw_device(backend, "output")
    fmt = backend.native_format(spk, "output", channels=min(2, spk.max_output_channels))
    sink = backend.open_sink(spk, fmt)
    blocks: list[tuple[int, int]] = []
    sink.on_rendered(lambda samples, t: blocks.append((samples.shape[0], t)))
    await sink.start()
    n = fmt.frame_samples
    k = np.arange(n * 100) / fmt.sample_rate
    tone = (0.1 * np.sin(2 * np.pi * 440 * k)).astype(np.float32)  # -23 dBFS, 1 s
    cmap = ChannelMap.uniform(fmt.channels)
    for i in range(100):
        while sink.buffer.available > 4 * n:  # keep about 40 ms queued
            await asyncio.sleep(0.002)
        chunk = np.repeat(tone[i * n : (i + 1) * n, None], fmt.channels, axis=1)
        sink.write(AudioFrame(chunk, fmt, cmap, now_ns(), i, "test"))
    await asyncio.sleep(0.2)
    await sink.stop()

    samples = sum(size for size, _ in blocks)
    span_ms = (blocks[-1][1] - blocks[0][1]) / MS
    print(
        f"\n{spk.name} [{spk.host_api}] {fmt.sample_rate} Hz x{fmt.channels}: "
        f"{len(blocks)} blocks, {samples} samples over {span_ms:.0f} ms of DAC time, "
        f"latency {sink.output_latency_ns() / MS:.1f} ms, underflows "
        f"{sink.output_underflows}, dropped {sink.buffer.overruns}"
    )
    assert samples >= 100 * n
    assert all(b[1] > a[1] for a, b in itertools.pairwise(blocks))
    assert sink.buffer.overruns == 0
