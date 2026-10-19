"""PortAudio backend through ``sounddevice`` (the ``audio`` extra).

Discovery lists every PortAudio device, preferred host API first: WASAPI on Windows,
ALSA on Linux (PipeWire and PulseAudio appear there as ALSA devices such as "pipewire",
"pulse" and "default"), Core Audio on macOS. Windows lists one physical device under
several host APIs; ``group_by_host_api`` keeps them apart for ``ral doctor``.

Devices open at their native rate (WASAPI shared mode cannot resample unless asked to,
and RAL resamples in its Normalizer instead), with ``blocksize=0`` so the host picks the
block size with the lowest latency; the Framer cuts any block size into 10 ms frames.

Callbacks only copy into a ring buffer and do integer arithmetic (CLAUDE.md rule 4).

Timestamps. PortAudio's callback time info gives ``currentTime``, ``inputBufferAdcTime``
and ``outputBufferDacTime`` in seconds on PortAudio's own clock. ``spikes/portaudio_time.py``
found that clock to be QueryPerformanceCounter on Windows, the same as the RAL clock, but
that is not guaranteed elsewhere, so :class:`PaClockMapper` measures the offset in every
callback. Some host APIs report 0 for the ADC or DAC time, and DirectSound reports a DAC
time equal to ``currentTime``; both fall back to ``currentTime`` minus (capture) or plus
(playback) the stream's reported latency (DESIGN section 5). PortAudio API reference:
https://files.portaudio.com/docs/v19-doxydocs/structPaStreamCallbackTimeInfo.html
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Iterable
from types import ModuleType
from typing import Any, Literal

import numpy.typing as npt

from ral.core.clock import now_ns
from ral.core.errors import DeviceError, DeviceNotFoundError, FormatError, MissingDependencyError
from ral.core.framer import Framer
from ral.core.protocols import RenderHook
from ral.core.ringbuffer import RingBuffer
from ral.core.types import AudioFormat, AudioFrame, ChannelMap, DeviceInfo, SampleDtype

__all__ = [
    "PaClockMapper",
    "PortAudioBackend",
    "PortAudioSink",
    "PortAudioSource",
    "group_by_host_api",
    "host_api_slug",
    "load_sounddevice",
    "preferred_host_apis",
]

Direction = Literal["input", "output"]

_PREFERRED_HOST_APIS: dict[str, tuple[str, ...]] = {
    "win32": ("Windows WASAPI",),
    "linux": ("ALSA", "JACK Audio Connection Kit"),
    "darwin": ("Core Audio",),
}


def load_sounddevice() -> ModuleType:
    """Import ``sounddevice``, or fail with an install hint.

    Raises:
        MissingDependencyError: If the ``audio`` extra is not installed.
        DeviceError: If the system PortAudio library cannot be loaded.
    """
    try:
        import sounddevice
    except ImportError as exc:
        raise MissingDependencyError("sounddevice", "audio", "The PortAudio backend") from exc
    except OSError as exc:
        # sounddevice raises OSError when it cannot find the PortAudio shared library.
        raise DeviceError(
            f"The PortAudio library could not be loaded ({exc}). On Debian and Ubuntu "
            f"install it with: sudo apt install libportaudio2"
        ) from exc
    module: ModuleType = sounddevice
    return module


def preferred_host_apis(platform: str | None = None) -> tuple[str, ...]:
    """Host API names RAL prefers on ``platform`` (default: this one), best first."""
    return _PREFERRED_HOST_APIS.get(platform or sys.platform, ())


def host_api_slug(name: str) -> str:
    """Short id for a host API name: ``"Windows WASAPI"`` becomes ``"wasapi"``."""
    return name.lower().removeprefix("windows ").strip().replace(" ", "-")


def group_by_host_api(devices: Iterable[DeviceInfo]) -> dict[str, list[DeviceInfo]]:
    """Devices keyed by host API name, in discovery order."""
    groups: dict[str, list[DeviceInfo]] = {}
    for dev in devices:
        groups.setdefault(dev.host_api or "", []).append(dev)
    return groups


class PaClockMapper:
    """Maps PortAudio stream time (float seconds) onto the RAL clock (integer ns).

    In each callback PortAudio reads its clock into ``currentTime`` shortly before the
    callback reads the RAL clock, so ``now_ns() - currentTime`` is the true offset plus a
    small, always positive delay. The lowest sample is the best estimate. To follow a
    clock that steps (a time base adjusted by NTP), the estimate drops to any lower sample
    at once and rises toward higher ones by 1/128 per callback, about a second at 100
    callbacks a second. Only integer arithmetic, so it is safe inside a callback.
    """

    def __init__(self, rise_shift: int = 7) -> None:
        self._rise_shift = rise_shift
        self._offset_ns: int | None = None

    @property
    def offset_ns(self) -> int | None:
        """RAL clock minus PortAudio clock, or None before the first observation."""
        return self._offset_ns

    def observe(self, pa_now_s: float, ral_now_ns: int | None = None) -> None:
        """Record one ``currentTime`` reading against the RAL clock (read now if None)."""
        ral = now_ns() if ral_now_ns is None else ral_now_ns
        sample = ral - round(pa_now_s * 1e9)
        if self._offset_ns is None or sample < self._offset_ns:
            self._offset_ns = sample
        else:
            self._offset_ns += (sample - self._offset_ns) >> self._rise_shift

    def to_ral_ns(self, pa_time_s: float) -> int:
        """Convert a PortAudio time to the RAL clock; call :meth:`observe` first."""
        assert self._offset_ns is not None, "observe() must run before to_ral_ns()"
        return round(pa_time_s * 1e9) + self._offset_ns


def _open_error(exc: Exception, kind: str, device: DeviceInfo, fmt: AudioFormat) -> DeviceError:
    return DeviceError(
        f"Could not open {kind} {device.name!r} ({device.id}) at {fmt.sample_rate} Hz with "
        f"{fmt.channels} channels: {exc}. Devices open at their native rate, here "
        f"{device.default_sample_rate:g} Hz; get a matching format from "
        f"PortAudioBackend.native_format(device)."
    )


class PortAudioSource:
    """Capture from a PortAudio input device as timestamped 10 ms frames.

    Args:
        sd: The ``sounddevice`` module (injected so tests can pass a fake).
        device: The input device.
        fmt: Native rate, the channels to open and the sample dtype.
        channel_map: Roles of the opened channels.
        capacity_s: Ring buffer length.
        poll_s: How often :meth:`frames` checks for new frames.
    """

    def __init__(
        self,
        sd: ModuleType,
        device: DeviceInfo,
        fmt: AudioFormat,
        channel_map: ChannelMap,
        *,
        capacity_s: float = 2.0,
        poll_s: float = 0.005,
    ) -> None:
        self.fmt = fmt
        self.channel_map = channel_map
        self.framer = Framer(fmt, channel_map, device.id, capacity_s=capacity_s)
        self.clock = PaClockMapper()
        self.input_overflows = 0
        """Callbacks in which PortAudio reported lost input."""
        self._sd = sd
        self._device = device
        self._poll_s = poll_s
        self._latency_s = 0.0
        self._stream: Any = None

    async def start(self) -> None:
        """Open and start the input stream.

        Raises:
            DeviceError: If PortAudio cannot open the device in this format.
        """
        try:
            stream = self._sd.InputStream(
                device=int(self._device.extra["index"]),
                samplerate=self.fmt.sample_rate,
                channels=self.fmt.channels,
                dtype=self.fmt.dtype,
                blocksize=0,
                latency="low",
                callback=self._callback,
            )
            self._latency_s = float(stream.latency)
            stream.start()
        except self._sd.PortAudioError as exc:
            raise _open_error(exc, "input", self._device, self.fmt) from exc
        self._stream = stream

    async def stop(self) -> None:
        """Stop and close the stream; :meth:`frames` ends."""
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    async def frames(self) -> AsyncIterator[AudioFrame]:
        """Frames as they arrive, until :meth:`stop`."""
        if self._stream is None:
            raise DeviceError(f"{self._device.id} is not started; call start() first.")
        while self._stream is not None:
            frame = self.framer.read_frame()
            if frame is None:
                await asyncio.sleep(self._poll_s)
            else:
                yield frame

    def _callback(self, indata: npt.NDArray[Any], frames: int, time: Any, status: Any) -> None:
        # PortAudio's thread: integer maths and one copy into the ring buffer, nothing else.
        if status.input_overflow:
            self.input_overflows += 1
        current = time.currentTime
        self.clock.observe(current)
        adc = time.inputBufferAdcTime
        if adc == 0.0 or adc == current:
            adc = current - self._latency_s
        self.framer.push(indata, self.clock.to_ral_ns(adc))


class PortAudioSink:
    """Play frames on a PortAudio output device and report every rendered block.

    :meth:`write` enqueues into a ``drop_newest`` ring buffer. The output callback takes
    what is there, pads with silence, and passes the block, silence included, to every
    render hook with its estimated DAC time. Hooks run on PortAudio's thread and receive
    PortAudio's own buffer, so they must copy it and return at once.

    TODO(M1): ``buffer.underruns`` also counts idle silence; the playback service knows
    whether more audio is due and should count real underruns. ``output_underflows`` is
    the device-level xrun count.
    """

    def __init__(
        self, sd: ModuleType, device: DeviceInfo, fmt: AudioFormat, *, capacity_s: float = 2.0
    ) -> None:
        self.fmt = fmt
        capacity = max(fmt.frame_samples, round(capacity_s * fmt.sample_rate))
        self.buffer = RingBuffer(fmt.channels, fmt.dtype, capacity, overrun="drop_newest")
        self.clock = PaClockMapper()
        self.output_underflows = 0
        """Callbacks in which PortAudio reported an output underflow."""
        self._sd = sd
        self._device = device
        self._hooks: list[RenderHook] = []
        self._latency_s = 0.0
        self._stream: Any = None

    async def start(self) -> None:
        """Open and start the output stream.

        Raises:
            DeviceError: If PortAudio cannot open the device in this format.
        """
        try:
            stream = self._sd.OutputStream(
                device=int(self._device.extra["index"]),
                samplerate=self.fmt.sample_rate,
                channels=self.fmt.channels,
                dtype=self.fmt.dtype,
                blocksize=0,
                latency="low",
                callback=self._callback,
            )
            self._latency_s = float(stream.latency)
            stream.start()
        except self._sd.PortAudioError as exc:
            raise _open_error(exc, "output", self._device, self.fmt) from exc
        self._stream = stream

    async def stop(self) -> None:
        """Stop and close the stream. Frames still queued are discarded."""
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    def write(self, frame: AudioFrame) -> None:
        """Enqueue a frame; a full buffer drops it and counts an overrun. Never blocks."""
        if frame.fmt != self.fmt:
            raise FormatError(f"Frame format {frame.fmt} does not match the sink's {self.fmt}.")
        self.buffer.push(frame.data, frame.t_ns)

    def output_latency_ns(self) -> int:
        """The stream's reported output latency (0 before :meth:`start`)."""
        return round(self._latency_s * 1e9)

    def on_rendered(self, hook: RenderHook) -> None:
        """Register a hook that runs on PortAudio's thread for every block."""
        self._hooks.append(hook)

    def _callback(self, outdata: npt.NDArray[Any], frames: int, time: Any, status: Any) -> None:
        # PortAudio's thread: copy out of the ring buffer, stamp, hand to the hooks.
        if status.output_underflow:
            self.output_underflows += 1
        self.buffer.read_available(outdata)
        current = time.currentTime
        self.clock.observe(current)
        dac = time.outputBufferDacTime
        if dac == 0.0 or dac == current:
            dac = current + self._latency_s
        t_dac_ns = self.clock.to_ral_ns(dac)
        for hook in self._hooks:
            hook(outdata, t_dac_ns)


class PortAudioBackend:
    """Sound cards and USB devices through PortAudio.

    Args:
        sd: A ``sounddevice``-compatible module; by default the real one, imported lazily.
        capacity_s: Ring buffer length for sources and sinks.

    Raises:
        MissingDependencyError: If ``sounddevice`` is not installed.
        DeviceError: If the PortAudio library cannot be loaded.
    """

    name = "portaudio"

    def __init__(self, sd: ModuleType | None = None, *, capacity_s: float = 2.0) -> None:
        self._sd = sd if sd is not None else load_sounddevice()
        self._capacity_s = capacity_s

    @property
    def portaudio_version(self) -> str:
        """PortAudio's version string."""
        return str(self._sd.get_portaudio_version()[1])

    def discover(self) -> list[DeviceInfo]:
        """All devices, preferred host API first, then by PortAudio index."""
        apis = self._sd.query_hostapis()
        preferred = preferred_host_apis()
        devices = []
        for raw in self._sd.query_devices():
            api = apis[raw["hostapi"]]
            index = int(raw["index"])
            extra = {
                "index": str(index),
                "low_input_latency_ms": f"{raw['default_low_input_latency'] * 1e3:.1f}",
                "low_output_latency_ms": f"{raw['default_low_output_latency'] * 1e3:.1f}",
            }
            if index == api["default_input_device"]:
                extra["default_input"] = "true"
            if index == api["default_output_device"]:
                extra["default_output"] = "true"
            devices.append(
                DeviceInfo(
                    id=f"portaudio:{host_api_slug(api['name'])}:{index}",
                    name=str(raw["name"]),
                    backend=self.name,
                    host_api=str(api["name"]),
                    max_input_channels=int(raw["max_input_channels"]),
                    max_output_channels=int(raw["max_output_channels"]),
                    default_sample_rate=float(raw["default_samplerate"]),
                    extra=extra,
                )
            )

        def rank(dev: DeviceInfo) -> tuple[int, int]:
            api = dev.host_api or ""
            pos = preferred.index(api) if api in preferred else len(preferred)
            return pos, int(dev.extra["index"])

        return sorted(devices, key=rank)

    def default_device(self, direction: Direction) -> DeviceInfo:
        """The default input or output device of the most preferred host API that has one.

        Raises:
            DeviceNotFoundError: If no host API has a default device for ``direction``.
        """
        key = f"default_{direction}"
        for dev in self.discover():  # already in preference order
            if dev.extra.get(key) == "true":
                return dev
        raise DeviceNotFoundError(
            f"PortAudio reports no default {direction} device. Check that a device is "
            f"connected and enabled, then run `ral doctor`."
        )

    def find(self, name: str, direction: Direction) -> DeviceInfo:
        """The first device, in preference order, whose name contains ``name``.

        Raises:
            DeviceNotFoundError: If no device with channels in ``direction`` matches.
        """
        for dev in self.discover():
            channels = dev.max_input_channels if direction == "input" else dev.max_output_channels
            if channels > 0 and name.lower() in dev.name.lower():
                return dev
        raise DeviceNotFoundError(
            f"No PortAudio {direction} device matches {name!r}. List devices with `ral doctor`."
        )

    def native_format(
        self,
        device: DeviceInfo,
        direction: Direction,
        dtype: SampleDtype = "float32",
        channels: int | None = None,
    ) -> AudioFormat:
        """The format to open ``device`` with: its default rate, all channels by default.

        Raises:
            FormatError: If the device's rate does not give whole 10 ms frames.
        """
        max_channels = (
            device.max_input_channels if direction == "input" else device.max_output_channels
        )
        if device.default_sample_rate is None or max_channels == 0:
            raise FormatError(f"{device.name!r} has no {direction} channels.")
        rate = int(device.default_sample_rate)
        try:
            return AudioFormat(rate, channels or max_channels, dtype)
        except FormatError as exc:
            raise FormatError(
                f"{device.name!r} runs at {rate} Hz, which RAL cannot frame in 10 ms. Set "
                f"the device to 16000, 44100 or 48000 Hz in the system sound settings."
            ) from exc

    def open_source(
        self, device: DeviceInfo, fmt: AudioFormat, channel_map: ChannelMap
    ) -> PortAudioSource:
        """Prepare a capture stream; :meth:`PortAudioSource.start` opens it.

        Raises:
            DeviceNotFoundError: If ``device`` is not a PortAudio device.
            FormatError: If the device lacks the channels or the map does not match.
        """
        self._check(device)
        if fmt.channels > device.max_input_channels:
            raise FormatError(
                f"{device.name!r} has {device.max_input_channels} input channels; "
                f"{fmt.channels} were requested."
            )
        if channel_map.channels != fmt.channels:
            raise FormatError(
                f"Channel map has {channel_map.channels} roles but {fmt.channels} channels "
                f"are opened."
            )
        return PortAudioSource(self._sd, device, fmt, channel_map, capacity_s=self._capacity_s)

    def open_sink(self, device: DeviceInfo, fmt: AudioFormat) -> PortAudioSink:
        """Prepare a playback stream; :meth:`PortAudioSink.start` opens it.

        Raises:
            DeviceNotFoundError: If ``device`` is not a PortAudio device.
            FormatError: If the device lacks the channels.
        """
        self._check(device)
        if fmt.channels > device.max_output_channels:
            raise FormatError(
                f"{device.name!r} has {device.max_output_channels} output channels; "
                f"{fmt.channels} were requested."
            )
        return PortAudioSink(self._sd, device, fmt, capacity_s=self._capacity_s)

    def _check(self, device: DeviceInfo) -> None:
        if device.backend != self.name or "index" not in device.extra:
            raise DeviceNotFoundError(
                f"{device.id!r} is not a PortAudio device; get one from discover()."
            )
