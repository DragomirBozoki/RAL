"""File backend: WAV files as capture and playback devices.

Uses only the stdlib ``wave`` module with 16-bit PCM, so tests, replays and benchmarks
need no audio hardware and no extra dependency. Each configured input file is a capture
device and each output file a playback device.

Timing. Both ends run either as fast as possible (``realtime=False``, the default, for
tests and offline replay) or paced by the RAL clock (``realtime=True``), which
behaves like a real device: a capture frame is delivered once its last sample would have
been captured, and the sink renders one frame every 10 ms, rendering silence when nothing
was written. Pacing uses absolute deadlines (``t_start + k * frame``), so scheduling
jitter never accumulates into drift. On Windows the event loop wakes late by several
milliseconds (``spikes/clock_resolution.py``), so single frames jitter while the average
rate stays exact.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import wave
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ral.core.clock import now_ns
from ral.core.convert import float32_to_int16, int16_to_float32
from ral.core.errors import DeviceError, DeviceNotFoundError, FormatError
from ral.core.protocols import RenderHook
from ral.core.ringbuffer import RingBuffer
from ral.core.types import AudioFormat, AudioFrame, ChannelMap, DeviceInfo

__all__ = ["FileBackend", "FileSink", "FileSource"]

_SAMPLE_WIDTH = 2  # bytes; 16-bit PCM only
_PCM_DTYPE = np.dtype("<i2")  # WAV samples are little-endian on every platform
# A WAV file can hold up to 65535 channels; this is a practical cap for discovery.
_MAX_OUTPUT_CHANNELS = 64


def _read_wav_format(path: Path) -> tuple[int, int]:
    """Sample rate and channel count of a 16-bit PCM WAV file."""
    try:
        with wave.open(str(path), "rb") as wav:
            width, rate, channels = wav.getsampwidth(), wav.getframerate(), wav.getnchannels()
    except (wave.Error, EOFError) as exc:
        raise FormatError(f"{path} is not a readable PCM WAV file: {exc}") from exc
    except OSError as exc:
        raise DeviceError(f"Cannot open {path}: {exc}") from exc
    if width != _SAMPLE_WIDTH:
        raise FormatError(
            f"{path} has {8 * width}-bit samples; the file backend reads 16-bit PCM only. "
            f"Convert it first, for example with: sox in.wav -b 16 out.wav"
        )
    return rate, channels


# asyncio runs every timer due within one step of its clock (time.monotonic: 15.6 ms on
# Windows before Python 3.13), so a shorter sleep returns at once.
_LOOP_STEP_S = time.get_clock_info("monotonic").resolution


async def _sleep_until(deadline_ns: int) -> None:
    # Timers may fire up to one clock step early, so check our clock again. Sleeping at
    # least one step instead of retrying at once keeps the loop from spinning, which held
    # the GIL and starved the DSP thread (spikes/dsp_gil_load.py); a frame is then late by
    # at most one step, like a sound card's block, and the next frames catch up.
    while (delay := deadline_ns - now_ns()) > 0:
        await asyncio.sleep(max(delay / 1e9, _LOOP_STEP_S))


class FileSource:
    """Reads a WAV file as 10 ms frames.

    Frames have the file's rate and channel count. With ``fmt.dtype == "int16"`` they
    carry the file's samples unchanged; with ``"float32"`` they are converted. A last
    partial frame is padded with zeros. ``t_ns`` of frame ``k`` is the start time plus
    ``k`` frames.
    """

    def __init__(
        self, path: Path, fmt: AudioFormat, channel_map: ChannelMap, clock_id: str, realtime: bool
    ) -> None:
        self.fmt = fmt
        self.channel_map = channel_map
        self._path = path
        self._clock_id = clock_id
        self._realtime = realtime
        self._wav: wave.Wave_read | None = None
        self._t_start = 0
        self._stopped = False

    async def start(self) -> None:
        """Open the file and set the stream's start time."""
        self._wav = wave.open(str(self._path), "rb")  # noqa: SIM115  closed in stop()
        self._t_start = now_ns()
        self._stopped = False

    async def stop(self) -> None:
        """Close the file; :meth:`frames` ends."""
        self._stopped = True
        if self._wav is not None:
            self._wav.close()
            self._wav = None

    async def frames(self) -> AsyncIterator[AudioFrame]:
        """Frames until the end of the file or :meth:`stop`."""
        if self._wav is None:
            raise DeviceError(f"FileSource for {self._path} is not started; call start() first.")
        fsamples, channels = self.fmt.frame_samples, self.fmt.channels
        seq = 0
        while not self._stopped and self._wav is not None:
            # Reads are a few hundred bytes from a local file; short enough for the loop.
            raw = self._wav.readframes(fsamples)
            if not raw:
                return
            pcm = np.frombuffer(raw, dtype=_PCM_DTYPE).reshape(-1, channels)
            block = np.zeros((fsamples, channels), dtype=np.int16)
            block[: pcm.shape[0]] = pcm
            t_ns = self._t_start + seq * self.fmt.frame_ns
            if self._realtime:
                # A device delivers a frame once its last sample has been captured.
                await _sleep_until(t_ns + self.fmt.frame_ns)
            else:
                await asyncio.sleep(0)  # let other tasks run between frames
            data = int16_to_float32(block) if self.fmt.dtype == "float32" else block
            yield AudioFrame(data, self.fmt, self.channel_map, t_ns, seq, self._clock_id)
            seq += 1


class FileSink:
    """Writes frames to a 16-bit PCM WAV file and runs the render hooks.

    With ``realtime=False`` each written frame is rendered at once and stamped as if
    playback had been continuous since :meth:`start`. With ``realtime=True`` a task
    renders one frame per 10 ms from a buffer (``drop_newest`` when full) and renders
    silence on underrun, so the file and the hooks see a continuous timeline, as a real
    device would. Hooks receive samples in ``fmt.dtype``; the file always gets int16.
    """

    def __init__(self, path: Path, fmt: AudioFormat, realtime: bool, capacity_s: float) -> None:
        self.fmt = fmt
        self._path = path
        self._realtime = realtime
        self._hooks: list[RenderHook] = []
        self._wav: wave.Wave_write | None = None
        self._task: asyncio.Task[None] | None = None
        self._t_start = 0
        self._rendered = 0  # samples per channel handed to the "device" so far
        capacity = max(fmt.frame_samples, round(capacity_s * fmt.sample_rate))
        self.buffer = RingBuffer(fmt.channels, fmt.dtype, capacity, overrun="drop_newest")
        """Queue between :meth:`write` and the render task (realtime mode)."""

    async def start(self) -> None:
        """Create the file and, in realtime mode, start rendering."""
        wav = wave.open(str(self._path), "wb")  # noqa: SIM115  closed in stop()
        wav.setnchannels(self.fmt.channels)
        wav.setsampwidth(_SAMPLE_WIDTH)
        wav.setframerate(self.fmt.sample_rate)
        self._wav = wav
        self._t_start = now_ns()
        self._rendered = 0
        if self._realtime:
            self._task = asyncio.create_task(self._render_loop())

    async def stop(self) -> None:
        """Stop rendering and close the file. Frames still queued are discarded."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._wav is not None:
            self._wav.close()
            self._wav = None

    def write(self, frame: AudioFrame) -> None:
        """Enqueue a frame; in realtime mode a full buffer drops it and counts an overrun."""
        if self._wav is None:
            raise DeviceError(f"FileSink for {self._path} is not started; call start() first.")
        if frame.fmt != self.fmt:
            raise FormatError(f"Frame format {frame.fmt} does not match the sink's {self.fmt}.")
        if self._realtime:
            self.buffer.push(frame.data, frame.t_ns)
        else:
            self._render(frame.data)

    def output_latency_ns(self) -> int:
        """A file plays the moment it is written."""
        return 0

    def on_rendered(self, hook: RenderHook) -> None:
        """Register a hook; it runs in the task or caller that renders."""
        self._hooks.append(hook)

    async def _render_loop(self) -> None:
        block = np.empty((self.fmt.frame_samples, self.fmt.channels), dtype=self.fmt.numpy_dtype)
        k = 0
        while True:
            await _sleep_until(self._t_start + k * self.fmt.frame_ns)
            self.buffer.read_available(block)  # pads with silence on underrun
            self._render(block)
            k += 1

    def _render(self, samples: npt.NDArray[Any]) -> None:
        assert self._wav is not None
        t_dac_ns = self._t_start + self._rendered * 1_000_000_000 // self.fmt.sample_rate
        pcm = float32_to_int16(samples) if self.fmt.dtype == "float32" else samples
        self._wav.writeframes(pcm.astype(_PCM_DTYPE).tobytes())
        self._rendered += samples.shape[0]
        for hook in self._hooks:
            hook(samples, t_dac_ns)


class FileBackend:
    """WAV files as devices.

    Args:
        sources: Capture devices, ``{name: path}``; each file must exist.
        sinks: Playback devices, ``{name: path}``; each file is created on start.
        realtime: Pace sources and sinks by the clock instead of running flat out.
        sink_capacity_s: Buffer length of realtime sinks.
    """

    name = "file"

    def __init__(
        self,
        sources: Mapping[str, str | Path] | None = None,
        sinks: Mapping[str, str | Path] | None = None,
        *,
        realtime: bool = False,
        sink_capacity_s: float = 2.0,
    ) -> None:
        self._sources = {name: Path(p) for name, p in (sources or {}).items()}
        self._sinks = {name: Path(p) for name, p in (sinks or {}).items()}
        self._realtime = realtime
        self._sink_capacity_s = sink_capacity_s

    def discover(self) -> list[DeviceInfo]:
        """One device per configured file; inputs report the file's format."""
        devices = []
        for name, path in self._sources.items():
            rate, channels = _read_wav_format(path)
            devices.append(
                DeviceInfo(
                    id=f"file:in:{name}",
                    name=name,
                    backend=self.name,
                    host_api=None,
                    max_input_channels=channels,
                    max_output_channels=0,
                    default_sample_rate=float(rate),
                    extra={"path": str(path)},
                )
            )
        for name, path in self._sinks.items():
            devices.append(
                DeviceInfo(
                    id=f"file:out:{name}",
                    name=name,
                    backend=self.name,
                    host_api=None,
                    max_input_channels=0,
                    max_output_channels=_MAX_OUTPUT_CHANNELS,
                    default_sample_rate=None,
                    extra={"path": str(path)},
                )
            )
        return devices

    def open_source(
        self, device: DeviceInfo, fmt: AudioFormat, channel_map: ChannelMap
    ) -> FileSource:
        """Open a capture file. ``fmt`` must match the file's rate and channels.

        Raises:
            DeviceNotFoundError: If ``device`` is not one of this backend's inputs.
            FormatError: If ``fmt`` or ``channel_map`` do not match the file.
        """
        path = self._lookup(device, self._sources, "file:in:")
        rate, channels = _read_wav_format(path)
        if (fmt.sample_rate, fmt.channels) != (rate, channels):
            raise FormatError(
                f"{path} is {rate} Hz with {channels} channels, but {fmt.sample_rate} Hz with "
                f"{fmt.channels} channels was requested. Use the file's format; the "
                f"Normalizer resamples later."
            )
        if channel_map.channels != channels:
            raise FormatError(
                f"Channel map has {channel_map.channels} roles but {path} has {channels} channels."
            )
        return FileSource(path, fmt, channel_map, device.id, self._realtime)

    def open_sink(self, device: DeviceInfo, fmt: AudioFormat) -> FileSink:
        """Open an output file that :meth:`FileSink.start` creates.

        Raises:
            DeviceNotFoundError: If ``device`` is not one of this backend's outputs.
        """
        path = self._lookup(device, self._sinks, "file:out:")
        return FileSink(path, fmt, self._realtime, self._sink_capacity_s)

    def _lookup(self, device: DeviceInfo, table: Mapping[str, Path], prefix: str) -> Path:
        name = device.id.removeprefix(prefix)
        if not device.id.startswith(prefix) or name not in table:
            known = ", ".join(prefix + n for n in table) or "none"
            raise DeviceNotFoundError(
                f"The file backend has no device {device.id!r}; configured: {known}."
            )
        return table[name]
