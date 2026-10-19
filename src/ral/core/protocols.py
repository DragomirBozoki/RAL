"""Device interfaces: sources, sinks and the backends that open them.

See ``docs/DESIGN.md`` section 4. These are ``typing.Protocol``s, so backends in plugin
packages implement them without inheriting from RAL classes. They are runtime-checkable,
but ``isinstance`` only checks that the members exist, not their signatures.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol, runtime_checkable

import numpy.typing as npt

from ral.core.types import AudioFormat, AudioFrame, ChannelMap, DeviceInfo

__all__ = ["AudioSink", "AudioSource", "DeviceBackend", "ReferenceProvider", "RenderHook"]

RenderHook = Callable[[npt.NDArray[Any], int], None]
"""``hook(samples, t_dac_ns)``: a block handed to the device and when it plays."""


@runtime_checkable
class AudioSource(Protocol):
    """A stream of captured frames in the device's native rate and dtype."""

    fmt: AudioFormat
    channel_map: ChannelMap

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def frames(self) -> AsyncIterator[AudioFrame]:
        """Frames in capture order; ends when the source stops or runs out."""
        ...


@runtime_checkable
class AudioSink(Protocol):
    """A playback device.

    Every block handed to the device, silence included, goes to the render hooks with its
    estimated DAC time. The ``ReferenceProvider`` subscribes there, which is how the echo
    canceller sees everything RAL plays (CLAUDE.md, reference invariant).
    """

    fmt: AudioFormat

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def write(self, frame: AudioFrame) -> None:
        """Enqueue a frame for playback; never blocks."""
        ...

    def output_latency_ns(self) -> int:
        """Time from handing a block to the device until it plays."""
        ...

    def on_rendered(self, hook: RenderHook) -> None:
        """Register a hook that runs on the output thread for every rendered block.

        Hooks must be real-time safe: copy the samples and return.
        """
        ...


@runtime_checkable
class DeviceBackend(Protocol):
    """Discovers devices of one kind and opens sources and sinks on them."""

    name: str  # "portaudio", "ros2_topic", "file", ...

    def discover(self) -> list[DeviceInfo]: ...

    def open_source(
        self, device: DeviceInfo, fmt: AudioFormat, channel_map: ChannelMap
    ) -> AudioSource: ...

    def open_sink(self, device: DeviceInfo, fmt: AudioFormat) -> AudioSink: ...


@runtime_checkable
class ReferenceProvider(Protocol):
    """What was played, as 10 ms frames on the engine clock and rate.

    ``push`` runs on the output thread (it is a render hook) and must only copy.
    ``frames_until`` runs on the DSP thread.
    """

    def push(self, samples: npt.NDArray[Any], t_dac_ns: int) -> None: ...

    def frames_until(self, t_ns: int) -> list[AudioFrame]:
        """Every buffered frame whose DAC time is earlier than ``t_ns``, oldest first."""
        ...
