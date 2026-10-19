"""Core types, errors and conversions shared by every RAL module."""

from ral.core.bus import EventBus, Subscription
from ral.core.clock import CLOCK_NAME, now_ns
from ral.core.convert import float32_to_int16, int16_to_float32
from ral.core.errors import (
    DeviceError,
    DeviceNotFoundError,
    FormatError,
    MissingDependencyError,
    RalError,
)
from ral.core.events import Event, event_from_json, event_to_json
from ral.core.framer import Framer
from ral.core.protocols import (
    AudioSink,
    AudioSource,
    DeviceBackend,
    PlaybackHandle,
    PlaybackService,
    ReferenceProvider,
    RenderHook,
)
from ral.core.ringbuffer import Anchor, OverrunPolicy, ReadResult, RingBuffer
from ral.core.types import (
    AudioFormat,
    AudioFrame,
    ChannelMap,
    ChannelRole,
    DeviceInfo,
    SampleDtype,
)

__all__ = [
    "CLOCK_NAME",
    "Anchor",
    "AudioFormat",
    "AudioFrame",
    "AudioSink",
    "AudioSource",
    "ChannelMap",
    "ChannelRole",
    "DeviceBackend",
    "DeviceError",
    "DeviceInfo",
    "DeviceNotFoundError",
    "Event",
    "EventBus",
    "FormatError",
    "Framer",
    "MissingDependencyError",
    "OverrunPolicy",
    "PlaybackHandle",
    "PlaybackService",
    "RalError",
    "ReadResult",
    "ReferenceProvider",
    "RenderHook",
    "RingBuffer",
    "SampleDtype",
    "Subscription",
    "event_from_json",
    "event_to_json",
    "float32_to_int16",
    "int16_to_float32",
    "now_ns",
]
