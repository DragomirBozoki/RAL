"""Core types, errors and conversions shared by every RAL module."""

from ral.core.convert import float32_to_int16, int16_to_float32
from ral.core.errors import (
    DeviceError,
    DeviceNotFoundError,
    FormatError,
    MissingDependencyError,
    RalError,
)
from ral.core.framer import Framer
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
    "Anchor",
    "AudioFormat",
    "AudioFrame",
    "ChannelMap",
    "ChannelRole",
    "DeviceError",
    "DeviceInfo",
    "DeviceNotFoundError",
    "FormatError",
    "Framer",
    "MissingDependencyError",
    "OverrunPolicy",
    "RalError",
    "ReadResult",
    "RingBuffer",
    "SampleDtype",
    "float32_to_int16",
    "int16_to_float32",
]
