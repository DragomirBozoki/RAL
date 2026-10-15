"""Core data types: audio formats, channel maps, frames and device descriptions.

See ``docs/DESIGN.md`` section 4. These types are validated on construction, so a
malformed frame fails where it is made, not three stages later inside the echo canceller.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, get_args

import numpy as np
import numpy.typing as npt

from ral.core.errors import FormatError

__all__ = [
    "AudioFormat",
    "AudioFrame",
    "ChannelMap",
    "ChannelRole",
    "DeviceInfo",
    "SampleDtype",
]

SampleDtype = Literal["int16", "float32"]
_SAMPLE_DTYPES: tuple[str, ...] = get_args(SampleDtype)

Position = tuple[float, float, float]


class ChannelRole(Enum):
    """What a channel carries."""

    MIC = "mic"
    REFERENCE = "reference"  # loopback of what the speaker plays, on the same clock as the mics
    PROCESSED = "processed"  # already echo-cancelled by hardware
    UNUSED = "unused"


@dataclass(frozen=True)
class AudioFormat:
    """Sample rate, channel count, sample type and frame length of a stream.

    The sample rate must give a whole number of samples per frame. With the default
    10 ms frames that means a rate divisible by 100: 16000, 44100 and 48000 work,
    22050 does not.

    Raises:
        FormatError: If any field is out of range.
    """

    sample_rate: int  # 16_000 inside the engine by default
    channels: int
    dtype: SampleDtype
    frame_ms: int = 10  # WebRTC APM requires exactly 10 ms frames

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise FormatError(f"sample_rate must be positive, got {self.sample_rate}.")
        if self.channels <= 0:
            raise FormatError(f"channels must be at least 1, got {self.channels}.")
        if self.dtype not in _SAMPLE_DTYPES:
            raise FormatError(
                f"dtype must be one of {', '.join(_SAMPLE_DTYPES)}, got {self.dtype!r}."
            )
        if self.frame_ms <= 0:
            raise FormatError(f"frame_ms must be positive, got {self.frame_ms}.")
        if (self.sample_rate * self.frame_ms) % 1000 != 0:
            raise FormatError(
                f"{self.sample_rate} Hz does not give a whole number of samples per "
                f"{self.frame_ms} ms frame. Use a rate divisible by {1000 // self.frame_ms} "
                f"(for example 16000 or 48000), or resample first."
            )

    @property
    def frame_samples(self) -> int:
        """Samples per channel in one frame."""
        return self.sample_rate * self.frame_ms // 1000

    @property
    def frame_ns(self) -> int:
        """Duration of one frame in nanoseconds."""
        return self.frame_ms * 1_000_000

    @property
    def numpy_dtype(self) -> np.dtype[Any]:
        """The numpy dtype that frame data of this format must have."""
        return np.dtype(self.dtype)


@dataclass(frozen=True)
class ChannelMap:
    """The role of each channel, in channel order, plus optional mic geometry.

    Raises:
        FormatError: If there are no roles, a role is not a :class:`ChannelRole`, or the
            positions do not match the roles.
    """

    roles: tuple[ChannelRole, ...]
    positions_m: tuple[Position | None, ...] | None = None  # optional mic geometry

    def __post_init__(self) -> None:
        if not self.roles:
            raise FormatError("A channel map needs at least one channel role.")
        for role in self.roles:
            if not isinstance(role, ChannelRole):
                raise FormatError(
                    f"Channel roles must be ChannelRole members, got {role!r}. "
                    f"Use ChannelMap.from_names() to build a map from strings."
                )
        if self.positions_m is not None:
            if len(self.positions_m) != len(self.roles):
                raise FormatError(
                    f"positions_m has {len(self.positions_m)} entries but there are "
                    f"{len(self.roles)} channels; give one position (or None) per channel."
                )
            for pos in self.positions_m:
                if pos is not None and len(pos) != 3:
                    raise FormatError(f"A mic position needs x, y and z in metres, got {pos!r}.")

    @classmethod
    def from_names(cls, names: Iterable[str]) -> ChannelMap:
        """Build a map from role names such as ``["mic", "mic", "reference"]``.

        Raises:
            FormatError: If a name is not a known role.
        """
        roles = []
        for name in names:
            try:
                roles.append(ChannelRole(name))
            except ValueError:
                valid = ", ".join(r.value for r in ChannelRole)
                raise FormatError(
                    f"Unknown channel role {name!r}; valid roles are {valid}."
                ) from None
        return cls(tuple(roles))

    @classmethod
    def uniform(cls, channels: int, role: ChannelRole = ChannelRole.MIC) -> ChannelMap:
        """A map where every one of ``channels`` channels has the same role."""
        if channels <= 0:
            raise FormatError(f"channels must be at least 1, got {channels}.")
        return cls((role,) * channels)

    @property
    def channels(self) -> int:
        """Number of channels in the map."""
        return len(self.roles)

    def indices(self, role: ChannelRole) -> tuple[int, ...]:
        """Indices of the channels that have ``role``, in channel order."""
        return tuple(i for i, r in enumerate(self.roles) if r is role)


@dataclass
class AudioFrame:
    """One frame of audio with its timing and channel roles.

    Before the Normalizer, frames keep the device's native rate and dtype; after it,
    they are float32 at the engine rate.

    Raises:
        FormatError: If ``data`` does not match ``fmt`` or ``channel_map``.
    """

    data: npt.NDArray[Any]  # shape (fmt.frame_samples, fmt.channels)
    fmt: AudioFormat
    channel_map: ChannelMap
    t_ns: int  # monotonic estimate: first sample at the ADC (capture) or DAC (playback)
    seq: int  # gap detection
    clock_id: str  # which device clock produced it

    def __post_init__(self) -> None:
        # Cheap checks only: this runs 100 times a second per stream.
        expected = (self.fmt.frame_samples, self.fmt.channels)
        if self.data.shape != expected:
            raise FormatError(
                f"Frame data has shape {self.data.shape}, expected {expected} "
                f"(samples, channels) for {self.fmt}."
            )
        if self.data.dtype != self.fmt.numpy_dtype:
            raise FormatError(
                f"Frame data has dtype {self.data.dtype}, but the format says {self.fmt.dtype}."
            )
        if self.channel_map.channels != self.fmt.channels:
            raise FormatError(
                f"Channel map has {self.channel_map.channels} roles but the format has "
                f"{self.fmt.channels} channels."
            )
        if self.seq < 0:
            raise FormatError(f"seq must not be negative, got {self.seq}.")


@dataclass(frozen=True)
class DeviceInfo:
    """A device as a backend reports it."""

    id: str  # stable within a backend, e.g. "portaudio:wasapi:3", "ros2:/hal/audio/capture"
    name: str
    backend: str
    host_api: str | None
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float | None
    # Left out of the hash so DeviceInfo can be a dict key or set member despite the dict.
    extra: dict[str, str] = field(default_factory=dict, hash=False)
