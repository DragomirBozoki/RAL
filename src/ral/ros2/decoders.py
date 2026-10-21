"""Turning ROS 2 audio messages into PCM, and PCM into messages (DESIGN section 7).

The core knows the standard ``audio_common_msgs`` types: ``AudioDataStamped`` (a header
and the bytes) and ``AudioData`` (bytes only; timed on arrival). Messages carry no
format, so the format comes from the profile (``capture.format``), and RAL publishes an
``AudioInfo`` next to its own audio. Vendor message types are decoder plugins in separate
packages (entry point group ``ral.decoders``), so definitions under a vendor licence never
enter the core (CLAUDE.md, rule 9).
"""

from __future__ import annotations

import array
from importlib.metadata import entry_points
from typing import Any, Protocol

from ral.core.errors import ConfigError
from ral.ros2 import import_msgs

__all__ = ["AudioCommonDecoder", "Decoder", "info_message", "make_decoder", "stamped_message"]


class Decoder(Protocol):
    """Reads one message type."""

    msg_type: Any
    """The rclpy message class to subscribe with."""

    def decode(self, msg: Any) -> tuple[bytes, int | None]:
        """``(pcm_bytes, stamp_ros_ns or None)`` of one message."""
        ...


class AudioCommonDecoder:
    """``audio_common_msgs/AudioDataStamped`` (default) or ``AudioData``."""

    def __init__(self, stamped: bool = True) -> None:
        msgs = import_msgs("audio_common_msgs")
        self.stamped = stamped
        self.msg_type = msgs.AudioDataStamped if stamped else msgs.AudioData

    def decode(self, msg: Any) -> tuple[bytes, int | None]:
        if self.stamped:
            stamp = msg.header.stamp
            ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            return _bytes(msg.audio.data), (ns if ns > 0 else None)
        return _bytes(msg.data), None


def _bytes(field: Any) -> bytes:
    return field.tobytes() if isinstance(field, array.array) else bytes(field)


def make_decoder(name: str) -> Decoder:
    """``audio_common`` (stamped), ``audio_common_unstamped`` or a plugin's name.

    Raises:
        ConfigError: If no decoder has that name.
    """
    if name == "audio_common":
        return AudioCommonDecoder(stamped=True)
    if name == "audio_common_unstamped":
        return AudioCommonDecoder(stamped=False)
    for ep in entry_points(group="ral.decoders"):
        if ep.name == name:
            decoder: Decoder = ep.load()()
            return decoder
    raise ConfigError(
        f"No ROS 2 audio decoder {name!r}. Built in: audio_common, audio_common_unstamped; "
        f"vendor message types come from plugin packages (entry point 'ral.decoders')."
    )


def stamped_message(pcm: bytes, stamp_ros_ns: int, frame_id: str = "") -> Any:
    """An ``AudioDataStamped`` carrying ``pcm`` (S16LE) stamped at ``stamp_ros_ns``."""
    msgs = import_msgs("audio_common_msgs")
    msg = msgs.AudioDataStamped()
    msg.header.stamp.sec = stamp_ros_ns // 1_000_000_000
    msg.header.stamp.nanosec = stamp_ros_ns % 1_000_000_000
    msg.header.frame_id = frame_id
    # uint8[] fields are array.array('B') in rclpy (rosidl_generator_py); assigning one
    # skips the per-element checks a bytes or list value would go through.
    msg.audio.data = array.array("B", pcm)
    return msg


def info_message(rate: int, channels: int) -> Any:
    """``AudioInfo`` for 16-bit little-endian PCM."""
    msgs = import_msgs("audio_common_msgs")
    info = msgs.AudioInfo()
    info.channels = channels
    info.sample_rate = rate
    info.sample_format = "S16LE"
    info.bitrate = rate * channels * 16
    info.coding_format = "wave"
    return info
