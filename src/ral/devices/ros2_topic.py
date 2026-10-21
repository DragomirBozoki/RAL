"""ROS 2 topic backend: audio from and to topics (DESIGN sections 4 and 5).

Capture subscribes to an audio topic (``audio_common_msgs/AudioDataStamped`` by default,
other types through decoder plugins). The subscription callback runs on the ROS executor
thread and only converts the message's bytes to a sample view and pushes it into a
:class:`~ral.core.Framer` with its capture time (CLAUDE.md, rule 4). The capture time is
the message stamp mapped to the RAL clock, or for unstamped messages the arrival time
minus the block's duration.

Playback publishes ``AudioDataStamped`` every 10 ms from a ring buffer, silence when
nothing was written, like a device that never stops (so the echo canceller's timeline
stays continuous). The DAC time of a block is unknown on a topic, so it is estimated as
publish time plus ``latency_ms`` (DESIGN 5, "ROS 2 sinks"), which ``ral probe`` measures.
An ``AudioInfo`` with the format is published once on ``<topic>_info`` (transient local).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import numpy.typing as npt

from ral.core.clock import now_ns
from ral.core.convert import float32_to_int16
from ral.core.errors import DeviceError, DeviceNotFoundError, FormatError
from ral.core.framer import Framer
from ral.core.protocols import RenderHook
from ral.core.ringbuffer import RingBuffer
from ral.core.types import AudioFormat, AudioFrame, ChannelMap, DeviceInfo
from ral.ros2 import Ros2Runtime, RosClockMapper, qos_profile
from ral.ros2.decoders import Decoder, info_message, make_decoder, stamped_message

__all__ = ["Ros2TopicBackend", "Ros2TopicSink", "Ros2TopicSource"]

_POLL_S = 0.005


class Ros2TopicSource:
    """Captures from one audio topic. ``fmt`` must describe the message payload."""

    def __init__(
        self,
        runtime: Ros2Runtime,
        topic: str,
        fmt: AudioFormat,
        channel_map: ChannelMap,
        decoder: Decoder,
        *,
        reliability: str = "reliable",
        capacity_s: float = 2.0,
    ) -> None:
        if fmt.dtype != "int16":
            raise FormatError("ROS 2 audio topics carry 16-bit PCM (S16LE); use dtype int16.")
        self.fmt = fmt
        self.channel_map = channel_map
        self.topic = topic
        self._runtime = runtime
        self._decoder = decoder
        self._reliability = reliability
        self._clock = RosClockMapper()
        self.framer = Framer(fmt, channel_map, f"ros2:{topic}", capacity_s=capacity_s)
        self._sub: Any = None
        self._running = False
        self.bad_messages = 0

    async def start(self) -> None:
        self._running = True
        self._sub = self._runtime.node.create_subscription(
            self._decoder.msg_type,
            self.topic,
            self._callback,
            qos_profile(self._reliability, depth=100),
        )

    async def stop(self) -> None:
        self._running = False
        if self._sub is not None:
            self._runtime.node.destroy_subscription(self._sub)
            self._sub = None

    async def frames(self) -> AsyncIterator[AudioFrame]:
        if not self._running:
            raise DeviceError(f"The source for {self.topic} is not started; call start() first.")
        while self._running:
            frame = self.framer.read_frame()
            if frame is None:
                await asyncio.sleep(_POLL_S)
                continue
            yield frame

    def _callback(self, msg: Any) -> None:
        # ROS executor thread: decode to a view, stamp, copy into the ring buffer. No logging.
        try:
            pcm, stamp = self._decoder.decode(msg)
            samples = np.frombuffer(pcm, dtype="<i2").reshape(-1, self.fmt.channels)
        except Exception:
            self.bad_messages += 1
            return
        arrival = now_ns()
        self._clock.observe(self._runtime.ros_now_ns(), arrival)
        if stamp is not None:
            t_ns = self._clock.to_ral_ns(stamp)
        else:
            t_ns = arrival - samples.shape[0] * 1_000_000_000 // self.fmt.sample_rate
        self.framer.push(samples.astype(np.int16, copy=False), t_ns)


class Ros2TopicSink:
    """Publishes playback to one audio topic every 10 ms."""

    def __init__(
        self,
        runtime: Ros2Runtime,
        topic: str,
        fmt: AudioFormat,
        *,
        latency_ms: int = 0,
        reliability: str = "reliable",
        capacity_s: float = 2.0,
    ) -> None:
        self.fmt = fmt
        self.topic = topic
        self._runtime = runtime
        self._latency_ns = latency_ms * 1_000_000
        self._reliability = reliability
        self._hooks: list[RenderHook] = []
        self.buffer = RingBuffer(
            fmt.channels,
            fmt.dtype,
            max(fmt.frame_samples, round(capacity_s * fmt.sample_rate)),
            overrun="drop_newest",
        )
        self._pub: Any = None
        self._info_pub: Any = None
        self._task: asyncio.Task[None] | None = None
        self.published = 0

    async def start(self) -> None:
        node = self._runtime.node
        self._pub = node.create_publisher(
            self._msg_type(), self.topic, qos_profile(self._reliability, depth=100)
        )
        self._info_pub = node.create_publisher(
            type(info_message(1, 1)),
            f"{self.topic}_info",
            qos_profile(depth=1, transient_local=True),
        )
        self._info_pub.publish(info_message(self.fmt.sample_rate, self.fmt.channels))
        self._task = asyncio.create_task(self._render_loop(), name=f"ral-ros2-sink-{self.topic}")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        for pub in (self._pub, self._info_pub):
            if pub is not None:
                self._runtime.node.destroy_publisher(pub)
        self._pub = self._info_pub = None

    def write(self, frame: AudioFrame) -> None:
        if frame.fmt != self.fmt:
            raise FormatError(f"Frame format {frame.fmt} does not match the sink's {self.fmt}.")
        self.buffer.push(frame.data, frame.t_ns)

    def output_latency_ns(self) -> int:
        return self._latency_ns

    def on_rendered(self, hook: RenderHook) -> None:
        self._hooks.append(hook)

    def _msg_type(self) -> Any:
        return type(stamped_message(b"", 0))

    async def _render_loop(self) -> None:
        block: npt.NDArray[Any] = np.empty(
            (self.fmt.frame_samples, self.fmt.channels), dtype=self.fmt.numpy_dtype
        )
        t_start = now_ns()
        k = 0
        while True:
            # Absolute deadlines, so scheduling jitter never accumulates into drift.
            deadline = t_start + k * self.fmt.frame_ns
            while (delay := deadline - now_ns()) > 0:
                await asyncio.sleep(delay / 1e9)
            self.buffer.read_available(block)  # silence on underrun
            pcm = block if self.fmt.dtype == "int16" else float32_to_int16(block)
            ros_now = self._runtime.ros_now_ns()
            self._pub.publish(stamped_message(pcm.astype("<i2").tobytes(), ros_now, "ral"))
            self.published += 1
            t_dac = now_ns() + self._latency_ns
            for hook in self._hooks:
                hook(block, t_dac)
            k += 1


class Ros2TopicBackend:
    """Audio topics as devices.

    Args:
        runtime: The ROS 2 runtime (created on first use if not given).
        decoder: Capture decoder name (``audio_common``, ``audio_common_unstamped`` or a plugin).
        reliability: QoS reliability for audio topics (``reliable`` or ``best_effort``).
        latency_ms: Publish-to-DAC estimate for sinks (from ``ral probe``).
    """

    name = "ros2_topic"

    def __init__(
        self,
        runtime: Ros2Runtime | None = None,
        *,
        decoder: str = "audio_common",
        reliability: str = "reliable",
        latency_ms: int = 0,
    ) -> None:
        self._runtime = runtime
        self.decoder_name = decoder
        self.reliability = reliability
        self.latency_ms = latency_ms

    @property
    def runtime(self) -> Ros2Runtime:
        if self._runtime is None:
            self._runtime = Ros2Runtime("ral_audio")
        return self._runtime

    def device(self, topic: str, direction: str) -> DeviceInfo:
        """A device description for ``topic`` (topics are not enumerated like sound cards)."""
        if not topic.startswith("/"):
            raise DeviceNotFoundError(f"ROS 2 topics are absolute names; got {topic!r}.")
        inputs, outputs = (64, 0) if direction == "input" else (0, 64)
        return DeviceInfo(f"ros2:{topic}", topic, self.name, None, inputs, outputs, None)

    def discover(self) -> list[DeviceInfo]:
        """Topics whose type is a known audio message (``ral doctor`` lists all audio-like)."""
        out = []
        for topic, types in self.runtime.node.get_topic_names_and_types():
            if any(t.startswith("audio_common_msgs/") for t in types):
                out.append(self.device(topic, "input"))
        return out

    def open_source(
        self, device: DeviceInfo, fmt: AudioFormat, channel_map: ChannelMap
    ) -> Ros2TopicSource:
        return Ros2TopicSource(
            self.runtime,
            device.name,
            fmt,
            channel_map,
            make_decoder(self.decoder_name),
            reliability=self.reliability,
        )

    def open_sink(self, device: DeviceInfo, fmt: AudioFormat) -> Ros2TopicSink:
        return Ros2TopicSink(
            self.runtime,
            device.name,
            fmt,
            latency_ms=self.latency_ms,
            reliability=self.reliability,
        )
