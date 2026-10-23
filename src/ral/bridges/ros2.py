"""ROS 2 bridge: the engine as ROS 2 topics (DESIGN section 7).

Under ``namespace`` (default ``/ral``):

- ``clean`` (``audio_common_msgs/AudioDataStamped``, 10 ms of 16-bit mono at the engine
  rate, stamped with its capture time in ROS time) and ``clean_info`` (``AudioInfo``,
  transient local, since the data message carries no format).
- ``events`` (``std_msgs/String``): every event as JSON, ``{"type": ..., ...}``.
- ``play`` (``AudioDataStamped`` in, 16-bit mono at the engine rate): audio to speak. Each
  burst becomes one playback item; an item ends after ``gap_ms`` without new audio.
- ``control`` (``std_msgs/String`` in): ``{"type": "interrupt" | "stop" | "duck" | "end"}``
  as in the WebSocket protocol.

With ``ral_interfaces`` (``integrations/ros2``, built with colcon) sourced, also the
``play_audio`` and ``say`` actions and the ``stop`` and ``duck`` services
(:mod:`ral.ros2.actions`). ``actions: auto`` (default) serves them when the package is
found, ``true`` requires it, ``false`` serves topics only.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ral.core.clock import now_ns
from ral.core.convert import float32_to_int16
from ral.core.errors import ConfigError, MissingDependencyError
from ral.core.events import event_to_json
from ral.core.protocols import PlaybackHandle
from ral.playback.stream import PcmStream
from ral.ros2 import Ros2Runtime, RosClockMapper, import_msgs, qos_profile
from ral.ros2.actions import Ros2Actions, TextToSpeechFn
from ral.ros2.decoders import AudioCommonDecoder, info_message, stamped_message

if TYPE_CHECKING:
    from ral.engine.engine import Engine

__all__ = ["Ros2Bridge"]

log = logging.getLogger(__name__)


class Ros2Bridge:
    """Publishes the clean stream and events; plays audio and obeys control topics.

    Args:
        namespace: Topic prefix.
        gap_ms: Silence on ``play`` that ends an item.
        runtime: An existing ROS 2 runtime to share (a new one by default).
        actions: Serve the actions and services: "auto", True or False.
        tts: Speaks ``Say`` goals (rejected without one).
    """

    def __init__(
        self,
        namespace: str = "/ral",
        gap_ms: int = 300,
        runtime: Ros2Runtime | None = None,
        *,
        actions: bool | str = "auto",
        tts: TextToSpeechFn | None = None,
    ) -> None:
        if actions not in ("auto", True, False):
            raise ConfigError("bridges.ros2.actions must be auto, true or false.")
        self.actions_mode = actions
        self.tts = tts
        self.actions: Ros2Actions | None = None
        self.ns = "/" + namespace.strip("/") if namespace.strip("/") else ""
        self.gap_ns = gap_ms * 1_000_000
        self._runtime = runtime
        self._owns_runtime = runtime is None
        self._engine: Engine | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clock = RosClockMapper()
        self._stream: PcmStream | None = None
        self._handle: PlaybackHandle | None = None
        self._last_play_ns = 0
        self._subs: list[Any] = []
        self._pubs: dict[str, Any] = {}
        self.published = 0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Ros2Bridge:
        unknown = set(cfg) - {"namespace", "gap_ms", "actions"}
        if unknown:
            raise ConfigError(f"bridges.ros2 has unknown keys {sorted(unknown)}.")
        return cls(
            str(cfg.get("namespace", "/ral")),
            int(cfg.get("gap_ms", 300)),
            actions=cfg.get("actions", "auto"),
        )

    def topic(self, name: str) -> str:
        return f"{self.ns}/{name}"

    async def attach(self, engine: Engine) -> None:
        self._engine = engine
        self._loop = asyncio.get_running_loop()
        if self._runtime is None:
            self._runtime = Ros2Runtime("ral_bridge")
        node = self._runtime.node
        std = import_msgs("std_msgs")
        audio_type = type(stamped_message(b"", 0))
        self._pubs = {
            "clean": node.create_publisher(audio_type, self.topic("clean"), qos_profile(depth=100)),
            "info": node.create_publisher(
                type(info_message(1, 1)),
                self.topic("clean_info"),
                qos_profile(depth=1, transient_local=True),
            ),
            "events": node.create_publisher(
                std.String, self.topic("events"), qos_profile(depth=100)
            ),
        }
        self._pubs["info"].publish(info_message(engine.engine_fmt.sample_rate, 1))
        self._subs = [
            node.create_subscription(
                AudioCommonDecoder(True).msg_type,
                self.topic("play"),
                self._on_play,
                qos_profile(depth=200),
            ),
            node.create_subscription(
                std.String, self.topic("control"), self._on_control, qos_profile(depth=50)
            ),
        ]
        if self.actions_mode is not False:
            try:
                self.actions = Ros2Actions(self._runtime, engine, self._loop, self.ns, self.tts)
            except MissingDependencyError:
                if self.actions_mode is True:
                    raise
                log.info("ral_interfaces not found; serving ROS 2 topics without actions.")
        self._tasks = [
            asyncio.create_task(self._publish_clean(engine.subscribe(200))),
            asyncio.create_task(self._publish_events(engine.bus.subscribe())),
            asyncio.create_task(self._end_idle_items()),
        ]

    async def detach(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []
        if self._stream is not None:
            self._stream.end()
        if self.actions is not None:
            self.actions.close()
            self.actions = None
        if self._runtime is not None:
            node = self._runtime.node
            for sub in self._subs:
                node.destroy_subscription(sub)
            for pub in self._pubs.values():
                node.destroy_publisher(pub)
            if self._owns_runtime:
                self._runtime.close()
                self._runtime = None
        self._subs, self._pubs = [], {}

    # Engine to ROS

    async def _publish_clean(self, results: Any) -> None:
        assert self._runtime is not None
        async for result in results:
            self._clock.observe(self._runtime.ros_now_ns())
            offset = self._clock.offset_ns or 0
            pcm = float32_to_int16(result.clean.data[:, 0]).astype("<i2").tobytes()
            # Stamp with the capture time, in ROS time.
            self._pubs["clean"].publish(stamped_message(pcm, result.clean.t_ns - offset, "ral"))
            self.published += 1

    async def _publish_events(self, events: Any) -> None:
        std = import_msgs("std_msgs")
        async for event in events:
            msg = std.String()
            msg.data = json.dumps(event_to_json(event))
            self._pubs["events"].publish(msg)

    # ROS to engine (callbacks run on the ROS executor thread: hand over to the loop)

    def _on_play(self, msg: Any) -> None:
        pcm = _audio_bytes(msg)  # a copy: the message may be reused after the callback
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._play_chunk, pcm)

    def _on_control(self, msg: Any) -> None:
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._control, str(msg.data))

    def _play_chunk(self, pcm: bytes) -> None:
        engine = self._engine
        assert engine is not None
        assert engine.playback is not None
        if self._stream is None or self._stream.ended:
            self._stream = PcmStream(engine.engine_fmt.sample_rate, engine.engine_fmt)
            self._handle = engine.playback.play(aiter(self._stream), tag="ros2")
        self._stream.push(pcm)
        self._last_play_ns = now_ns()

    async def _end_idle_items(self) -> None:
        while True:
            await asyncio.sleep(0.05)
            if (
                self._stream is not None
                and not self._stream.ended
                and now_ns() - self._last_play_ns > self.gap_ns
            ):
                self._stream.end()

    def _control(self, text: str) -> None:
        engine = self._engine
        if engine is None or engine.playback is None:
            return
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            return
        kind = msg.get("type")
        if kind == "interrupt":
            engine.playback.interrupt(int(msg.get("fade_ms", 40)))
        elif kind == "duck":
            engine.playback.duck(float(msg["gain_db"]), int(msg["duration_ms"]))
        elif kind in ("stop", "end"):
            if self._stream is not None:
                self._stream.end()
            if kind == "stop" and self._handle is not None:
                task = asyncio.ensure_future(
                    engine.playback.stop(self._handle, int(msg.get("fade_ms", 40)))
                )
                self._tasks.append(task)


def _audio_bytes(msg: Any) -> bytes:
    data = msg.audio.data
    return data.tobytes() if hasattr(data, "tobytes") else bytes(data)
