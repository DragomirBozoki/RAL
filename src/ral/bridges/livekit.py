"""LiveKit bridge: RAL joins the room as the robot (DESIGN section 7).

The bridge publishes the clean capture stream as the robot's microphone track, plays the
agent's audio through the playback service (so it becomes the echo reference), and sends
every event as a data message on topic ``ral.events``. Data messages on topic
``ral.control`` accept the WebSocket bridge's control messages (``interrupt``, ``duck``,
``stop``). Nothing changes in the agent: it sees an ordinary participant.

Utterances. A remote track is continuous: between turns it carries silence. Playing it as
one endless item would mark the robot as speaking all the time, which breaks the gate and
barge-in. The bridge therefore cuts the track into utterances: an utterance starts at the
first frame above ``speech_db`` and ends after ``silence_ms`` below it (the quiet tail
is played too, so word endings are not clipped). After a barge-in, the rest of the
cancelled turn is dropped until the agent falls silent, so it does not start playing
again a moment later.

Timing (``spikes/livekit_room.py``): ``AudioStream(..., frame_size_ms=10)`` delivers
10 ms mono frames at the engine rate, so neither direction is resampled; the room adds
about 64 ms on localhost.

Credentials come from the profile (usually ``${LIVEKIT_URL}`` and friends from ``.env``):
either a ready ``token``, or ``api_key`` and ``api_secret`` to mint one for ``room`` and
``identity``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ral.core.convert import float32_to_int16, int16_to_float32
from ral.core.errors import ConfigError, MissingDependencyError
from ral.core.events import BargeIn, Event, event_to_json
from ral.core.protocols import PlaybackHandle
from ral.dsp.levels import rms_db
from ral.playback.stream import PcmStream

if TYPE_CHECKING:
    from ral.engine.engine import Engine

__all__ = ["LiveKitBridge", "UtteranceSegmenter"]

log = logging.getLogger(__name__)

EVENTS_TOPIC = "ral.events"
CONTROL_TOPIC = "ral.control"
_KEYS = {
    "url",
    "room",
    "identity",
    "token",
    "api_key",
    "api_secret",
    "publish",
    "play",
    "agent_identity",
    "speech_db",
    "silence_ms",
}


class UtteranceSegmenter:
    """Cuts a continuous track into utterances by level (no LiveKit dependency).

    :meth:`feed` returns ``"start"``, ``"continue"``, ``"end"`` or ``"skip"`` for each
    frame: ``skip`` means not part of any utterance (silence, or a dropped turn).
    """

    def __init__(self, speech_db: float = -50.0, silence_ms: int = 300) -> None:
        self.speech_db = speech_db
        self.silence_frames = max(1, silence_ms // 10)
        self.active = False
        self._quiet = 0
        self._dropping = False

    def drop_current_turn(self) -> None:
        """After a barge-in: ignore audio until the agent has been quiet once."""
        self._dropping = True
        self.active = False
        self._quiet = 0

    def feed(self, level_db: float) -> str:
        loud = level_db > self.speech_db
        if self._dropping:
            self._quiet = 0 if loud else self._quiet + 1
            if self._quiet >= self.silence_frames:
                self._dropping = False
                self._quiet = 0
            return "skip"
        if not self.active:
            if loud:
                self.active = True
                self._quiet = 0
                return "start"
            return "skip"
        self._quiet = 0 if loud else self._quiet + 1
        if self._quiet >= self.silence_frames:
            self.active = False
            self._quiet = 0
            return "end"
        return "continue"


class LiveKitBridge:
    """Joins a LiveKit room as the robot.

    Args:
        url: LiveKit server URL (``wss://...``).
        room: Room name.
        identity: The robot's participant identity.
        token: A ready access token; otherwise ``api_key`` and ``api_secret`` mint one.
        publish: ``clean_mic`` (default) or ``none``.
        play: ``agent_audio`` (default) or ``none``.
        agent_identity: Only play audio from this participant (default: any).
        speech_db, silence_ms: Utterance segmentation, see :class:`UtteranceSegmenter`.
    """

    def __init__(
        self,
        url: str,
        room: str,
        *,
        identity: str = "robot",
        token: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        publish: str = "clean_mic",
        play: str = "agent_audio",
        agent_identity: str | None = None,
        speech_db: float = -50.0,
        silence_ms: int = 300,
    ) -> None:
        if publish not in ("clean_mic", "none") or play not in ("agent_audio", "none"):
            raise ConfigError(
                "bridges.livekit: publish is clean_mic|none, play is agent_audio|none."
            )
        self.url = url
        self.room_name = room
        self.identity = identity
        self.token = token
        self.api_key = api_key or os.environ.get("LIVEKIT_API_KEY")
        self.api_secret = api_secret or os.environ.get("LIVEKIT_API_SECRET")
        self.publish = publish
        self.play = play
        self.agent_identity = agent_identity
        self.speech_db = speech_db
        self.silence_ms = silence_ms
        self.room: Any = None
        self._rtc: Any = None
        self._engine: Engine | None = None
        self._source: Any = None
        self._tasks: list[asyncio.Task[None]] = []
        self._side: set[asyncio.Future[None]] = set()
        self._readers: dict[str, asyncio.Task[None]] = {}
        self._segmenters: dict[str, UtteranceSegmenter] = {}
        self._streams: dict[str, tuple[PcmStream, PlaybackHandle]] = {}
        self.frames_published = 0
        self.frames_received = 0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> LiveKitBridge:
        """Build from a profile's ``bridges.livekit`` block.

        Raises:
            ConfigError: On unknown or missing keys.
        """
        unknown = set(cfg) - _KEYS
        if unknown:
            raise ConfigError(f"bridges.livekit has unknown keys {sorted(unknown)}.")
        missing = [k for k in ("url", "room") if not cfg.get(k)]
        if missing:
            raise ConfigError(f"bridges.livekit needs {', '.join(missing)}.")
        kwargs = {k: v for k, v in cfg.items() if k not in ("url", "room")}
        return cls(str(cfg["url"]), str(cfg["room"]), **kwargs)

    def _make_token(self) -> str:
        if self.token:
            return self.token
        if not (self.api_key and self.api_secret):
            raise ConfigError(
                "bridges.livekit needs a token, or api_key and api_secret (for example "
                "LIVEKIT_API_KEY and LIVEKIT_API_SECRET in .env)."
            )
        try:
            from livekit import api
        except ImportError:
            raise MissingDependencyError(
                "livekit-api", "livekit", "Minting LiveKit tokens"
            ) from None
        grants = api.VideoGrants(room_join=True, room=self.room_name)
        jwt: str = (
            api.AccessToken(self.api_key, self.api_secret)
            .with_identity(self.identity)
            .with_name(self.identity)
            .with_grants(grants)
            .to_jwt()
        )
        return jwt

    async def attach(self, engine: Engine) -> None:
        try:
            from livekit import rtc
        except ImportError:
            raise MissingDependencyError("livekit", "livekit", "The LiveKit bridge") from None
        self._rtc = rtc
        self._engine = engine
        token = self._make_token()
        room = rtc.Room()
        room.on("track_subscribed", self._on_track)
        room.on("track_unsubscribed", self._on_track_gone)
        room.on("data_received", self._on_data)
        await room.connect(self.url, token)
        self.room = room
        if self.publish == "clean_mic":
            rate = engine.engine_fmt.sample_rate
            self._source = rtc.AudioSource(rate, 1)
            track = rtc.LocalAudioTrack.create_audio_track("ral-clean-mic", self._source)
            opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            await room.local_participant.publish_track(track, opts)
            self._tasks.append(asyncio.create_task(self._publish_audio(engine.subscribe(200))))
        self._tasks.append(asyncio.create_task(self._publish_events(engine.bus.subscribe())))
        log.info("LiveKit bridge joined %s as %s", self.room_name, self.identity)

    async def detach(self) -> None:
        for task in [*self._tasks, *self._readers.values()]:
            task.cancel()
        for task in [*self._tasks, *self._readers.values()]:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks, self._readers = [], {}
        for stream, _ in self._streams.values():
            stream.end()
        self._streams.clear()
        if self._source is not None:
            with contextlib.suppress(Exception):
                await self._source.aclose()
            self._source = None
        if self.room is not None:
            with contextlib.suppress(Exception):
                await self.room.disconnect()
            self.room = None

    # Robot to room

    async def _publish_audio(self, frames: Any) -> None:
        rtc = self._rtc
        async for result in frames:
            pcm = float32_to_int16(result.clean.data[:, 0])
            n = pcm.shape[0]
            frame = rtc.AudioFrame(
                bytearray(pcm.astype("<i2").tobytes()), result.clean.fmt.sample_rate, 1, n
            )
            await self._source.capture_frame(frame)
            self.frames_published += 1

    async def _publish_events(self, events: Any) -> None:
        async for event in events:
            if isinstance(event, BargeIn):
                for seg in self._segmenters.values():
                    seg.drop_current_turn()
                for stream, _ in self._streams.values():
                    stream.end()
                self._streams.clear()
            await self._send_event(event)

    async def _send_event(self, event: Event) -> None:
        if self.room is None:
            return
        with contextlib.suppress(Exception):  # a lost data message must not stop audio
            await self.room.local_participant.publish_data(
                json.dumps(event_to_json(event)), reliable=True, topic=EVENTS_TOPIC
            )

    # Room to robot

    def _on_track(self, track: Any, publication: Any, participant: Any) -> None:
        rtc = self._rtc
        if self.play != "agent_audio" or track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if self.agent_identity and participant.identity != self.agent_identity:
            return
        key = f"{participant.identity}/{track.sid}"
        self._segmenters[key] = UtteranceSegmenter(self.speech_db, self.silence_ms)
        self._readers[key] = asyncio.create_task(self._read_track(key, track, participant.identity))

    def _on_track_gone(self, track: Any, publication: Any, participant: Any) -> None:
        key = f"{participant.identity}/{track.sid}"
        task = self._readers.pop(key, None)
        if task is not None:
            task.cancel()
        entry = self._streams.pop(key, None)
        if entry is not None:
            entry[0].end()

    async def _read_track(self, key: str, track: Any, who: str) -> None:
        engine = self._engine
        assert engine is not None
        rate = engine.engine_fmt.sample_rate
        stream = self._rtc.AudioStream(track, sample_rate=rate, num_channels=1, frame_size_ms=10)
        try:
            async for ev in stream:
                pcm = np.frombuffer(ev.frame.data, dtype=np.int16)
                self.frames_received += 1
                self._route(key, who, int16_to_float32(pcm.copy()))
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()

    def _route(self, key: str, who: str, samples: npt.NDArray[np.float32]) -> None:
        engine = self._engine
        assert engine is not None
        assert engine.playback is not None
        action = self._segmenters[key].feed(rms_db(samples))
        if action == "skip":
            return
        if action == "start":
            stream = PcmStream(engine.engine_fmt.sample_rate, engine.engine_fmt)
            handle = engine.playback.play(aiter(stream), tag=f"livekit:{who}")
            self._streams[key] = (stream, handle)
        entry = self._streams.get(key)
        if entry is None:
            return
        entry[0].push(samples)
        if action == "end":
            entry[0].end()
            del self._streams[key]

    def _on_data(self, packet: Any) -> None:
        if getattr(packet, "topic", "") != CONTROL_TOPIC or self._engine is None:
            return
        playback = self._engine.playback
        if playback is None:
            return
        try:
            msg = json.loads(bytes(packet.data).decode())
            kind = msg.get("type")
            if kind == "interrupt":
                playback.interrupt(int(msg.get("fade_ms", 40)))
            elif kind == "duck":
                playback.duck(float(msg["gain_db"]), int(msg["duration_ms"]))
            elif kind == "stop":
                for stream, handle in list(self._streams.values()):
                    stream.end()
                    task = asyncio.ensure_future(playback.stop(handle, int(msg.get("fade_ms", 40))))
                    self._side.add(task)  # keep a reference until it finishes
                    task.add_done_callback(self._side.discard)
                self._streams.clear()
        except Exception as exc:
            log.warning("Ignoring bad control message from LiveKit: %s", exc)
