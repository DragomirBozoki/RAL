"""WebSocket bridge: clean audio and events out, agent audio and control in.

For clients in any language (DESIGN section 7). Protocol version 1, one JSON text message
or one binary message per WebSocket message:

Server to client
    - ``{"type": "hello", "protocol": 1, "session_id": ..., "capture": {...},
      "playback": {...}}`` once, right after connecting. Both formats are
      ``{"format": "pcm-s16", "rate": 16000, "channels": 1}``.
    - Binary: one clean 10 ms frame, int16 little-endian, mono (320 bytes at 16 kHz).
      Connect with ``?audio=0`` for events only.
    - Text: every engine event as ``{"type": "<EventName>", ...}``; replies
      ``{"type": "pong"}`` and ``{"type": "error", "message": ...}``.

Client to server
    - Binary: audio to play, int16 little-endian mono at the playback rate, any length.
      Chunks join the connection's current utterance; the first chunk starts one if needed.
    - ``{"type": "play", "tag": "answer", "priority": 50, "interruptible": true,
      "rate": 24000}`` starts a new utterance (ending the previous one); ``rate`` is the
      sample rate of the binary audio that follows (default: the playback rate).
    - ``{"type": "end"}`` ends the current utterance once its audio has played.
    - ``{"type": "stop", "fade_ms": 40}`` stops this connection's utterance.
    - ``{"type": "interrupt"}`` fades out whatever is playing (like a barge-in).
    - ``{"type": "duck", "gain_db": -12, "duration_ms": 2000}``.
    - ``{"type": "ping"}``.

The server listens on 127.0.0.1 by default; binding to another address exposes the
robot's microphone to that network, so it has to be asked for explicitly. A client that
reads too slowly loses its oldest messages rather than delaying audio for others.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from ral.core.convert import float32_to_int16
from ral.core.errors import ConfigError, MissingDependencyError
from ral.core.events import event_to_json
from ral.core.protocols import PlaybackHandle
from ral.playback.stream import PcmStream

if TYPE_CHECKING:
    from ral.engine.engine import Engine

__all__ = ["PROTOCOL_VERSION", "WebSocketBridge"]

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
_CLIENT_QUEUE = 400  # about 4 s of audio frames plus events


class _Client:
    def __init__(self, conn: Any, audio: bool) -> None:
        self.conn = conn
        self.audio = audio
        self.queue: asyncio.Queue[bytes | str] = asyncio.Queue(_CLIENT_QUEUE)
        self.dropped = 0
        self.stream: PcmStream | None = None
        self.handle: PlaybackHandle | None = None

    def offer(self, message: bytes | str) -> None:
        if self.queue.full():
            self.queue.get_nowait()
            self.dropped += 1
        self.queue.put_nowait(message)


class WebSocketBridge:
    """Serves the engine over WebSocket.

    Args:
        host: Address to listen on (default 127.0.0.1, this machine only).
        port: TCP port; 0 picks a free one (see :attr:`port` after attaching).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._engine: Engine | None = None
        self._server: Any = None
        self._clients: set[_Client] = set()
        self._tasks: list[asyncio.Task[None]] = []

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> WebSocketBridge:
        """Build from a profile's ``bridges.websocket`` block.

        Raises:
            ConfigError: On unknown keys.
        """
        unknown = set(cfg) - {"host", "port"}
        if unknown:
            raise ConfigError(f"bridges.websocket has unknown keys {sorted(unknown)}.")
        return cls(str(cfg.get("host", "127.0.0.1")), int(cfg.get("port", 8765)))

    async def attach(self, engine: Engine) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError:
            raise MissingDependencyError("websockets", "server", "The WebSocket bridge") from None
        self._engine = engine
        # Audio is incompressible enough that deflate only costs CPU.
        self._server = await serve(self._handle, self.host, self.port, compression=None)
        sockets = list(self._server.sockets)
        if sockets:
            self.port = int(sockets[0].getsockname()[1])
        frames = engine.subscribe(maxsize=200)
        events = engine.bus.subscribe()
        self._tasks = [
            asyncio.create_task(self._fan_audio(frames), name="ral-ws-audio"),
            asyncio.create_task(self._fan_events(events), name="ral-ws-events"),
        ]
        log.info("WebSocket bridge on ws://%s:%d", self.host, self.port)

    async def detach(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for client in list(self._clients):
            if client.stream is not None:
                client.stream.end()

    # Fan-out to clients

    async def _fan_audio(self, frames: Any) -> None:
        async for result in frames:
            if not self._clients:
                continue
            payload = float32_to_int16(result.clean.data[:, 0]).astype("<i2").tobytes()
            for client in list(self._clients):
                if client.audio:
                    client.offer(payload)

    async def _fan_events(self, events: Any) -> None:
        async for event in events:
            text = json.dumps(event_to_json(event))
            for client in list(self._clients):
                client.offer(text)

    # One connection

    async def _handle(self, conn: Any) -> None:
        engine = self._engine
        assert engine is not None
        query = parse_qs(urlparse(getattr(conn.request, "path", "/")).query)
        client = _Client(conn, audio=query.get("audio", ["1"])[0] not in ("0", "false"))
        fmt = {"format": "pcm-s16", "rate": engine.engine_fmt.sample_rate, "channels": 1}
        await conn.send(
            json.dumps(
                {
                    "type": "hello",
                    "protocol": PROTOCOL_VERSION,
                    "session_id": engine.session_id,
                    "capture": fmt,
                    "playback": fmt,
                }
            )
        )
        self._clients.add(client)
        sender = asyncio.create_task(self._send_loop(client))
        try:
            async for message in conn:
                if isinstance(message, bytes):
                    self._audio_in(client, message)
                else:
                    await self._control(client, message)
        except Exception as exc:  # a broken client must not take the bridge down
            log.debug("WebSocket client closed: %s", exc)
        finally:
            self._clients.discard(client)
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender
            if client.stream is not None:
                client.stream.end()

    async def _send_loop(self, client: _Client) -> None:
        while True:
            message = await client.queue.get()
            await client.conn.send(message)

    def _start_utterance(
        self,
        client: _Client,
        tag: str = "ws",
        priority: int = 50,
        interruptible: bool = True,
        rate: int | None = None,
    ) -> None:
        engine = self._engine
        assert engine is not None
        assert engine.playback is not None
        if rate is not None and rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if client.stream is not None:
            client.stream.end()
        # Audio at another rate (a TTS at 24 kHz, say) is resampled to the engine rate.
        client.stream = PcmStream(rate or engine.engine_fmt.sample_rate, engine.engine_fmt)
        client.handle = engine.playback.play(
            aiter(client.stream), priority=priority, interruptible=interruptible, tag=tag
        )

    def _audio_in(self, client: _Client, data: bytes) -> None:
        if client.stream is None or client.stream.ended:
            self._start_utterance(client)
        assert client.stream is not None
        try:
            client.stream.push(data)
        except Exception as exc:
            client.offer(json.dumps({"type": "error", "message": str(exc)}))

    async def _control(self, client: _Client, text: str) -> None:
        engine = self._engine
        assert engine is not None
        assert engine.playback is not None
        try:
            msg = json.loads(text)
            kind = msg.get("type")
            if kind == "play":
                self._start_utterance(
                    client,
                    str(msg.get("tag", "ws")),
                    int(msg.get("priority", 50)),
                    bool(msg.get("interruptible", True)),
                    int(msg["rate"]) if "rate" in msg else None,
                )
            elif kind == "end":
                if client.stream is not None:
                    client.stream.end()
            elif kind == "stop":
                if client.stream is not None:
                    client.stream.end()
                if client.handle is not None:
                    await engine.playback.stop(client.handle, int(msg.get("fade_ms", 40)))
            elif kind == "interrupt":
                engine.playback.interrupt(int(msg.get("fade_ms", 40)))
            elif kind == "duck":
                engine.playback.duck(float(msg["gain_db"]), int(msg["duration_ms"]))
            elif kind == "ping":
                client.offer(json.dumps({"type": "pong"}))
            else:
                raise ValueError(f"unknown message type {kind!r}")
        except Exception as exc:
            client.offer(json.dumps({"type": "error", "message": str(exc)}))
