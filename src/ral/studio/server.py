"""The Studio: a local web page for tuning a running engine (DESIGN section 9, "Local UI").

Bridge ``studio`` (``ral run -b studio``, then open http://127.0.0.1:8770). One asyncio
HTTP server, no dependency:

- ``GET /`` the page (``page.html``: canvas meters, a scrolling spectrogram, the channel
  map editor and the echo-mode switch);
- ``GET /events`` a Server-Sent Events stream: a ``meters`` message every 100 ms (level
  per raw channel, clean level, reference level, a 128-band spectrum of the clean stream)
  and every bus event as ``event``
  (https://html.spec.whatwg.org/multipage/server-sent-events.html);
- ``GET /api/state`` profile, echo mode, channel roles and the health snapshot;
- ``POST /api/echo`` ``{"mode": ...}`` switches echo control live
  (:meth:`ral.engine.engine.Engine.set_echo_mode`);
- ``POST /api/channels`` ``{"channels": [...]}`` checks a new channel map against the
  profile schema and returns the profile as YAML. Capture is opened with its map, so a new
  map takes effect when the engine is started with the saved profile; the Studio writes
  no files.

The server listens on 127.0.0.1 unless the profile says otherwise, because the page shows
what the robot hears. A web page from elsewhere cannot drive it from the visitor's
browser: POSTs must carry ``Content-Type: application/json`` (which a cross-site form
cannot send without a CORS preflight that this server never answers) and, if the browser
names an ``Origin``, it must be this server.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from collections.abc import Callable, Mapping
from importlib import resources
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ral.core.errors import ConfigError, RalError
from ral.core.events import event_to_json

if TYPE_CHECKING:
    from ral.engine.capture import CaptureResult
    from ral.engine.engine import Engine

__all__ = ["LiveAnalyser", "StudioBridge"]

log = logging.getLogger(__name__)

_FFT = 512
_BANDS = 128
_FLOOR_DB = -100.0


def _db(power: float) -> float:
    return round(10 * float(np.log10(power)), 1) if power > 1e-10 else _FLOOR_DB


class LiveAnalyser:
    """Turns processed frames into ``meters`` messages, one per ``frames_per_message``.

    Levels are mean power over the interval in dBFS. The spectrum is a Hann-windowed
    512-point FFT of the latest clean audio, folded into 128 equal bands (62.5 Hz each
    at 16 kHz), in dB.
    """

    def __init__(self, frames_per_message: int = 10) -> None:
        self.frames_per_message = frames_per_message
        self._window = np.hanning(_FFT)
        self._tail: npt.NDArray[np.float32] = np.zeros(_FFT, np.float32)
        self._reset()

    def _reset(self) -> None:
        self._frames = 0
        self._raw: npt.NDArray[np.float64] | None = None
        self._clean = 0.0
        self._ref: list[float] = []

    def feed(self, result: CaptureResult) -> dict[str, Any] | None:
        """Add one frame; returns a message when the interval is complete."""
        raw = result.raw.data
        power = np.mean(np.square(raw, dtype=np.float64), axis=0)
        self._raw = (
            power if self._raw is None or len(self._raw) != len(power) else self._raw + power
        )
        clean = result.clean.data[:, 0]
        self._clean += float(np.mean(np.square(clean, dtype=np.float64)))
        if "ref_db" in result.ctx.info:
            self._ref.append(result.ctx.info["ref_db"])
        self._tail = np.concatenate([self._tail, clean])[-_FFT:]
        self._frames += 1
        if self._frames < self.frames_per_message:
            return None
        n = self._frames
        spectrum = np.abs(np.fft.rfft(self._tail * self._window))[:-1] ** 2 / _FFT
        bands = spectrum.reshape(_BANDS, -1).mean(axis=1)
        msg = {
            "t_ns": result.raw.t_ns,
            "rate": result.raw.fmt.sample_rate,
            "raw_db": [_db(p / n) for p in (self._raw if self._raw is not None else [])],
            "roles": [r.value for r in result.raw.channel_map.roles],
            "clean_db": _db(self._clean / n),
            "ref_db": round(max(self._ref), 1) if self._ref else None,
            "speaking": result.ctx.robot_speaking,
            "spectrum": [round(max(_FLOOR_DB, 10 * float(np.log10(b + 1e-12)))) for b in bands],
        }
        self._reset()
        return msg


class _Client:
    """One SSE connection; a slow browser loses its oldest messages."""

    def __init__(self, maxsize: int = 100) -> None:
        self.queue: deque[str] = deque(maxlen=maxsize)
        self.ready = asyncio.Event()

    def send(self, kind: str, data: Mapping[str, Any]) -> None:
        self.queue.append(f"event: {kind}\ndata: {json.dumps(data)}\n\n")
        self.ready.set()


class StudioBridge:
    """Serves the Studio page and its API for one engine."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8770) -> None:
        self.host = host
        self.port = port
        self.engine: Engine | None = None
        self.analyser = LiveAnalyser()
        self._clients: set[_Client] = set()
        self._server: asyncio.Server | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._closers: list[Callable[[], None]] = []

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> StudioBridge:
        """Build from ``bridges.studio`` (``host``, ``port``).

        Raises:
            ConfigError: On unknown keys.
        """
        unknown = set(cfg) - {"host", "port"}
        if unknown:
            raise ConfigError(f"bridges.studio has unknown keys {sorted(unknown)}.")
        return cls(str(cfg.get("host", "127.0.0.1")), int(cfg.get("port", 8770)))

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    async def attach(self, engine: Engine) -> None:
        self.engine = engine
        frames = engine.subscribe(maxsize=100)
        events = engine.bus.subscribe()
        self._closers = [frames.close, events.close]

        async def meters() -> None:
            async for result in frames:
                msg = self.analyser.feed(result)
                if msg is not None:
                    self._broadcast("meters", msg)

        async def bus() -> None:
            async for event in events:
                self._broadcast("event", event_to_json(event))

        self._tasks = [
            asyncio.create_task(meters(), name="ral-studio-meters"),
            asyncio.create_task(bus(), name="ral-studio-events"),
        ]
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        if self._server.sockets:
            self.port = int(self._server.sockets[0].getsockname()[1])
        log.info("Studio on %s", self.url)

    async def detach(self) -> None:
        for close in self._closers:
            close()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._server is not None:
            self._server.close()
            for client in self._clients:
                client.ready.set()  # wake the SSE writers so they see the closed server
            self._clients.clear()
            await self._server.wait_closed()
            self._server = None

    def _broadcast(self, kind: str, data: Mapping[str, Any]) -> None:
        for client in self._clients:
            client.send(kind, data)

    # HTTP

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5.0)
            lines = head.decode("latin-1").split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            headers = {
                k.strip().lower(): v.strip()
                for k, _, v in (line.partition(":") for line in lines[1:] if line)
            }
            path = target.split("?", 1)[0]
            length = int(headers.get("content-length", "0"))
            body = await reader.readexactly(min(length, 65536)) if length else b""
            if method == "GET" and path == "/events":
                await self._sse(writer)
                return
            status, ctype, payload = self._route(method, path, headers, body)
            await self._respond(writer, status, ctype, payload)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError, ValueError):
            pass  # a client that hung up or did not speak HTTP
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    async def _respond(
        self, writer: asyncio.StreamWriter, status: str, ctype: str, payload: bytes
    ) -> None:
        writer.write(
            f"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {len(payload)}\r\n"
            "Cache-Control: no-store\r\nConnection: close\r\n\r\n".encode()
            + payload
        )
        await writer.drain()

    def _route(
        self, method: str, path: str, headers: Mapping[str, str], body: bytes
    ) -> tuple[str, str, bytes]:
        if method == "GET" and path == "/":
            page = resources.files("ral.studio").joinpath("page.html").read_bytes()
            return "200 OK", "text/html; charset=utf-8", page
        if method == "GET" and path == "/api/state":
            return _json(self.state())
        if method == "POST" and path in ("/api/echo", "/api/channels"):
            if headers.get("content-type", "").split(";")[0].strip() != "application/json":
                return _json({"error": "Content-Type must be application/json"}, "415")
            origin = headers.get("origin")
            if origin is not None and origin not in self._own_origins(headers):
                return _json({"error": f"origin {origin} may not control the Studio"}, "403")
            try:
                request = json.loads(body or b"{}")
                if path == "/api/echo":
                    assert self.engine is not None
                    self.engine.set_echo_mode(str(request.get("mode")))
                    return _json(self.state())
                return _json({"yaml": self.profile_with_channels(request.get("channels"))})
            except (RalError, ValueError, TypeError, AttributeError) as exc:
                return _json({"error": str(exc)}, "400")
        return _json({"error": "not found"}, "404")

    def _own_origins(self, headers: Mapping[str, str]) -> set[str]:
        host = headers.get("host", f"{self.host}:{self.port}")
        return {f"http://{host}", f"http://{self.host}:{self.port}"}

    async def _sse(self, writer: asyncio.StreamWriter) -> None:
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\n"
            b"Connection: keep-alive\r\n\r\nretry: 1000\n\n"
        )
        await writer.drain()
        client = _Client()
        self._clients.add(client)
        client.send("state", self.state())
        try:
            while self._server is not None and self._server.is_serving():
                await client.ready.wait()
                client.ready.clear()
                while client.queue:
                    writer.write(client.queue.popleft().encode())
                await writer.drain()
        finally:
            self._clients.discard(client)

    # API

    def state(self) -> dict[str, Any]:
        """What the page shows outside the live meters."""
        assert self.engine is not None
        e = self.engine
        source = e.source
        snapshot = e.health.snapshot() if e.health else {}
        return {
            "session_id": e.session_id,
            "robot": e.profile.robot.model,
            "echo_mode": e.profile.echo.mode,
            "echo_modes": ["hardware", "software_ref", "loopback", "gate", "off"],
            "capture": {
                "backend": e.profile.capture.backend,
                "format": str(source.fmt) if source else None,
                "roles": [r.value for r in source.channel_map.roles] if source else [],
            },
            "health": snapshot,
        }

    def profile_with_channels(self, channels: object) -> str:
        """The running profile with a new ``capture.channels``, as YAML.

        Raises:
            ConfigError: If the roles are not valid or do not match the channel count.
        """
        import yaml

        from ral.profiles.schema import Profile

        assert self.engine is not None
        if not isinstance(channels, list) or not all(isinstance(c, str) for c in channels):
            raise ConfigError("channels must be a list of role names, one per capture channel.")
        source = self.engine.source
        if source is not None and len(channels) != source.fmt.channels:
            raise ConfigError(
                f"The capture device has {source.fmt.channels} channels; give one role each, "
                f"not {len(channels)}."
            )
        data = self.engine.profile.model_dump(mode="json")
        data["capture"]["channels"] = channels
        try:
            Profile.model_validate(data)
        except ValueError as exc:  # pydantic's ValidationError is a ValueError
            raise ConfigError(f"The channel map is not valid: {exc}") from None
        return str(yaml.safe_dump(data, sort_keys=False))


def _json(data: Any, status: str = "200") -> tuple[str, str, bytes]:
    reason = {"200": "OK", "400": "Bad Request", "403": "Forbidden", "404": "Not Found"}
    reason["415"] = "Unsupported Media Type"
    return f"{status} {reason[status]}", "application/json", json.dumps(data).encode()
