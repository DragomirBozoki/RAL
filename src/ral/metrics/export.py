"""Metrics export: a Prometheus endpoint and an OpenTelemetry (OTLP/HTTP JSON) pusher.

Both are bridges (``bridges: {prometheus: {...}}`` or ``bridges: {otlp: {...}}``), so they
run only when a profile or ``ral run --bridge`` names them; the core sends nothing by
default (CLAUDE.md, privacy). Neither needs a dependency:

- **Prometheus** scrapes text exposition format 0.0.4 from ``GET /metrics`` on a small
  asyncio HTTP server, bound to 127.0.0.1 unless configured otherwise
  (https://prometheus.io/docs/instrumenting/exposition_formats/).
- **OTLP** posts ``ExportMetricsServiceRequest`` and ``ExportTraceServiceRequest`` as JSON
  to ``/v1/metrics`` and ``/v1/traces`` (OTLP 1.x, "JSON Protobuf Encoding": lowerCamelCase
  keys, 64-bit integers as decimal strings, enums as integers;
  https://opentelemetry.io/docs/specs/otlp/). Every turn becomes a trace with one span per
  stage (hangover, agent, playback) under a ``turn`` span.

:class:`MetricsCollector` turns the health snapshot, the event counts and the turn
latencies into one list of :class:`Metric` that both formats render.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ral import __version__
from ral.core.clock import now_ns
from ral.core.errors import ConfigError
from ral.core.events import Event, TurnLatency

if TYPE_CHECKING:
    from ral.engine.engine import Engine

__all__ = [
    "Metric",
    "MetricsCollector",
    "OtlpExporter",
    "Point",
    "PrometheusExporter",
    "otlp_metrics",
    "otlp_turn_trace",
    "render_prometheus",
]

log = logging.getLogger(__name__)

LATENCY_BOUNDS_MS = (100.0, 200.0, 300.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0)
_STAGES = ("total", "hangover", "agent", "playback")
_GAUGES = (
    ("drift_ppm", "ral_drift_ppm", "Clock drift of the speaker against the microphones.", "ppm"),
    (
        "echo_delay_ms",
        "ral_echo_delay_ms",
        "Measured echo delay after the reference's DAC time.",
        "ms",
    ),
    ("lead_ms", "ral_lead_ms", "Reference lead given to echo control.", "ms"),
    ("dsp_us_per_frame", "ral_dsp_us_per_frame", "Average DSP cost per 10 ms frame.", "us"),
)


@dataclass
class Point:
    """One labelled value; a histogram point also has bucket counts (not cumulative)."""

    labels: dict[str, str]
    value: float
    count: int = 0
    buckets: list[int] | None = None


@dataclass
class Metric:
    """A metric in a format-neutral shape. ``name`` is the Prometheus name."""

    name: str
    kind: Literal["gauge", "counter", "histogram"]
    help: str
    unit: str
    points: list[Point] = field(default_factory=list)
    bounds: tuple[float, ...] = ()

    @property
    def otel_name(self) -> str:
        """``ral_xruns_total`` becomes ``ral.xruns``."""
        return self.name.removesuffix("_total").replace("_", ".", 1)


@dataclass
class _Histogram:
    buckets: list[int] = field(default_factory=lambda: [0] * (len(LATENCY_BOUNDS_MS) + 1))
    count: int = 0
    total: float = 0.0

    def add(self, value: float) -> None:
        i = next((k for k, b in enumerate(LATENCY_BOUNDS_MS) if value <= b), -1)
        self.buckets[i] += 1
        self.count += 1
        self.total += value


class MetricsCollector:
    """Counts events and turn latencies; :meth:`collect` adds the health snapshot.

    Args:
        snapshot: Returns :meth:`ral.metrics.health.HealthMonitor.snapshot`.
        session_id: Reported on ``ral_info``.
    """

    def __init__(self, snapshot: Callable[[], dict[str, Any]], session_id: str = "") -> None:
        self.snapshot = snapshot
        self.session_id = session_id
        self.events: Counter[str] = Counter()
        self.latency = {stage: _Histogram() for stage in _STAGES}

    def on_event(self, event: Event) -> None:
        """Bus listener."""
        self.events[type(event).__name__] += 1
        if isinstance(event, TurnLatency):
            for stage in _STAGES:
                self.latency[stage].add(getattr(event, f"{stage}_ms"))

    def collect(self) -> list[Metric]:
        snap = self.snapshot()
        out = [
            Metric(
                "ral_info",
                "gauge",
                "RAL version and session.",
                "1",
                [Point({"version": __version__, "session_id": self.session_id}, 1.0)],
            ),
            Metric(
                "ral_frames_total",
                "counter",
                "Engine frames captured.",
                "1",
                [Point({}, float(snap.get("frames", 0)))],
            ),
            Metric(
                "ral_xruns_total",
                "counter",
                "Lost or late audio per stream and kind.",
                "1",
                [
                    Point(_xrun_labels(name), float(n))
                    for name, n in sorted(snap.get("xruns", {}).items())
                ],
            ),
            Metric(
                "ral_clipping_events_total",
                "counter",
                "Seconds of a channel that clipped.",
                "1",
                [Point({}, float(snap.get("clipping_events", 0)))],
            ),
            Metric(
                "ral_events_total",
                "counter",
                "Events published on the bus, by type.",
                "1",
                [Point({"type": t}, float(n)) for t, n in sorted(self.events.items())],
            ),
            Metric(
                "ral_capture_alive",
                "gauge",
                "1 while capture delivers audio.",
                "1",
                [Point({}, 1.0 if snap.get("capture_alive", True) else 0.0)],
            ),
        ]
        floors = snap.get("noise_floor_db", [])
        dead = set(snap.get("dead_channels", []))
        out.append(
            Metric(
                "ral_noise_floor_dbfs",
                "gauge",
                "Lowest one-second level of the last 30 s.",
                "dBFS",
                [Point({"channel": str(c)}, f) for c, f in enumerate(floors) if f is not None],
            )
        )
        out.append(
            Metric(
                "ral_channel_dead",
                "gauge",
                "1 if a capture channel carries nothing.",
                "1",
                [Point({"channel": str(c)}, float(c in dead)) for c in range(len(floors))],
            )
        )
        for key, name, help_, unit in _GAUGES:
            value = snap.get(key)
            if value is not None:
                out.append(Metric(name, "gauge", help_, unit, [Point({}, float(value))]))
        out.append(
            Metric(
                "ral_turn_latency_ms",
                "histogram",
                "Turn latency per stage, end of user speech to robot audio at the DAC.",
                "ms",
                [
                    Point({"stage": s}, h.total, h.count, list(h.buckets))
                    for s, h in self.latency.items()
                ],
                LATENCY_BOUNDS_MS,
            )
        )
        return out


def _xrun_labels(name: str) -> dict[str, str]:
    stream, _, kind = name.partition(".")
    return {"stream": stream, "kind": kind}


# Prometheus


def _labels(labels: Mapping[str, str]) -> str:
    if not labels:
        return ""

    def esc(v: str) -> str:
        return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    return "{" + ",".join(f'{k}="{esc(v)}"' for k, v in labels.items()) + "}"


def _num(v: float) -> str:
    return repr(float(v)) if v == v else "NaN"


def render_prometheus(metrics: list[Metric]) -> str:
    """Text exposition format 0.0.4."""
    lines: list[str] = []
    for m in metrics:
        lines.append(f"# HELP {m.name} {m.help}")
        lines.append(f"# TYPE {m.name} {m.kind}")
        for p in m.points:
            if m.kind != "histogram":
                lines.append(f"{m.name}{_labels(p.labels)} {_num(p.value)}")
                continue
            cumulative = 0
            for bound, n in zip((*m.bounds, float("inf")), p.buckets or [], strict=True):
                cumulative += n
                le = "+Inf" if bound == float("inf") else _num(bound)
                lines.append(f"{m.name}_bucket{_labels({**p.labels, 'le': le})} {cumulative}")
            lines.append(f"{m.name}_sum{_labels(p.labels)} {_num(p.value)}")
            lines.append(f"{m.name}_count{_labels(p.labels)} {p.count}")
    return "\n".join(lines) + "\n"


class PrometheusExporter:
    """Serves ``GET /metrics`` (bridge ``prometheus``; default 127.0.0.1:9464).

    The port is the one OpenTelemetry's Prometheus exporter uses by default. Binding to
    another address than loopback is the profile's explicit choice.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9464) -> None:
        self.host = host
        self.port = port
        self.collector: MetricsCollector | None = None
        self._server: asyncio.Server | None = None
        self._remove: Callable[[], None] | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> PrometheusExporter:
        """Build from ``bridges.prometheus`` (``host``, ``port``).

        Raises:
            ConfigError: On unknown keys.
        """
        unknown = set(cfg) - {"host", "port"}
        if unknown:
            raise ConfigError(f"bridges.prometheus has unknown keys {sorted(unknown)}.")
        return cls(str(cfg.get("host", "127.0.0.1")), int(cfg.get("port", 9464)))

    async def attach(self, engine: Engine) -> None:
        self.collector = _collector(engine)
        self._remove = engine.bus.add_listener(self.collector.on_event)
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        sockets = self._server.sockets
        if sockets:
            self.port = int(sockets[0].getsockname()[1])
        log.info("Prometheus metrics on http://%s:%d/metrics", self.host, self.port)

    async def detach(self) -> None:
        if self._remove is not None:
            self._remove()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5.0)
            method, path, *_ = request.split(b"\r\n", 1)[0].decode("latin-1").split(" ")
            if method == "GET" and path.split("?", 1)[0] == "/metrics":
                assert self.collector is not None
                body = render_prometheus(self.collector.collect()).encode()
                status, ctype = "200 OK", "text/plain; version=0.0.4; charset=utf-8"
            else:
                body, status, ctype = b"not found\n", "404 Not Found", "text/plain"
            writer.write(
                f"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError, ValueError):
            pass  # a client that hung up or spoke something else than HTTP
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()


def _collector(engine: Engine) -> MetricsCollector:
    health = engine.health
    return MetricsCollector(health.snapshot if health else dict, engine.session_id)


# OpenTelemetry (OTLP/HTTP JSON)

_CUMULATIVE = 2  # AggregationTemporality.AGGREGATION_TEMPORALITY_CUMULATIVE
_SPAN_INTERNAL = 1  # SpanKind.SPAN_KIND_INTERNAL


def _attrs(labels: Mapping[str, str]) -> list[dict[str, Any]]:
    return [{"key": k, "value": {"stringValue": v}} for k, v in labels.items()]


def _resource(session_id: str) -> dict[str, Any]:
    return {
        "attributes": _attrs(
            {"service.name": "ral", "service.version": __version__, "ral.session_id": session_id}
        )
    }


_SCOPE = {"name": "ral", "version": __version__}


def otlp_metrics(
    metrics: list[Metric], session_id: str, start_unix_ns: int, time_unix_ns: int
) -> dict[str, Any]:
    """An ``ExportMetricsServiceRequest`` in OTLP JSON."""
    out = []
    start, now = str(start_unix_ns), str(time_unix_ns)
    for m in metrics:
        entry: dict[str, Any] = {"name": m.otel_name, "unit": m.unit, "description": m.help}
        if m.kind == "histogram":
            entry["histogram"] = {
                "aggregationTemporality": _CUMULATIVE,
                "dataPoints": [
                    {
                        "attributes": _attrs(p.labels),
                        "startTimeUnixNano": start,
                        "timeUnixNano": now,
                        "count": str(p.count),
                        "sum": p.value,
                        "bucketCounts": [str(n) for n in p.buckets or []],
                        "explicitBounds": list(m.bounds),
                    }
                    for p in m.points
                ],
            }
        else:
            points = [
                {"attributes": _attrs(p.labels), "timeUnixNano": now, "asDouble": p.value}
                for p in m.points
            ]
            if m.kind == "counter":
                for p in points:
                    p["startTimeUnixNano"] = start
                entry["sum"] = {
                    "aggregationTemporality": _CUMULATIVE,
                    "isMonotonic": True,
                    "dataPoints": points,
                }
            else:
                entry["gauge"] = {"dataPoints": points}
        out.append(entry)
    return {
        "resourceMetrics": [
            {
                "resource": _resource(session_id),
                "scopeMetrics": [{"scope": _SCOPE, "metrics": out}],
            }
        ]
    }


def otlp_turn_trace(turn: TurnLatency, wall_offset_ns: int) -> list[dict[str, Any]]:
    """One turn as OTLP spans: a ``turn`` span with ``hangover``, ``agent`` and
    ``playback`` children, laid end to end. ``wall_offset_ns`` maps the RAL clock to Unix
    time (the audio path never needs it)."""
    trace_id = os.urandom(16).hex()
    root_id = os.urandom(8).hex()
    t0 = turn.t_ns + wall_offset_ns

    def span(name: str, start: float, ms: float, span_id: str, parent: str = "") -> dict[str, Any]:
        s: dict[str, Any] = {
            "traceId": trace_id,
            "spanId": span_id,
            "name": name,
            "kind": _SPAN_INTERNAL,
            "startTimeUnixNano": str(t0 + round(start * 1e6)),
            "endTimeUnixNano": str(t0 + round((start + ms) * 1e6)),
            "attributes": _attrs({"ral.tag": turn.tag}),
        }
        if parent:
            s["parentSpanId"] = parent
        return s

    spans = [span("turn", 0.0, turn.total_ms, root_id)]
    start = 0.0
    for stage in ("hangover", "agent", "playback"):
        ms = float(getattr(turn, f"{stage}_ms"))
        spans.append(span(stage, start, ms, os.urandom(8).hex(), root_id))
        start += ms
    return spans


class OtlpExporter:
    """Pushes metrics every ``interval_s`` and turn traces to an OTLP/HTTP collector.

    Bridge ``otlp``: ``endpoint`` (default ``http://127.0.0.1:4318``, a local collector),
    ``interval_s`` (10), ``headers`` (for example an API key for a hosted backend, given
    as ``${ENV_VAR}`` in the profile) and ``traces`` (true). A collector that is down
    costs a warning, not the session.
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:4318",
        *,
        interval_s: float = 10.0,
        headers: Mapping[str, str] | None = None,
        traces: bool = True,
        timeout_s: float = 5.0,
        post: Callable[[str, bytes, Mapping[str, str]], None] | None = None,
    ) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ConfigError(
                f"bridges.otlp.endpoint must be an http:// or https:// URL, not {endpoint!r}."
            )
        if interval_s <= 0:
            raise ConfigError("bridges.otlp.interval_s must be positive.")
        self.endpoint = endpoint.rstrip("/")
        self.interval_s = interval_s
        self.headers = dict(headers or {})
        self.traces = traces
        self.timeout_s = timeout_s
        self.post = post or self._post
        self.collector: MetricsCollector | None = None
        self.session_id = ""
        self.sent = 0
        self.failed = 0
        self._spans: list[dict[str, Any]] = []
        self._start_unix_ns = 0
        self._task: asyncio.Task[None] | None = None
        self._removers: list[Callable[[], None]] = []

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> OtlpExporter:
        """Build from ``bridges.otlp``.

        Raises:
            ConfigError: On unknown keys or an invalid endpoint.
        """
        unknown = set(cfg) - {"endpoint", "interval_s", "headers", "traces"}
        if unknown:
            raise ConfigError(f"bridges.otlp has unknown keys {sorted(unknown)}.")
        return cls(
            str(cfg.get("endpoint", "http://127.0.0.1:4318")),
            interval_s=float(cfg.get("interval_s", 10.0)),
            headers={str(k): str(v) for k, v in (cfg.get("headers") or {}).items()},
            traces=bool(cfg.get("traces", True)),
        )

    async def attach(self, engine: Engine) -> None:
        self.collector = _collector(engine)
        self.session_id = engine.session_id
        self._start_unix_ns = time.time_ns()
        self._removers = [engine.bus.add_listener(self.collector.on_event)]
        if self.traces:
            self._removers.append(engine.bus.add_listener(self._on_turn))
        self._task = asyncio.create_task(self._loop(), name="ral-otlp")
        log.info("OTLP export to %s every %.0f s", self.endpoint, self.interval_s)

    async def detach(self) -> None:
        for remove in self._removers:
            remove()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
            await self.push()  # the last numbers of the session

    def _on_turn(self, event: Event) -> None:
        if isinstance(event, TurnLatency):
            self._spans.extend(otlp_turn_trace(event, time.time_ns() - now_ns()))

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval_s)
            await self.push()

    async def push(self) -> None:
        """Send the current metrics and pending spans now."""
        assert self.collector is not None
        requests = [
            (
                "/v1/metrics",
                otlp_metrics(
                    self.collector.collect(), self.session_id, self._start_unix_ns, time.time_ns()
                ),
            )
        ]
        if self._spans:
            spans, self._spans = self._spans, []
            requests.append(
                (
                    "/v1/traces",
                    {
                        "resourceSpans": [
                            {
                                "resource": _resource(self.session_id),
                                "scopeSpans": [{"scope": _SCOPE, "spans": spans}],
                            }
                        ]
                    },
                )
            )
        headers = {"Content-Type": "application/json", **self.headers}
        for path, payload in requests:
            body = json.dumps(payload).encode()
            try:
                # urllib blocks; a thread keeps the loop (and barge-in) responsive.
                await asyncio.to_thread(self.post, self.endpoint + path, body, headers)
                self.sent += 1
            except (OSError, urllib.error.URLError) as exc:
                self.failed += 1
                level = logging.WARNING if self.failed == 1 else logging.DEBUG
                log.log(level, "OTLP export to %s%s failed: %s", self.endpoint, path, exc)

    def _post(self, url: str, body: bytes, headers: Mapping[str, str]) -> None:
        req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            resp.read()
