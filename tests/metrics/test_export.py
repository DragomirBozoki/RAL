import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ral.bridges import make_bridge, start_bridges, stop_bridges
from ral.core.bus import EventBus
from ral.core.errors import ConfigError
from ral.core.events import BargeIn, TurnLatency
from ral.metrics.export import (
    MetricsCollector,
    OtlpExporter,
    PrometheusExporter,
    otlp_metrics,
    otlp_turn_trace,
    render_prometheus,
)

SNAP: dict[str, Any] = {
    "frames": 700,
    "xruns": {"capture.overrun": 2},
    "clipping_events": 1,
    "dead_channels": [1],
    "noise_floor_db": [-40.0, None],
    "capture_alive": True,
    "drift_ppm": 12.5,
    "echo_delay_ms": None,
    "lead_ms": 20.0,
    "dsp_us_per_frame": 310.0,
}


def turn(total: float = 900.0) -> TurnLatency:
    return TurnLatency(
        10**12, "s", total_ms=total, hangover_ms=300, agent_ms=total - 400, playback_ms=100, tag="a"
    )


def collector() -> MetricsCollector:
    c = MetricsCollector(lambda: SNAP, 's"1')
    c.on_event(turn(900))
    c.on_event(turn(2500))
    c.on_event(BargeIn(0, "s", interrupted_tag="a"))
    return c


def test_prometheus_text() -> None:
    text = render_prometheus(collector().collect())
    lines = text.splitlines()
    assert "# TYPE ral_xruns_total counter" in lines
    assert 'ral_xruns_total{stream="capture",kind="overrun"} 2.0' in lines
    assert 'ral_info{version="0.0.1.dev0",session_id="s\\"1"} 1.0' in lines
    assert 'ral_noise_floor_dbfs{channel="0"} -40.0' in lines
    assert 'ral_channel_dead{channel="1"} 1.0' in lines
    assert "ral_drift_ppm 12.5" in lines
    assert not any(line.startswith("ral_echo_delay_ms") for line in lines)  # no estimate yet
    assert 'ral_events_total{type="BargeIn"} 1.0' in lines
    # Histogram buckets are cumulative and end with +Inf = count.
    assert 'ral_turn_latency_ms_bucket{stage="total",le="1000.0"} 1' in lines
    assert 'ral_turn_latency_ms_bucket{stage="total",le="3000.0"} 2' in lines
    assert 'ral_turn_latency_ms_bucket{stage="total",le="+Inf"} 2' in lines
    assert 'ral_turn_latency_ms_sum{stage="total"} 3400.0' in lines
    assert 'ral_turn_latency_ms_count{stage="total"} 2' in lines
    assert text.endswith("\n")


def test_otlp_metrics_follow_the_json_encoding() -> None:
    payload = otlp_metrics(collector().collect(), "s", 1_000, 2_000)
    json.dumps(payload)
    (rm,) = payload["resourceMetrics"]
    attrs = {a["key"]: a["value"]["stringValue"] for a in rm["resource"]["attributes"]}
    assert attrs["service.name"] == "ral"
    metrics = {m["name"]: m for m in rm["scopeMetrics"][0]["metrics"]}
    xruns = metrics["ral.xruns"]["sum"]
    assert xruns["aggregationTemporality"] == 2
    assert xruns["isMonotonic"] is True
    (p,) = xruns["dataPoints"]
    assert p["startTimeUnixNano"] == "1000"
    assert p["timeUnixNano"] == "2000"
    assert p["asDouble"] == 2.0
    assert metrics["ral.drift_ppm"]["gauge"]["dataPoints"][0]["asDouble"] == 12.5
    hist = metrics["ral.turn_latency_ms"]["histogram"]
    total = next(
        d for d in hist["dataPoints"] if d["attributes"][0]["value"]["stringValue"] == "total"
    )
    assert total["count"] == "2"
    assert total["sum"] == 3400.0
    assert len(total["bucketCounts"]) == len(total["explicitBounds"]) + 1
    assert sum(int(n) for n in total["bucketCounts"]) == 2


def test_turn_trace_has_one_span_per_stage() -> None:
    spans = otlp_turn_trace(turn(900), wall_offset_ns=5)
    root, *stages = spans
    assert [s["name"] for s in stages] == ["hangover", "agent", "playback"]
    assert all(s["parentSpanId"] == root["spanId"] for s in stages)
    assert len({s["traceId"] for s in spans}) == 1
    assert len(root["traceId"]) == 32
    assert root["startTimeUnixNano"] == str(10**12 + 5)
    assert stages[0]["startTimeUnixNano"] == root["startTimeUnixNano"]
    assert stages[-1]["endTimeUnixNano"] == root["endTimeUnixNano"]  # stages add up


def fake_engine() -> Any:
    bus = EventBus()
    return SimpleNamespace(bus=bus, session_id="s", health=SimpleNamespace(snapshot=lambda: SNAP))


async def test_otlp_exporter_pushes_metrics_and_turns() -> None:
    sent: list[tuple[str, dict[str, Any], Mapping[str, str]]] = []

    def post(url: str, body: bytes, headers: Mapping[str, str]) -> None:
        sent.append((url, json.loads(body), headers))

    exporter = OtlpExporter(
        "http://collector:4318/", interval_s=0.05, headers={"x-key": "k"}, post=post
    )
    engine = fake_engine()
    await exporter.attach(engine)
    engine.bus.publish(turn())
    await asyncio.sleep(0.2)
    await exporter.detach()
    urls = [u for u, _, _ in sent]
    assert urls.count("http://collector:4318/v1/traces") == 1
    assert urls.count("http://collector:4318/v1/metrics") >= 2  # periodic plus the final one
    _, traces, headers = next(s for s in sent if s[0].endswith("traces"))
    assert len(traces["resourceSpans"][0]["scopeSpans"][0]["spans"]) == 4
    assert headers["Content-Type"] == "application/json"
    assert headers["x-key"] == "k"


async def test_a_collector_that_is_down_is_not_fatal() -> None:
    def post(url: str, body: bytes, headers: Mapping[str, str]) -> None:
        raise ConnectionRefusedError("refused")

    exporter = OtlpExporter(post=post)
    await exporter.attach(fake_engine())
    await exporter.push()
    await exporter.detach()
    assert exporter.failed == 2
    assert exporter.sent == 0


def test_config_is_checked() -> None:
    assert isinstance(make_bridge("prometheus", {"port": 0}), PrometheusExporter)
    assert isinstance(make_bridge("otlp", {"interval_s": 5}), OtlpExporter)
    with pytest.raises(ConfigError, match="unknown keys"):
        make_bridge("prometheus", {"path": "/m"})
    with pytest.raises(ConfigError, match="http"):
        make_bridge("otlp", {"endpoint": "collector:4318"})


async def http_get(port: int, path: str) -> tuple[str, str]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(f"GET {path} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
    data = await asyncio.wait_for(reader.read(), 5)
    writer.close()
    head, _, body = data.decode().partition("\r\n\r\n")
    return head, body


async def test_prometheus_endpoint_on_a_running_engine(tmp_path: Path) -> None:
    from ral.engine.engine import Engine
    from ral.profiles import parse_profile

    mic = (Path(__file__).resolve().parents[1] / "data" / "speech_16k.wav").as_posix()
    prof = parse_profile(
        "ral: 1\n"
        f"capture: {{backend: file, path: '{mic}', realtime: false}}\n"
        f"playback: {{backend: file, path: '{(tmp_path / 'o.wav').as_posix()}', "
        "realtime: false}\n"
        "echo: {mode: off}\npipeline: [aec]\nbridges: {prometheus: {port: 0}}\n"
    )
    engine = Engine(prof)
    async with engine:
        bridges = await start_bridges(engine, prof.bridges)
        try:
            (bridge,) = bridges
            assert isinstance(bridge, PrometheusExporter)
            await asyncio.wait_for(engine.capture_done.wait(), 30)
            await asyncio.sleep(0.2)
            head, body = await http_get(bridge.port, "/metrics")
            assert head.startswith("HTTP/1.1 200")
            assert "version=0.0.4" in head
            frames = next(ln for ln in body.splitlines() if ln.startswith("ral_frames_total"))
            assert float(frames.split()[1]) > 0
            head, _ = await http_get(bridge.port, "/other")
            assert head.startswith("HTTP/1.1 404")
        finally:
            await stop_bridges(bridges)
