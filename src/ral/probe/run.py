"""Running the probe and writing its results (profile YAML and HTML report)."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from ral import __version__
from ral.probe.analyze import ProbeResult, analyze
from ral.probe.io import ProbeIO
from ral.probe.signals import exp_sweep, inverse_filter

__all__ = ["ProbeSettings", "probe_profile", "run_probe", "write_report"]


@dataclass
class ProbeSettings:
    """How the probe measures."""

    noise_s: float = 2.0
    sweep_s: float = 2.0
    level_db: float = -20.0
    """Sweep level; capped volume (DESIGN 6: the probe warns and caps)."""
    tail_s: float = 0.8
    drift_s: float = 0.0
    """Repeat the sweep over this many seconds to estimate drift (0: one sweep)."""
    quiet: bool = False
    """Skip everything that makes sound (noise floor only)."""


async def run_probe(
    io: ProbeIO,
    settings: ProbeSettings | None = None,
    *,
    hardware_aec_hint: bool = False,
    same_device: bool = True,
) -> ProbeResult:
    """Measure noise, then sweeps (unless quiet), and analyse."""
    s = settings or ProbeSettings()
    silence = await io.record(s.noise_s)
    rate = io.rate
    sweep = exp_sweep(rate, s.sweep_s, level_db=s.level_db)
    inv = inverse_filter(sweep, rate)
    if s.quiet:
        empty = [np.zeros((len(sweep), io.channels))]
        result = analyze(silence, empty, [0.0], inv, rate)
        for c in result.channels:
            if c.role != "unused" or "dead" not in c.note:
                c.role, c.gain_db, c.delay_ms, c.snr_db, c.drr_db = (
                    "unknown",
                    None,
                    None,
                    None,
                    None,
                )
                c.note = "quiet mode: not measured"
        result.suggested_mode = "software_ref"
        result.reasons = ["quiet mode: no sweep was played, run without --quiet to map channels"]
        return result
    repeats = 1 if s.drift_s <= 0 else max(2, int(s.drift_s // (s.sweep_s + s.tail_s)) + 1)
    sweeps, times = [], []
    elapsed = 0.0
    for _ in range(repeats):
        rec = await io.play_and_record(sweep, s.tail_s)
        sweeps.append(rec)
        times.append(elapsed)
        elapsed += len(rec) / rate
    return analyze(
        silence,
        sweeps,
        times,
        inv,
        rate,
        hardware_aec_hint=hardware_aec_hint,
        same_device=same_device,
    )


def probe_profile(
    result: ProbeResult,
    *,
    robot: str,
    capture: dict[str, Any],
    playback: dict[str, Any],
) -> dict[str, Any]:
    """A profile (as a plain dict, ready for YAML) built from the measurements."""
    roles = ["unused" if r == "unknown" else r for r in result.roles]
    echo: dict[str, Any] = {"mode": result.suggested_mode, "mic": 0, "lead_ms": 20}
    if result.drift_ppm is not None:
        echo["drift_ppm"] = round(result.drift_ppm, 1)
    pipeline = ["aec", "noise_suppression", "high_pass", "vad"]
    profile: dict[str, Any] = {
        "ral": 1,
        "robot": {"model": robot, "validated_by": "community"},
        "capture": {**capture, "channels": roles},
        "playback": playback,
        "echo": echo,
        "pipeline": pipeline,
        "vad": {"engine": "silero"},
        "barge_in": {"policy": "vad+dtd"},
        "recording": {"enabled": False},
    }
    return profile


def profile_yaml(result: ProbeResult, profile: dict[str, Any]) -> str:
    """The profile with a header that records what was measured."""
    lines = [f"# Written by ral probe {__version__}. Measured:"]
    for c in result.channels:
        bits = [f"ch {c.index}: {c.role}", f"noise {c.noise_db:.0f} dBFS"]
        if c.delay_ms is not None and c.role in ("mic", "reference"):
            bits.append(f"delay {c.delay_ms:.1f} ms")
        if c.gain_db is not None and c.role == "mic":
            bits.append(f"echo gain {c.gain_db:.0f} dB")
        if c.note:
            bits.append(c.note)
        lines.append("#   " + ", ".join(bits))
    if result.drift_ppm is not None:
        lines.append(f"#   clock drift {result.drift_ppm:.1f} ppm")
    lines.append(f"#   suggested echo mode {result.suggested_mode}: {'; '.join(result.reasons)}")
    return "\n".join(lines) + "\n" + yaml.safe_dump(profile, sort_keys=False)


def _svg_ir(h: npt.NDArray[np.float64], rate: int, width: int = 560, height: int = 120) -> str:
    n = min(len(h), int(0.3 * rate))
    seg = np.abs(h[:n])
    if seg.max() <= 0:
        return "<svg></svg>"
    step = max(1, n // width)
    peaks = seg[: (n // step) * step].reshape(-1, step).max(axis=1)
    db = 20 * np.log10(np.maximum(peaks / seg.max(), 1e-4))  # 0 .. -80 dB
    pts = " ".join(f"{i * width / len(db):.1f},{(-v / 80) * height:.1f}" for i, v in enumerate(db))
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="impulse response"><polyline fill="none" '
        f'stroke="currentColor" stroke-width="1" points="{pts}"/></svg>'
    )


def write_report(result: ProbeResult, path: str | Path, title: str) -> Path:
    """A self-contained HTML report: the numbers and each channel's impulse response."""
    out = Path(path)
    rows = []
    for c in result.channels:

        def fmt(v: float | None, unit: str) -> str:
            return "-" if v is None else f"{v:.1f} {unit}"

        rows.append(
            "<tr>"
            f"<td>{c.index}</td><td>{html.escape(c.role)}</td><td>{fmt(c.noise_db, 'dBFS')}</td>"
            f"<td>{fmt(c.delay_ms, 'ms')}</td><td>{fmt(c.gain_db, 'dB')}</td>"
            f"<td>{fmt(c.drr_db, 'dB')}</td><td>{html.escape(c.note)}</td>"
            "</tr>"
        )
    irs = "".join(
        f"<h3>Channel {c.index} ({html.escape(c.role)})</h3>"
        + _svg_ir(result.irs[:, c.index], result.rate)
        for c in result.channels
        if result.irs.shape[0]
    )
    drift = "not measured" if result.drift_ppm is None else f"{result.drift_ppm:.1f} ppm"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(title)}</title>
<style>
:root {{ --fg: #1d1d1f; --bg: #ffffff; --line: #d0d0d5; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg: #e8e8ea; --bg: #161618; --line: #3a3a3f; }} }}
body {{ font-family: system-ui, sans-serif; color: var(--fg); background: var(--bg);
       max-width: 860px; margin: 0 auto; padding: 16px; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border-bottom: 1px solid var(--line); padding: 4px 8px; text-align: left; }}
svg {{ color: var(--fg); }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>Suggested echo mode: <b>{html.escape(result.suggested_mode)}</b>
({html.escape("; ".join(result.reasons))}). Clock drift: {drift}. Sample rate {result.rate} Hz.</p>
<table><tr><th>ch</th><th>role</th><th>noise</th><th>delay</th><th>echo gain</th>
<th>direct/reverb</th><th>note</th></tr>{"".join(rows)}</table>
<h2>Impulse responses (first 300 ms, 0 to -80 dB)</h2>{irs}
<p>Written by ral probe {__version__}.</p>
</body></html>
"""
    out.write_text(page, encoding="utf-8")
    return out
