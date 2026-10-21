"""``ral probe``: play test sweeps, map the channels, measure delay and drift, write a profile."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from ral.core.errors import RalError

console = Console()


def probe(
    input_device: str | None = typer.Option(None, "--input", "-i", help="Capture device name."),
    output_device: str | None = typer.Option(None, "--output", "-o", help="Playback device name."),
    robot: str = typer.Option("my-robot", help="robot.model in the written profile."),
    out: Path = typer.Option(Path("my-robot.yaml"), "--out", help="Profile to write."),
    report: Path = typer.Option(
        Path("probe-report.html"), "--report", help="HTML report to write."
    ),
    level_db: float = typer.Option(-20.0, "--level", help="Sweep level in dBFS (capped at -6)."),
    drift_s: float = typer.Option(
        0.0, "--drift-s", help="Repeat sweeps over this long to measure drift."
    ),
    quiet: bool = typer.Option(
        False, "--quiet", help="Make no sound: measure the noise floor only."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before playing sound."),
    simulate: bool = typer.Option(
        False, "--simulate", help="Probe a simulated robot (no devices)."
    ),
) -> None:
    """Measure the audio devices and write a profile (DESIGN section 6)."""
    from ral.probe.run import ProbeSettings

    settings = ProbeSettings(level_db=min(level_db, -6.0), drift_s=drift_s, quiet=quiet)
    if not quiet and not simulate and not yes:
        console.print(
            f"[yellow]The probe plays test sweeps at {settings.level_db:.0f} dBFS through "
            f"{output_device or 'the default output'}. Keep the volume moderate and warn "
            f"people nearby.[/yellow]"
        )
        if not typer.confirm("Play the sweeps?", default=False):
            raise typer.Exit(1)
    try:
        result, capture, playback = asyncio.run(
            _simulate(settings) if simulate else _devices(settings, input_device, output_device)
        )
    except RalError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    _show(result)
    from ral.probe.run import probe_profile, profile_yaml, write_report

    write_report(result, report, f"ral probe: {robot}")
    console.print(f"Report: {report}")
    if quiet:
        console.print("Quiet mode: channels were not mapped, so no profile was written.")
        return
    profile = probe_profile(result, robot=robot, capture=capture, playback=playback)
    out.write_text(profile_yaml(result, profile), encoding="utf-8")
    console.print(f"Profile: {out}  (try it: ral run --profile {out})")


async def _devices(
    settings: Any, input_device: str | None, output_device: str | None
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from ral.devicedb import find_entry
    from ral.devices.portaudio import PortAudioBackend
    from ral.engine.engine import Engine
    from ral.probe.io import EngineProbeIO
    from ral.probe.run import run_probe
    from ral.profiles.schema import Profile

    backend = PortAudioBackend()
    cap_dev = (
        backend.find(input_device, "input") if input_device else backend.default_device("input")
    )
    out_dev = (
        backend.find(output_device, "output") if output_device else backend.default_device("output")
    )
    prof = Profile.model_validate(
        {
            "capture": {"backend": "portaudio", "device": cap_dev.name},
            "playback": {"backend": "portaudio", "device": out_dev.name},
            "echo": {"mode": "off"},
            "pipeline": [],
            "barge_in": {"policy": "off"},
            "vad": {"engine": "energy"},
        }
    )
    entry = find_entry(cap_dev)
    same = cap_dev.name.split("(")[0].strip() == out_dev.name.split("(")[0].strip()
    async with Engine(prof) as engine:
        assert engine.source is not None
        fmt = engine.source.fmt
        result = await run_probe(
            EngineProbeIO(engine),
            settings,
            hardware_aec_hint=bool(entry and entry.hardware_aec),
            same_device=same,
        )
    capture = {
        "backend": "portaudio",
        "device": cap_dev.name,
        "format": {"rate": fmt.sample_rate, "channels": fmt.channels, "dtype": "float32"},
    }
    return result, capture, {"backend": "portaudio", "device": out_dev.name}


async def _simulate(settings: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from ral.bench.synthetic import echo_path
    from ral.probe.io import SimulatedProbeIO
    from ral.probe.run import run_probe

    console.print("Simulated robot: two mics 25 ms from the speaker, a loopback, a dead channel.")
    io = SimulatedProbeIO(
        16000,
        ["mic", "mic", "reference", "dead"],
        [echo_path(25, 16000, 1) * 0.3, echo_path(26, 16000, 2) * 0.3],
    )
    result = await run_probe(io, settings)
    capture = {
        "backend": "portaudio",
        "device": "simulated",
        "format": {"rate": 16000, "channels": 4, "dtype": "float32"},
    }
    return result, capture, {"backend": "portaudio", "device": "simulated"}


def _show(result: Any) -> None:
    table = Table(title=f"Channels (suggested echo mode: {result.suggested_mode})")
    for col in ("ch", "role", "noise dBFS", "delay ms", "echo gain dB", "note"):
        table.add_column(col)
    for c in result.channels:
        table.add_row(
            str(c.index),
            c.role,
            f"{c.noise_db:.0f}",
            "-" if c.delay_ms is None else f"{c.delay_ms:.1f}",
            "-" if c.gain_db is None else f"{c.gain_db:.0f}",
            c.note,
        )
    console.print(table)
    for reason in result.reasons:
        console.print(f"  {reason}")
    if result.drift_ppm is not None:
        console.print(f"  clock drift {result.drift_ppm:.1f} ppm")
