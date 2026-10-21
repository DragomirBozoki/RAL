"""``ral bench``: replay one recorded session through several configurations."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ral.core.errors import RalError

console = Console()

_MODES = ("hardware", "software_ref", "loopback", "gate", "off")


def bench(
    session: Path | None = typer.Argument(
        None, help="A recording made with `ral run --record` (.mcap)."
    ),
    echo: list[str] = typer.Option(
        [], "--echo", "-e", help="Compare this echo mode (repeatable), other settings from --base."
    ),
    config: list[Path] = typer.Option(
        [], "--config", "-c", help="Compare this profile's processing settings (repeatable)."
    ),
    base: str = typer.Option("laptop", "--base", help="Profile that --echo variants start from."),
    vad: str | None = typer.Option(
        None, "--vad", help="Use this VAD engine in every configuration (silero or energy)."
    ),
    demo: bool = typer.Option(
        False, "--demo", help="Bench a synthetic session (no recording needed)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the results as JSON."),
) -> None:
    """Compare echo control, VAD and barge-in settings on the same recorded audio."""
    from ral.bench.report import bench_rows, recorded_latency_ms, rows_as_json
    from ral.engine.stages import build_stages
    from ral.profiles import load_profile
    from ral.record.replay import load_session, replay

    try:
        path = _demo_session() if demo else session
        if path is None:
            console.print("[red]Give a recording, or use --demo.[/red]")
            raise typer.Exit(2)
        inputs = load_session(path)
        variants = []
        for mode in echo or ([] if config else ["off", "software_ref"]):
            if mode not in _MODES:
                console.print(
                    f"[red]Unknown echo mode {mode!r}; use one of {', '.join(_MODES)}.[/red]"
                )
                raise typer.Exit(2)
            prof = load_profile(base)
            prof.echo.mode = mode  # type: ignore[assignment]
            variants.append((f"echo={mode}", prof))
        for cfg in config:
            variants.append((cfg.stem, load_profile(cfg)))
        results = []
        for name, prof in variants:
            if vad is not None:
                prof.vad.engine = vad  # type: ignore[assignment]
            results.append(replay(inputs, build_stages(prof), name=name, lead_ms=prof.echo.lead_ms))
        rows = bench_rows(inputs, results)
    except RalError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    latency = recorded_latency_ms(inputs)
    if as_json:
        print(rows_as_json(rows, latency))
        return
    table = Table(title=f"{path} ({inputs.seconds:.0f} s, {len(inputs.playback)} playback items)")
    for col in (
        "config",
        "ERLE dB*",
        "starts/min of robot speech",
        "barge-ins",
        "echo leaks",
        "us/frame",
    ):
        table.add_column(col)
    for r in rows:
        table.add_row(
            r.name,
            "-" if r.erle_db is None else f"{r.erle_db:.1f}",
            f"{r.starts_per_min:.1f}",
            str(r.barge_ins),
            str(r.echo_leaks),
            f"{r.us_per_frame:.0f}",
        )
    console.print(table)
    console.print(
        "* while the robot spoke, including any double talk, so a lower bound; for gate it "
        "is the gate's attenuation. Speech starts during robot speech bound the self-trigger "
        "rate from above."
    )
    if latency:
        median = sorted(latency)[len(latency) // 2]
        console.print(
            f"Recorded turn latency (agent included): median {median:.0f} ms over "
            f"{len(latency)} turns."
        )


def _demo_session() -> Path:
    from ral.bench.report import write_scene_recording
    from ral.bench.synthetic import echo_scene

    utterances = [(0.5, 3.5), (5.0, 8.0), (9.5, 12.5), (14.0, 17.0), (18.0, 19.5)]
    scene = echo_scene(seconds=20, delay_ms=20, far_spans=utterances, seed=5)
    console.print("Synthetic session: 20 s, the robot says five things, nobody else talks.")
    return write_scene_recording(scene, playback_s=utterances)
