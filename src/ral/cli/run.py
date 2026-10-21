"""``ral run``: run the engine from a profile until Ctrl-C."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from ral.core.clock import now_ns
from ral.core.errors import RalError
from ral.core.events import Event, event_to_json

console = Console()


def run(
    profile: str = typer.Option("laptop", "--profile", "-p", help="Profile file or built-in name."),
    duration: float = typer.Option(0.0, help="Stop after this many seconds (0: run until Ctrl-C)."),
    play: Path | None = typer.Option(None, help="Play this WAV file through RAL once started."),
    json_events: bool = typer.Option(False, "--json", help="Print events as JSON lines."),
    echo: str | None = typer.Option(
        None, "--echo", help="Override the echo mode: software_ref, loopback, gate or off."
    ),
    bridge: list[str] = typer.Option(
        [], "--bridge", "-b", help="Also start this bridge with default settings (repeatable)."
    ),
    record: bool | None = typer.Option(
        None,
        "--record/--no-record",
        help="Record the session to MCAP (default: the profile's recording.enabled, off).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show engine log messages."),
) -> None:
    """Run RAL: capture, echo control, VAD and barge-in, playback, bridges."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    from ral.profiles import load_profile

    try:
        prof = load_profile(profile)
    except RalError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None
    try:
        for name in bridge:
            prof.bridges.setdefault(name, {})
        if echo is not None:
            if echo not in ("software_ref", "loopback", "gate", "off"):
                console.print(f"[red]--echo must be an echo mode, not {echo!r}[/red]")
                raise typer.Exit(2)
            prof.echo.mode = echo  # type: ignore[assignment]
        asyncio.run(_run(prof, duration, play, json_events, record))
    except KeyboardInterrupt:
        pass
    except RalError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None


async def _run(
    prof: object, duration: float, play: Path | None, json_events: bool, record: bool | None
) -> None:
    from ral.bridges import start_bridges, stop_bridges
    from ral.engine.engine import Engine
    from ral.playback.clip import load_wav
    from ral.playback.service import audio_frames
    from ral.profiles.schema import Profile

    assert isinstance(prof, Profile)
    engine = Engine(prof, record=record)
    events = engine.bus.subscribe()
    async with engine:
        bridges = await start_bridges(engine, prof.bridges)
        t0 = now_ns()
        if not json_events:
            console.print(
                f"[green]RAL running[/green] session {engine.session_id}, echo mode "
                f"{prof.echo.mode}. Press Ctrl-C to stop."
            )
        if play is not None:
            assert engine.playback is not None
            engine.playback.play(
                audio_frames(load_wav(play, engine.engine_fmt.sample_rate)), tag="play"
            )

        async def printer() -> None:
            async for event in events:
                _print_event(event, t0, json_events)

        task = asyncio.create_task(printer())
        waits: list[asyncio.Task[Any]] = [asyncio.create_task(engine.capture_done.wait())]
        if duration > 0:
            waits.append(asyncio.create_task(asyncio.sleep(duration)))
        try:
            await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
        finally:
            await stop_bridges(bridges)
            for w in waits:
                w.cancel()
            events.close()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            if not json_events:
                _print_latency(engine)


def _print_event(event: Event, t0: int, as_json: bool) -> None:
    if as_json:
        print(json.dumps(event_to_json(event)), flush=True)
        return
    data = event_to_json(event)
    name = data.pop("type")
    data.pop("session_id", None)
    t = (data.pop("t_ns") - t0) / 1e9
    extra = " ".join(f"{k}={v}" for k, v in data.items() if not k.endswith("_ns"))
    console.print(f"{t:8.3f}s  [bold]{name}[/bold] {extra}")


def _print_latency(engine: object) -> None:
    from rich.table import Table

    from ral.engine.engine import Engine

    assert isinstance(engine, Engine)
    stats = engine.latency.summary()
    if not stats:
        return
    table = Table(title=f"Turn latency over {len(engine.latency.turns)} turns (ms)")
    for col in ("stage", "median", "p90"):
        table.add_column(col)
    for st in stats:
        table.add_row(st.name, f"{st.median_ms:.0f}", f"{st.p90_ms:.0f}")
    console.print(table)
