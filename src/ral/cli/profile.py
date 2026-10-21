"""``ral profile``: list, show, validate and migrate profiles."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from ral.core.errors import RalError

app = typer.Typer(help="List, show, validate and migrate profiles.")
console = Console()


@app.command("list")
def list_cmd() -> None:
    """Every profile RAL can load by name, and where it comes from."""
    from ral.profiles.registry import list_profiles, profile_dirs

    table = Table(title="Profiles")
    for col in ("name", "robot", "echo mode", "validated by", "source"):
        table.add_column(col)
    for e in list_profiles():
        source = e.source if not e.error else f"{e.source} (unreadable: {e.error})"
        table.add_row(e.name, e.model, e.echo_mode, e.validated_by, source)
    console.print(table)
    console.print("User folders: " + ", ".join(str(d) for d in profile_dirs()))


@app.command()
def show(name: str = typer.Argument(..., help="Profile name or file.")) -> None:
    """Print a profile as RAL reads it, after migrations and with defaults filled in."""
    from ral.profiles import load_profile

    try:
        prof = load_profile(name)
    except RalError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    console.print(yaml.safe_dump(prof.model_dump(mode="json"), sort_keys=False))


@app.command()
def validate(files: list[Path] = typer.Argument(..., help="Profile files to check.")) -> None:
    """Check profile files; exit 1 if any is invalid."""
    from ral.profiles import parse_profile

    bad = 0
    for f in files:
        try:
            parse_profile(f.read_text(encoding="utf-8"), str(f))
            console.print(f"[green]ok[/green] {f}")
        except (RalError, OSError) as exc:
            bad += 1
            console.print(f"[red]invalid[/red] {exc}")
    if bad:
        raise typer.Exit(1)


@app.command()
def migrate(
    file: Path = typer.Argument(..., help="Profile file to upgrade to the current schema."),
    write: bool = typer.Option(False, "--write", help="Rewrite the file (keeps a .bak copy)."),
) -> None:
    """Upgrade an old profile to the current schema version."""
    from ral.profiles.migrations import migrate as run_migrations

    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        new, applied = run_migrations(data, source=str(file))
    except (RalError, OSError, yaml.YAMLError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    if not applied:
        console.print(f"{file} is already at the current schema version.")
        return
    text = yaml.safe_dump(new, sort_keys=False)
    if write:
        file.with_suffix(file.suffix + ".bak").write_text(
            file.read_text(encoding="utf-8"), encoding="utf-8"
        )
        file.write_text(text, encoding="utf-8")
        console.print(f"{file} migrated through versions {applied}; the old file is {file}.bak")
    else:
        console.print(text)
