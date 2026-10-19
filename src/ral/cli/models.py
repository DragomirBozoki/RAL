"""``ral models``: list and download model files."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ral.core.errors import RalError
from ral.models import MODELS, fetch_model, models_dir

app = typer.Typer(help="List and download model files (nothing is bundled or uploaded).")
console = Console()


@app.command("list")
def list_models() -> None:
    """Show known models and whether they are downloaded."""
    table = Table(title=f"Models in {models_dir()}")
    for col in ("name", "description", "licence", "size", "downloaded"):
        table.add_column(col)
    for spec in MODELS.values():
        present = (models_dir() / spec.filename).is_file()
        table.add_row(
            spec.name,
            spec.description,
            spec.licence,
            f"{spec.size / 1e6:.1f} MB",
            "yes" if present else "no",
        )
    console.print(table)


@app.command("fetch")
def fetch(
    names: list[str] | None = typer.Argument(None, help="Models to fetch; all if omitted."),
    force: bool = typer.Option(False, help="Download again even if a verified copy exists."),
) -> None:
    """Download models from their official source and verify their checksums."""
    for name in names or list(MODELS):
        try:
            path = fetch_model(name, force=force)
        except RalError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from None
        console.print(f"[green]ok[/green] {name}: {path}")
