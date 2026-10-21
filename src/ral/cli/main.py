"""Entry point for the ``ral`` command."""

from __future__ import annotations

import typer
from rich.console import Console

from ral import __version__
from ral.cli import models, profile
from ral.cli.bench import bench
from ral.cli.doctor import doctor
from ral.cli.probe import probe
from ral.cli.run import run

app = typer.Typer(
    name="ral",
    help="RAL, the Robot Audio Layer.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"ral {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """RAL, the Robot Audio Layer."""


app.command()(doctor)
app.command()(run)
app.command()(bench)
app.command()(probe)
app.add_typer(models.app, name="models")
app.add_typer(profile.app, name="profile")
