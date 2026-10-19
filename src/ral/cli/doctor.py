"""``ral doctor``: print what RAL can use on this host and why the rest is not usable."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from ral.devices.doctor import DoctorReport, Level, run_doctor
from ral.devices.doctor.host import HostScanner
from ral.devices.doctor.portaudio_scan import PortAudioScanner
from ral.devices.doctor.report import Scanner
from ral.devices.doctor.ros2 import Ros2Scanner
from ral.devices.portaudio import preferred_host_apis

_STYLE = {Level.OK: "green", Level.INFO: "dim", Level.WARN: "yellow", Level.ERROR: "red bold"}
_SUMMARY = {Level.OK: "green", Level.INFO: "green", Level.WARN: "amber", Level.ERROR: "red"}


def doctor(
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
    ros2: bool = typer.Option(True, "--ros2/--no-ros2", help="Check the ROS 2 graph."),
    ros2_wait: float = typer.Option(
        2.0, "--ros2-wait", min=0.0, help="Seconds to wait for ROS 2 discovery."
    ),
    all_devices: bool = typer.Option(
        False, "--all-devices", help="List devices of every host API, not only the preferred."
    ),
) -> None:
    """Check the host, its audio devices and the ROS 2 graph. Plays and records nothing.

    Exits with status 1 when a check finds an error.
    """
    scanners: list[Scanner] = [HostScanner(), PortAudioScanner()]
    if ros2:
        scanners.append(Ros2Scanner(wait_s=ros2_wait))
    report = run_doctor(scanners)
    console = Console()
    if as_json:
        console.print_json(json.dumps(report.to_dict()))
    else:
        render(report, console, all_devices=all_devices)
    if report.has_errors:
        raise typer.Exit(1)


def render(report: DoctorReport, console: Console, *, all_devices: bool = False) -> None:
    """Print the report as rich tables."""
    findings = Table(title="ral doctor", show_lines=False, expand=True)
    findings.add_column("Area", no_wrap=True)
    findings.add_column("Level", no_wrap=True)
    findings.add_column("Finding", ratio=3)
    findings.add_column("Hint", ratio=2, style="cyan")
    for f in report.findings:
        style = _STYLE[f.level]
        findings.add_row(f.area, f"[{style}]{f.level.value}[/]", f.message, f.hint or "")
    console.print(findings)

    if report.devices:
        console.print(_device_table(report, all_devices))
    if report.topics:
        topics = Table(title="ROS 2 audio topics")
        for col in ("Topic", "Type", "Publisher", "Reliability", "Durability", "Depth"):
            topics.add_column(col)
        for topic in report.topics:
            for pub in topic.publishers or ():
                topics.add_row(
                    topic.name,
                    ", ".join(topic.types),
                    pub.node,
                    pub.reliability,
                    pub.durability,
                    str(pub.depth),
                )
            if not topic.publishers:
                topics.add_row(topic.name, ", ".join(topic.types), "-", "", "", "")
        console.print(topics)

    for area, reason in report.skipped.items():
        console.print(f"[dim]{area}: skipped, {reason}.[/]")
    summary = "  ".join(
        f"{area}: [{_STYLE[level]}]{_SUMMARY[level]}[/]"
        for area, level in report.area_levels().items()
    )
    console.print(f"Summary  {summary}")
    # TODO(M2): suggest an echo mode once the device database knows hardware AEC arrays.


def _device_table(report: DoctorReport, all_devices: bool) -> Table:
    preferred = preferred_host_apis()
    devices = report.devices
    shown = devices if all_devices else [d for d in devices if d.host_api in preferred]
    if not shown:  # nothing under a preferred host API: show everything
        shown = devices
    hidden = len(devices) - len(shown)
    caption = f"{hidden} more under other host APIs; use --all-devices" if hidden else None
    table = Table(title="Audio devices", caption=caption)
    table.add_column("ID", overflow="fold")
    for col in ("Name", "Host API", "In", "Out", "Rate", "Default"):
        table.add_column(col)
    for d in shown:
        defaults = [k.removeprefix("default_") for k in ("default_input", "default_output")]
        marks = ", ".join(k for k in defaults if d.extra.get(f"default_{k}") == "true")
        rate = f"{d.default_sample_rate:g}" if d.default_sample_rate else ""
        table.add_row(
            d.id,
            d.name,
            d.host_api or "",
            str(d.max_input_channels),
            str(d.max_output_channels),
            rate,
            marks,
        )
    return table
