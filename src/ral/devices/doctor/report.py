"""Doctor findings and report, and the runner that collects them.

``ral doctor`` only looks; it never plays or records (DESIGN section 6). Every check
returns structured findings with an area, a level, a message and a hint, so the CLI can
print a table or JSON from the same data.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol

from ral.core.types import DeviceInfo

__all__ = [
    "DoctorReport",
    "Finding",
    "Level",
    "PublisherQos",
    "Scanner",
    "TopicInfo",
    "run_doctor",
]

logger = logging.getLogger(__name__)


class Level(Enum):
    """How a finding affects RAL. ``INFO`` is context and never changes an area's colour."""

    OK = "ok"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


_SEVERITY = {Level.INFO: 0, Level.OK: 1, Level.WARN: 2, Level.ERROR: 3}


@dataclass(frozen=True)
class Finding:
    """One observation about the host."""

    area: str
    level: Level
    message: str
    hint: str | None = None


@dataclass(frozen=True)
class PublisherQos:
    """A ROS 2 publisher and the QoS it offers."""

    node: str
    reliability: str
    durability: str
    history: str
    depth: int


@dataclass(frozen=True)
class TopicInfo:
    """An audio-like ROS 2 topic."""

    name: str
    types: tuple[str, ...]
    publishers: tuple[PublisherQos, ...]


@dataclass
class DoctorReport:
    """Everything one doctor run found."""

    findings: list[Finding] = field(default_factory=list)
    devices: list[DeviceInfo] = field(default_factory=list)
    topics: list[TopicInfo] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    """Areas not checked, with the reason."""

    def add(self, area: str, level: Level, message: str, hint: str | None = None) -> None:
        """Append a finding."""
        self.findings.append(Finding(area, level, message, hint))

    def area_levels(self) -> dict[str, Level]:
        """The worst level per area, in the order areas first appear."""
        levels: dict[str, Level] = {}
        for f in self.findings:
            current = levels.get(f.area)
            if current is None or _SEVERITY[f.level] > _SEVERITY[current]:
                levels[f.area] = f.level
        return levels

    @property
    def has_errors(self) -> bool:
        """True if any finding is an error."""
        return any(f.level is Level.ERROR for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready view of the report."""
        return {
            "summary": {area: level.value for area, level in self.area_levels().items()},
            "findings": [{**asdict(f), "level": f.level.value} for f in self.findings],
            "devices": [asdict(d) for d in self.devices],
            "ros2_topics": [asdict(t) for t in self.topics],
            "skipped": dict(self.skipped),
        }


class Scanner(Protocol):
    """One read-only check that adds to the report."""

    area: str

    def scan(self, report: DoctorReport) -> None: ...


def run_doctor(scanners: Iterable[Scanner] | None = None) -> DoctorReport:
    """Run every scanner and collect the report; a crashing scanner becomes an error finding.

    Args:
        scanners: The checks to run; by default host, PortAudio and ROS 2.
    """
    if scanners is None:
        from ral.devices.doctor.host import HostScanner
        from ral.devices.doctor.portaudio_scan import PortAudioScanner
        from ral.devices.doctor.ros2 import Ros2Scanner

        scanners = [HostScanner(), PortAudioScanner(), Ros2Scanner()]
    report = DoctorReport()
    for scanner in scanners:
        try:
            scanner.scan(report)
        except Exception as exc:
            # Doctor exists to explain broken setups, so it must not die on one.
            logger.debug("doctor scanner %s failed", scanner.area, exc_info=True)
            report.add(
                scanner.area,
                Level.ERROR,
                f"The {scanner.area} check failed: {type(exc).__name__}: {exc}",
                "Please report this with the output of `ral doctor --json`.",
            )
    return report
