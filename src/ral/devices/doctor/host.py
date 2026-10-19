"""Host checks: operating system, Python, the RAL clock, and Linux sound access."""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Callable, Mapping

from ral.core import clock
from ral.devices.doctor.report import DoctorReport, Level

__all__ = ["HostScanner"]

_CLOCK_LABELS = {"monotonic": "time.monotonic", "perf_counter": "QueryPerformanceCounter"}


def _format_ns(ns: int) -> str:
    if ns >= 1_000_000:
        return f"{ns / 1e6:g} ms"
    if ns >= 1_000:
        return f"{ns / 1e3:g} us"
    return f"{ns} ns"


class HostScanner:
    """Read-only checks of the machine RAL runs on.

    The arguments exist so tests can describe another machine; the defaults read this one.
    """

    area = "host"

    def __init__(
        self,
        *,
        system: str | None = None,
        platform_name: str | None = None,
        env: Mapping[str, str] | None = None,
        exists: Callable[[str], bool] = os.path.exists,
        accessible: Callable[[str], bool] = lambda p: os.access(p, os.R_OK | os.W_OK),
        clock_name: str = clock.CLOCK_NAME,
        clock_resolution_ns: int = clock.CLOCK_RESOLUTION_NS,
    ) -> None:
        self._system = system or platform.system()
        self._platform_name = platform_name or platform.platform()
        self._env = os.environ if env is None else env
        self._exists = exists
        self._accessible = accessible
        self._clock_name = clock_name
        self._clock_resolution_ns = clock_resolution_ns

    def scan(self, report: DoctorReport) -> None:
        """Add host findings to ``report``."""
        report.add(self.area, Level.INFO, f"{self._platform_name}, Python {sys.version.split()[0]}")
        self._check_clock(report)
        if self._system == "Linux":
            self._check_alsa(report)
            self._check_sound_server(report)

    def _check_clock(self, report: DoctorReport) -> None:
        label = _CLOCK_LABELS.get(self._clock_name, self._clock_name)
        message = f"RAL clock: {label}, resolution {_format_ns(self._clock_resolution_ns)}"
        if self._clock_resolution_ns > 1_000_000:
            report.add(
                self.area,
                Level.WARN,
                message,
                "Timestamps are coarser than 1 ms; echo alignment numbers will be unreliable.",
            )
        else:
            report.add(self.area, Level.OK, message)

    def _check_alsa(self, report: DoctorReport) -> None:
        if not self._exists("/dev/snd"):
            report.add(
                self.area,
                Level.WARN,
                "No ALSA sound devices on this host (/dev/snd is missing).",
                "The microphones may live on another board: look for audio on ROS 2 topics. "
                "In a container, pass the devices in with --device /dev/snd.",
            )
        elif not self._accessible("/dev/snd"):
            report.add(
                self.area,
                Level.ERROR,
                "/dev/snd exists but this user cannot open it.",
                "Add the user to the audio group (sudo usermod -aG audio $USER), then log in "
                "again.",
            )
        else:
            report.add(self.area, Level.OK, "ALSA sound devices are accessible (/dev/snd).")

    def _check_sound_server(self, report: DoctorReport) -> None:
        runtime = self._env.get("XDG_RUNTIME_DIR", "")
        pipewire = bool(runtime) and self._exists(f"{runtime}/pipewire-0")
        pulse = bool(runtime) and self._exists(f"{runtime}/pulse/native")
        if pipewire:
            extra = " with its PulseAudio service" if pulse else ""
            report.add(self.area, Level.OK, f"PipeWire is running{extra}.")
        elif pulse:
            report.add(self.area, Level.OK, "PulseAudio is running.")
        else:
            report.add(
                self.area,
                Level.INFO,
                "No PipeWire or PulseAudio server for this user; PortAudio uses ALSA directly.",
                "Without a sound server a device can be opened by one program at a time.",
            )
        # TODO(M2): report whether the sound server holds a device that RAL wants.
