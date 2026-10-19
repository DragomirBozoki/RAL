"""PortAudio checks: library, host APIs, devices and defaults."""

from __future__ import annotations

from collections.abc import Callable

from ral.core.errors import DeviceError, DeviceNotFoundError, FormatError, MissingDependencyError
from ral.core.types import DeviceInfo
from ral.devices.doctor.report import DoctorReport, Level
from ral.devices.portaudio import (
    Direction,
    PortAudioBackend,
    group_by_host_api,
    preferred_host_apis,
)

__all__ = ["PortAudioScanner"]

# MME cuts device names to 31 characters, so compare names on that prefix.
_MME_NAME_LENGTH = 31


class PortAudioScanner:
    """Lists PortAudio devices and checks the default capture and playback devices.

    Args:
        backend_factory: Builds the backend; tests pass one with a fake ``sounddevice``.
    """

    area = "audio"

    def __init__(self, backend_factory: Callable[[], PortAudioBackend] = PortAudioBackend) -> None:
        self._backend_factory = backend_factory

    def scan(self, report: DoctorReport) -> None:
        """Add PortAudio findings and the device list to ``report``."""
        try:
            backend = self._backend_factory()
        except MissingDependencyError as exc:
            report.add(self.area, Level.WARN, str(exc), exc.install_hint)
            return
        except DeviceError as exc:
            report.add(
                self.area,
                Level.ERROR,
                str(exc),
                "On Debian and Ubuntu: sudo apt install libportaudio2",
            )
            return
        report.add(self.area, Level.OK, f"PortAudio library loaded: {backend.portaudio_version}")

        devices = backend.discover()
        report.devices.extend(devices)
        if not devices:
            report.add(
                self.area,
                Level.ERROR,
                "PortAudio found no audio devices.",
                "Connect a USB microphone or speaker, or look for audio on ROS 2 topics.",
            )
            return
        groups = group_by_host_api(devices)
        counts = ", ".join(f"{api} {len(devs)}" for api, devs in groups.items())
        report.add(self.area, Level.INFO, f"{len(devices)} devices by host API: {counts}")
        preferred = preferred_host_apis()
        if preferred and not any(api in groups for api in preferred):
            report.add(
                self.area,
                Level.WARN,
                f"The preferred host API ({preferred[0]}) is not available.",
                "RAL falls back to another host API; latency may be higher.",
            )
        self._check_default(report, backend, "input")
        self._check_default(report, backend, "output")
        self._note_duplicates(report, groups)

    def _check_default(
        self, report: DoctorReport, backend: PortAudioBackend, direction: Direction
    ) -> None:
        kind = "capture" if direction == "input" else "playback"
        try:
            dev = backend.default_device(direction)
        except DeviceNotFoundError:
            hint = (
                "Look for audio on ROS 2 topics, or connect a USB microphone."
                if direction == "input"
                else "Connect a speaker, or play through a ROS 2 topic."
            )
            report.add(self.area, Level.WARN, f"No {kind} device on this host.", hint)
            return
        try:
            fmt = backend.native_format(dev, direction)
        except FormatError as exc:
            report.add(self.area, Level.WARN, str(exc))
            return
        report.add(
            self.area,
            Level.OK,
            f"Default {kind}: {dev.name} [{dev.host_api}], {fmt.channels} ch, "
            f"{fmt.sample_rate} Hz ({dev.id})",
        )

    def _note_duplicates(self, report: DoctorReport, groups: dict[str, list[DeviceInfo]]) -> None:
        apis_by_name: dict[str, set[str]] = {}
        for api, devs in groups.items():
            for dev in devs:
                apis_by_name.setdefault(dev.name[:_MME_NAME_LENGTH], set()).add(api)
        shared = sum(1 for apis in apis_by_name.values() if len(apis) > 1)
        if shared:
            preferred = preferred_host_apis()
            target = preferred[0] if preferred else "the first host API"
            report.add(
                self.area,
                Level.INFO,
                f"{shared} device names appear under several host APIs; RAL uses {target}.",
            )
