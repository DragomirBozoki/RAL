"""Error hierarchy for RAL.

Every error RAL raises subclasses :class:`RalError`, so callers can catch RAL failures
in one place. Messages say what failed and how to fix it.
"""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "DeviceError",
    "DeviceNotFoundError",
    "FormatError",
    "MissingDependencyError",
    "RalError",
]


class RalError(Exception):
    """Base class for all errors raised by RAL."""


class FormatError(RalError, ValueError):
    """An audio format, channel map or frame is invalid or does not match what was expected."""


class DeviceError(RalError):
    """An audio device could not be opened, used or closed."""


class DeviceNotFoundError(DeviceError, LookupError):
    """No device matches the requested id or name."""


class MissingDependencyError(RalError, ImportError):
    """An optional dependency is needed but not installed.

    Args:
        package: The importable module that is missing, for example ``"sounddevice"``.
        extra: The ``ral-audio`` extra that installs it, for example ``"audio"``.
        purpose: What needs it, for example ``"the PortAudio backend"``.
    """

    def __init__(self, package: str, extra: str, purpose: str) -> None:
        self.package = package
        self.extra = extra
        self.install_hint = f'pip install "ral-audio[{extra}]"'
        super().__init__(
            f"{purpose} needs the optional package '{package}', which is not installed. "
            f"Install it with: {self.install_hint}"
        )


class ConfigError(RalError, ValueError):
    """A profile or setting is missing or invalid."""
