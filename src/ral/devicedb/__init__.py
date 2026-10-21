"""Device database: what known audio devices can do (DESIGN section 6).

A small YAML catalogue (``devices.yaml`` in this package, community-maintained) that turns
knowledge such as "this array expects the reference on the left channel" into data that
``ral doctor`` and profile tools apply automatically. Entries match a device by USB
vendor and product id (when the platform reports it), by a substring of its name, and
optionally by its number of input channels (one board, several firmwares).

Every fact in an entry cites a public source; unknown facts are left out for
``ral probe`` to measure.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ral.core.errors import ConfigError
from ral.core.types import DeviceInfo

__all__ = ["DeviceEntry", "DeviceMatch", "find_entry", "load_devicedb"]

_ROLES = {"mic", "reference", "processed", "unused"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DeviceMatch(_Strict):
    usb: list[str] = []
    """``"vvvv:pppp"`` in hex, lower case."""
    name: list[str] = []
    """Case-insensitive substrings of the device name."""
    input_channels: int | None = None


class DeviceEntry(_Strict):
    id: str
    name: str
    match: DeviceMatch
    hardware_aec: bool = False
    rates: list[int] = []
    channels: list[str] | None = None
    """Role of each input channel, when documented."""
    reference_channel: Literal["left", "right"] | None = None
    """Which playback channel the device uses as its AEC reference."""
    aec_tail_ms: int | None = None
    microphones: int | None = None
    echo_mode: Literal["hardware", "loopback", "software_ref", "gate"] | None = None
    notes: str = ""
    sources: list[str] = Field(min_length=1)
    validated_by: Literal["docs", "community", "maintainer", "certified"] = "docs"

    @field_validator("channels")
    @classmethod
    def _roles(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and any(r not in _ROLES for r in v):
            raise ValueError(f"channel roles must be among {sorted(_ROLES)}")
        return v

    def matches(self, device: DeviceInfo) -> bool:
        """Whether ``device`` (as a backend reports it) is this entry."""
        usb = device.extra.get("usb", "").lower()
        by_usb = bool(usb) and usb in [u.lower() for u in self.match.usb]
        by_name = any(p.lower() in device.name.lower() for p in self.match.name)
        if not (by_usb or by_name):
            return False
        want = self.match.input_channels
        return want is None or device.max_input_channels == want


class _Db(_Strict):
    version: Literal[1]
    devices: list[DeviceEntry]


@lru_cache(maxsize=1)
def load_devicedb() -> tuple[DeviceEntry, ...]:
    """The built-in device database.

    Raises:
        ConfigError: If the catalogue is malformed (a packaging bug).
    """
    text = (resources.files("ral.devicedb") / "devices.yaml").read_text(encoding="utf-8")
    try:
        db = _Db.model_validate(yaml.safe_load(text))
    except ValidationError as exc:
        raise ConfigError(f"The built-in device database is invalid: {exc}") from None
    ids = [d.id for d in db.devices]
    if len(ids) != len(set(ids)):
        raise ConfigError("The built-in device database has duplicate ids.")
    return tuple(db.devices)


def find_entry(device: DeviceInfo) -> DeviceEntry | None:
    """The first database entry that matches ``device``, if any."""
    for entry in load_devicedb():
        if entry.matches(device):
            return entry
    return None
