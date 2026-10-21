"""Where profiles come from: files, user folders and the built-in set (DESIGN section 6).

A name is looked up in order: an existing file path; each folder in ``$RAL_PROFILES_PATH``
(separated by the platform's path separator); the user's profile folder
(``%APPDATA%\\ral\\profiles`` on Windows, ``~/Library/Application Support/ral/profiles``
on macOS, ``$XDG_CONFIG_HOME/ral/profiles`` or ``~/.config/ral/profiles`` elsewhere);
then the profiles that ship with RAL. A user profile with a built-in name overrides it, so
a team can roll out a tuned version and roll it back by deleting the file.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

__all__ = ["ProfileEntry", "find_profile", "list_profiles", "profile_dirs"]


@dataclass(frozen=True)
class ProfileEntry:
    """One profile the registry can load."""

    name: str
    source: str
    """``"built-in"`` or the file path."""
    model: str
    validated_by: str
    echo_mode: str
    error: str = ""
    """Why the entry could not be read, if it could not."""


def profile_dirs() -> list[Path]:
    """User profile folders, in search order."""
    dirs = [Path(p) for p in os.environ.get("RAL_PROFILES_PATH", "").split(os.pathsep) if p]
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    dirs.append(base / "ral" / "profiles")
    return dirs


def _builtin() -> dict[str, str]:
    folder = resources.files("ral.profiles") / "builtin"
    return {
        p.name.removesuffix(".yaml"): p.read_text(encoding="utf-8")
        for p in folder.iterdir()
        if p.name.endswith(".yaml")
    }


def find_profile(name: str) -> tuple[str, str] | None:
    """``(yaml_text, source)`` for a path or name, or None if nothing matches."""
    path = Path(name)
    if path.is_file():
        return path.read_text(encoding="utf-8"), str(path)
    for folder in profile_dirs():
        candidate = folder / f"{name}.yaml"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8"), str(candidate)
    text = _builtin().get(name)
    return (text, f"built-in profile {name!r}") if text is not None else None


def _entry(name: str, text: str, source: str) -> ProfileEntry:
    try:
        data = yaml.safe_load(text) or {}
        robot = data.get("robot", {}) or {}
        echo = data.get("echo", {}) or {}
        mode = echo.get("mode", "software_ref")
        return ProfileEntry(
            name,
            source,
            str(robot.get("model", "unknown")),
            str(robot.get("validated_by", "community")),
            "off" if mode is False else str(mode),
        )
    except Exception as exc:  # a broken user file must not hide the others
        return ProfileEntry(name, source, "?", "?", "?", error=str(exc))


def list_profiles() -> list[ProfileEntry]:
    """Every profile by name; user folders override built-ins of the same name."""
    found: dict[str, ProfileEntry] = {
        name: _entry(name, text, "built-in") for name, text in _builtin().items()
    }
    for folder in reversed(profile_dirs()):
        if folder.is_dir():
            for path in sorted(folder.glob("*.yaml")):
                found[path.stem] = _entry(path.stem, path.read_text(encoding="utf-8"), str(path))
    return sorted(found.values(), key=lambda e: e.name)
