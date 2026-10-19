"""Profiles: per-robot YAML files the engine runs from (DESIGN section 6).

``${VAR}`` and ``${VAR:-default}`` in string values are replaced from the environment
when the profile is loaded, so secrets stay in ``.env`` and out of profile files.
Built-in profiles ship in this package: ``load_profile("laptop")`` finds
``ral/profiles/builtin/laptop.yaml``.
"""

from __future__ import annotations

import os
import re
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ral.core.errors import ConfigError
from ral.profiles.schema import Profile

__all__ = ["Profile", "builtin_profiles", "load_profile", "parse_profile"]

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any, where: str) -> Any:
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(
                f"{where} uses ${{{name}}}, which is not set. Set it in the environment or "
                f"in .env (see .env.example), or give a default as ${{{name}:-value}}."
            )

        return _VAR.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand(v, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, f"{where}[{i}]") for i, v in enumerate(value)]
    return value


def parse_profile(text: str, source: str = "<profile>") -> Profile:
    """Parse and validate profile YAML.

    Raises:
        ConfigError: With every problem and where it is, if the profile is invalid.
    """
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{source} must be a mapping at the top level.")
    data = _expand(data, "profile")
    try:
        return Profile.model_validate(data)
    except ValidationError as exc:
        lines = [
            f"  {'.'.join(str(p) for p in err['loc']) or '(top level)'}: {err['msg']}"
            for err in exc.errors()
        ]
        raise ConfigError(f"{source} is not a valid profile:\n" + "\n".join(lines)) from None


def builtin_profiles() -> list[str]:
    """Names of the profiles that ship with RAL."""
    folder = resources.files("ral.profiles") / "builtin"
    return sorted(
        p.name.removesuffix(".yaml") for p in folder.iterdir() if p.name.endswith(".yaml")
    )


def load_profile(path_or_name: str | Path) -> Profile:
    """Load a profile file, or a built-in profile by name.

    Raises:
        ConfigError: If the file does not exist or is invalid.
    """
    path = Path(path_or_name)
    if not path.is_file() and isinstance(path_or_name, str) and path_or_name in builtin_profiles():
        text = (resources.files("ral.profiles") / "builtin" / f"{path_or_name}.yaml").read_text(
            encoding="utf-8"
        )
        return parse_profile(text, f"built-in profile {path_or_name!r}")
    if not path.is_file():
        known = ", ".join(builtin_profiles())
        raise ConfigError(f"No profile file {str(path)!r}; built-in profiles: {known}.")
    return parse_profile(path.read_text(encoding="utf-8"), str(path))
