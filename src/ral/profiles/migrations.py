"""Profile schema migrations (DESIGN section 4, "Extension points").

Profiles are shared and outlive RAL versions, so the schema is versioned (the ``ral`` key)
and every change that breaks old files ships with a migration from version ``N`` to
``N + 1``. Loading applies the chain up to :data:`SCHEMA_VERSION`; a profile newer than
this RAL is refused with a hint to upgrade RAL instead of being half understood.

Migrations work on plain dicts before validation and must be pure: same input, same
output, no I/O. Version 1 is the first and current version, so the chain is empty.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from ral.core.errors import ConfigError
from ral.profiles.schema import SCHEMA_VERSION

__all__ = ["MIGRATIONS", "migrate", "migration"]

Migration = Callable[[dict[str, Any]], dict[str, Any]]

MIGRATIONS: dict[int, Migration] = {}
"""``MIGRATIONS[n]`` turns a version ``n`` profile into version ``n + 1``."""


def migration(from_version: int) -> Callable[[Migration], Migration]:
    """Register a migration from ``from_version`` to ``from_version + 1``."""

    def register(fn: Migration) -> Migration:
        if from_version in MIGRATIONS:
            raise ValueError(f"A migration from version {from_version} already exists.")
        MIGRATIONS[from_version] = fn
        return fn

    return register


def migrate(
    data: dict[str, Any], target: int = SCHEMA_VERSION, source: str = "profile"
) -> tuple[dict[str, Any], list[int]]:
    """Bring ``data`` to ``target``; returns the new dict and the versions it passed through.

    A profile without ``ral`` is taken as the current version.

    Raises:
        ConfigError: If the profile is newer than ``target`` or a step is missing.
    """
    version = data.get("ral", target)
    if not isinstance(version, int) or version < 1:
        raise ConfigError(f"{source}: `ral` must be a schema version number, got {version!r}.")
    if version > target:
        raise ConfigError(
            f"{source} uses profile schema version {version}, but this RAL reads up to "
            f"{target}. Upgrade RAL (pip install -U ral-audio)."
        )
    out = copy.deepcopy(data)
    applied = []
    while version < target:
        step = MIGRATIONS.get(version)
        if step is None:
            raise ConfigError(f"{source}: no migration from schema version {version}.")
        out = step(out)
        version += 1
        out["ral"] = version
        applied.append(version)
    out.setdefault("ral", target)
    return out, applied
