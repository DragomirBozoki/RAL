from typing import Any

import pytest

from ral.core import ConfigError
from ral.profiles import migrations as mig
from ral.profiles.schema import SCHEMA_VERSION


def test_current_profiles_pass_through_unchanged() -> None:
    data = {"ral": SCHEMA_VERSION, "echo": {"mode": "gate"}}
    out, applied = mig.migrate(data)
    assert out == data
    assert applied == []
    assert out is not data  # never mutates the input


def test_missing_version_means_current() -> None:
    out, _ = mig.migrate({"echo": {"mode": "off"}})
    assert out["ral"] == SCHEMA_VERSION


def test_newer_profiles_are_refused_with_an_upgrade_hint() -> None:
    with pytest.raises(ConfigError, match="Upgrade RAL"):
        mig.migrate({"ral": SCHEMA_VERSION + 1})


@pytest.mark.parametrize("bad", [0, "one", None])
def test_bad_versions(bad: object) -> None:
    with pytest.raises(ConfigError):
        mig.migrate({"ral": bad})


def test_a_chain_of_migrations_runs_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    # A pretend v1 -> v2 -> v3 history: v2 renamed echo.lead to echo.lead_ms, v3 added a key.
    monkeypatch.setattr(mig, "MIGRATIONS", {})

    @mig.migration(1)
    def lead_renamed(d: dict[str, Any]) -> dict[str, Any]:
        d["echo"]["lead_ms"] = d["echo"].pop("lead")
        return d

    @mig.migration(2)
    def add_engine(d: dict[str, Any]) -> dict[str, Any]:
        d.setdefault("engine", {"rate": 16000})
        return d

    out, applied = mig.migrate({"ral": 1, "echo": {"lead": 30}}, target=3)
    assert applied == [2, 3]
    assert out == {"ral": 3, "echo": {"lead_ms": 30}, "engine": {"rate": 16000}}
    with pytest.raises(ValueError, match="already exists"):
        mig.migration(1)(lead_renamed)


def test_a_gap_in_the_chain_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mig, "MIGRATIONS", {})
    with pytest.raises(ConfigError, match="no migration from schema version 1"):
        mig.migrate({"ral": 1}, target=2)
