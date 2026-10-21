from pathlib import Path

import pytest
from typer.testing import CliRunner

from ral.cli.main import app
from ral.profiles import load_profile
from ral.profiles.registry import find_profile, list_profiles

runner = CliRunner()


@pytest.fixture
def user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "profiles"
    folder.mkdir()
    monkeypatch.setenv("RAL_PROFILES_PATH", str(folder))
    return folder


def test_builtin_profiles_are_valid() -> None:
    names = {e.name for e in list_profiles() if e.source == "built-in"}
    assert {"laptop", "respeaker-usb-v2"} <= names
    for name in names:
        load_profile(name)
    array = load_profile("respeaker-usb-v2")
    assert array.echo.mode == "hardware"
    assert array.capture.channels == ["processed", "mic", "mic", "mic", "mic", "reference"]


def test_user_profiles_are_found_by_name_and_override_builtins(user_dir: Path) -> None:
    (user_dir / "lab-bot.yaml").write_text("robot: {model: lab-bot}\necho: {mode: gate}\n")
    (user_dir / "laptop.yaml").write_text("robot: {model: tuned-laptop}\n")
    assert load_profile("lab-bot").echo.mode == "gate"
    assert load_profile("laptop").robot.model == "tuned-laptop"  # rolled out
    entries = {e.name: e for e in list_profiles()}
    assert entries["laptop"].source.endswith("laptop.yaml")
    (user_dir / "laptop.yaml").unlink()
    assert load_profile("laptop").robot.model == "laptop"  # rolled back


def test_a_broken_user_profile_does_not_hide_the_others(user_dir: Path) -> None:
    (user_dir / "broken.yaml").write_text("robot: [unclosed\n")
    entries = {e.name: e for e in list_profiles()}
    assert entries["broken"].error
    assert "laptop" in entries


def test_paths_win_over_names(tmp_path: Path) -> None:
    f = tmp_path / "laptop"
    f.write_text("robot: {model: from-path}\n")
    found = find_profile(str(f))
    assert found is not None
    assert found[1] == str(f)


def test_cli_list_show_validate(user_dir: Path, tmp_path: Path) -> None:
    (user_dir / "lab-bot.yaml").write_text("robot: {model: lab-bot}\n")
    assert "lab-bot" in runner.invoke(app, ["profile", "list"]).output
    shown = runner.invoke(app, ["profile", "show", "lab-bot"])
    assert shown.exit_code == 0
    assert "lead_ms: 20" in shown.output  # defaults filled in
    good, bad = tmp_path / "good.yaml", tmp_path / "bad.yaml"
    good.write_text("echo: {mode: off}\n")
    bad.write_text("echo: {mode: magic}\n")
    assert runner.invoke(app, ["profile", "validate", str(good)]).exit_code == 0
    result = runner.invoke(app, ["profile", "validate", str(good), str(bad)])
    assert result.exit_code == 1
    assert "echo.mode" in result.output


def test_cli_migrate_reports_current_files(tmp_path: Path) -> None:
    f = tmp_path / "p.yaml"
    f.write_text("ral: 1\n")
    result = runner.invoke(app, ["profile", "migrate", str(f)])
    assert result.exit_code == 0
    assert "already at the current schema" in result.output
