"""Run on a laptop with a speaker and a microphone: uv run pytest -m hardware tests/probe"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ral.cli.main import app
from ral.profiles import parse_profile


@pytest.mark.hardware
def test_probe_default_devices_writes_a_working_profile(tmp_path: Path) -> None:
    out, report = tmp_path / "laptop.yaml", tmp_path / "report.html"
    result = CliRunner().invoke(
        app, ["probe", "--yes", "--robot", "laptop", "--out", str(out), "--report", str(report)]
    )
    assert result.exit_code == 0, result.output
    profile = parse_profile(out.read_text(encoding="utf-8"))
    assert "mic" in (profile.capture.channels or [])
    print(out.read_text(encoding="utf-8"))
