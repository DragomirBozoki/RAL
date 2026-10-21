import json

import pytest
from typer.testing import CliRunner

from ral.cli.main import app

pytest.importorskip("mcap")
runner = CliRunner()


def test_bench_demo_json_compares_configs() -> None:
    pytest.importorskip("livekit.rtc")
    result = runner.invoke(
        app,
        ["bench", "--demo", "--echo", "off", "--echo", "software_ref", "--vad", "energy", "--json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output[result.output.index("{") :])
    names = [c["name"] for c in data["configs"]]
    assert names == ["echo=off", "echo=software_ref"]
    off, aec = data["configs"]
    assert aec["starts_per_min"] < off["starts_per_min"]


def test_bench_needs_a_session() -> None:
    result = runner.invoke(app, ["bench"])
    assert result.exit_code == 2


def test_bench_rejects_unknown_modes() -> None:
    result = runner.invoke(app, ["bench", "--demo", "--echo", "magic"])
    assert result.exit_code == 2
