import json
from pathlib import Path

from typer.testing import CliRunner

from ral.cli.main import app

DATA = Path(__file__).resolve().parents[1] / "data"
runner = CliRunner()


def test_run_with_a_file_profile_prints_json_events(tmp_path: Path) -> None:
    profile = tmp_path / "p.yaml"
    mic = (DATA / "speech_16k.wav").as_posix()
    out = (tmp_path / "out.wav").as_posix()
    profile.write_text(
        "ral: 1\n"
        f"capture: {{backend: file, path: '{mic}', realtime: false}}\n"
        f"playback: {{backend: file, path: '{out}', realtime: false}}\n"
        "echo: {mode: off}\n"
        "pipeline: [aec, vad]\n"
        "vad: {engine: energy}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["run", "--profile", str(profile), "--json"])
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]
    assert any(e["type"] == "SpeechStarted" for e in events)


def test_run_with_a_bad_profile_exits_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("echo: {mode: magic}\n", encoding="utf-8")
    result = runner.invoke(app, ["run", "--profile", str(bad)])
    assert result.exit_code == 2
    assert "echo.mode" in result.output
