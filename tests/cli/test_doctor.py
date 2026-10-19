import json

import pytest
from typer.testing import CliRunner

from ral.cli import doctor as doctor_cli
from ral.cli.main import app
from ral.core import DeviceInfo
from ral.devices.doctor import DoctorReport, Level, PublisherQos, TopicInfo

runner = CliRunner()


def fake_report(error: bool = False) -> DoctorReport:
    report = DoctorReport()
    report.add("host", Level.OK, "RAL clock: QueryPerformanceCounter, resolution 100 ns")
    report.add("audio", Level.ERROR if error else Level.WARN, "No capture device.", "Use ROS 2.")
    report.devices.append(
        DeviceInfo(
            "portaudio:wasapi:3",
            "Speakers",
            "portaudio",
            "Windows WASAPI",
            0,
            2,
            48000.0,
            {"index": "3", "default_output": "true"},
        )
    )
    report.topics.append(
        TopicInfo(
            "/mic",
            ("audio_common_msgs/msg/AudioData",),
            (PublisherQos("/n", "reliable", "volatile", "keep_last", 10),),
        )
    )
    report.skipped["x"] = "not here"
    return report


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> list[list[object]]:
    calls: list[list[object]] = []

    def run(scanners: list[object]) -> DoctorReport:
        calls.append(scanners)
        return fake_report()

    monkeypatch.setattr(doctor_cli, "run_doctor", run)
    monkeypatch.setattr(doctor_cli, "preferred_host_apis", lambda: ("Windows WASAPI",))
    return calls


def test_table_output(patched: list[list[object]]) -> None:
    result = runner.invoke(app, ["doctor"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    for text in (
        "ral doctor",
        "No capture device.",
        "Use ROS 2.",
        "Audio devices",
        "portaudio:wasapi:3",
        "ROS 2 audio topics",
        "x: skipped, not here.",
        "Summary",
        "audio: amber",
    ):
        assert text in result.output
    assert [s.area for s in patched[0]] == ["host", "audio", "ros2"]  # type: ignore[attr-defined]


def test_json_output_and_no_ros2(patched: list[list[object]]) -> None:
    result = runner.invoke(app, ["doctor", "--json", "--no-ros2"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["summary"] == {"host": "ok", "audio": "warn"}
    assert data["devices"][0]["extra"]["default_output"] == "true"
    assert [s.area for s in patched[0]] == ["host", "audio"]  # type: ignore[attr-defined]


def test_errors_set_exit_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_cli, "run_doctor", lambda scanners: fake_report(error=True))
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 1
