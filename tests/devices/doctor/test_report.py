import json

from ral.core import DeviceInfo
from ral.devices.doctor import DoctorReport, Finding, Level, PublisherQos, TopicInfo, run_doctor


class AddsOne:
    area = "a"

    def __init__(self, level: Level) -> None:
        self.level = level

    def scan(self, report: DoctorReport) -> None:
        report.add(self.area, self.level, "msg", "hint")


class Crashes:
    area = "b"

    def scan(self, report: DoctorReport) -> None:
        raise RuntimeError("boom")


def test_area_level_is_the_worst_and_info_never_raises_it() -> None:
    report = run_doctor([AddsOne(Level.INFO), AddsOne(Level.OK)])
    assert report.area_levels() == {"a": Level.OK}
    report = run_doctor([AddsOne(Level.OK), AddsOne(Level.WARN), AddsOne(Level.INFO)])
    assert report.area_levels() == {"a": Level.WARN}
    assert not report.has_errors


def test_crashing_scanner_becomes_an_error_finding() -> None:
    report = run_doctor([Crashes(), AddsOne(Level.OK)])
    assert report.findings[0].area == "b"
    assert report.findings[0].level is Level.ERROR
    assert "RuntimeError: boom" in report.findings[0].message
    assert report.findings[1] == Finding("a", Level.OK, "msg", "hint")
    assert report.has_errors


def test_to_dict_is_json_ready() -> None:
    report = DoctorReport()
    report.add("audio", Level.WARN, "no mic", "connect one")
    report.devices.append(DeviceInfo("file:in:x", "x", "file", None, 1, 0, 16000.0))
    report.topics.append(
        TopicInfo(
            "/mic",
            ("audio_common_msgs/msg/AudioData",),
            (PublisherQos("/n", "reliable", "volatile", "keep_last", 10),),
        )
    )
    report.skipped["ros2"] = "no rclpy"
    data = json.loads(json.dumps(report.to_dict()))
    assert data["summary"] == {"audio": "warn"}
    assert data["findings"] == [
        {"area": "audio", "level": "warn", "message": "no mic", "hint": "connect one"}
    ]
    assert data["devices"][0]["id"] == "file:in:x"
    assert data["ros2_topics"][0]["publishers"][0]["reliability"] == "reliable"
    assert data["skipped"] == {"ros2": "no rclpy"}
