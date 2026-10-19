from __future__ import annotations

import sys
from types import TracebackType

import pytest

from ral.devices.doctor import DoctorReport, Level, PublisherQos
from ral.devices.doctor.ros2 import Ros2Scanner, is_audio_like

AUDIO = "audio_common_msgs/msg/AudioDataStamped"
RELIABLE = PublisherQos("/hal/audio_node", "reliable", "volatile", "keep_last", 10)
BEST_EFFORT = PublisherQos("/vendor/hal", "best_effort", "volatile", "keep_last", 5)


class FakeGraph:
    def __init__(self, topics: dict[str, tuple[list[str], list[PublisherQos]]]) -> None:
        self._topics = topics
        self.closed = False

    def __enter__(self) -> FakeGraph:
        return self

    def __exit__(
        self, t: type[BaseException] | None, e: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.closed = True

    def topics(self) -> list[tuple[str, list[str]]]:
        return [(name, types) for name, (types, _) in self._topics.items()]

    def publishers(self, topic: str) -> list[PublisherQos]:
        return self._topics[topic][1]


def scan(graph: FakeGraph, env: dict[str, str] | None = None) -> DoctorReport:
    report = DoctorReport()
    Ros2Scanner(lambda: graph, env={"ROS_DISTRO": "humble", **(env or {})}).scan(report)
    return report


def messages(report: DoctorReport) -> list[tuple[Level, str]]:
    return [(f.level, f.message) for f in report.findings]


@pytest.mark.parametrize(
    ("name", "types", "expected"),
    [
        ("/hal/audio/capture", ["vendor_msgs/msg/Frame"], True),
        ("/robot/mic_array", ["std_msgs/msg/ByteMultiArray"], True),
        ("/speaker_out", ["std_msgs/msg/ByteMultiArray"], True),
        ("/x", [AUDIO], True),
        ("/cmd_vel", ["geometry_msgs/msg/Twist"], False),
        ("/dynamic_reconfigure", ["std_msgs/msg/String"], False),
    ],
)
def test_audio_like_topics(name: str, types: list[str], expected: bool) -> None:
    assert is_audio_like(name, types) is expected


def test_skipped_when_rclpy_does_not_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rclpy", None)
    report = DoctorReport()
    Ros2Scanner().scan(report)
    assert report.findings == []
    assert "source your ROS 2 setup" in report.skipped["ros2"]


def test_reports_environment_defaults() -> None:
    report = scan(FakeGraph({"/audio": ([AUDIO], [RELIABLE])}))
    assert messages(report)[:2] == [
        (Level.INFO, "ROS 2 humble, RMW not set (the distro default, usually Fast DDS)"),
        (Level.INFO, "ROS_DOMAIN_ID not set (domain 0)."),
    ]


def test_lists_audio_topics_with_qos_and_closes_the_graph() -> None:
    graph = FakeGraph(
        {
            "/hal/audio/capture": ([AUDIO], [RELIABLE]),
            "/cmd_vel": (["geometry_msgs/msg/Twist"], []),
        }
    )
    report = scan(graph, {"RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp", "ROS_DOMAIN_ID": "7"})
    assert (Level.INFO, "ROS_DOMAIN_ID=7") in messages(report)
    assert messages(report)[-1] == (
        Level.OK,
        f"/hal/audio/capture [{AUDIO}]: reliable/volatile/depth 10 from /hal/audio_node",
    )
    assert [t.name for t in report.topics] == ["/hal/audio/capture"]
    assert report.topics[0].publishers == (RELIABLE,)
    assert graph.closed


def test_best_effort_publisher_and_vendor_type_are_flagged() -> None:
    report = scan(FakeGraph({"/hal/audio": (["vendor_msgs/msg/AudioFrame"], [BEST_EFFORT])}))
    found = messages(report)
    assert (
        Level.WARN,
        "/hal/audio: a publisher is best-effort, so a reliable subscription gets nothing.",
    ) in found
    vendor = next(f for f in report.findings if "not audio_common_msgs" in f.message)
    assert vendor.level is Level.INFO
    assert vendor.hint is not None
    assert "decoder plugin" in vendor.hint


def test_no_audio_topics_and_topic_without_publishers() -> None:
    assert messages(scan(FakeGraph({"/cmd_vel": (["geometry_msgs/msg/Twist"], [])})))[-1] == (
        Level.WARN,
        "No audio-like topics are visible.",
    )
    assert messages(scan(FakeGraph({"/audio": ([AUDIO], [])})))[-1] == (
        Level.WARN,
        f"/audio [{AUDIO}] has no publishers.",
    )


@pytest.mark.parametrize(
    ("env", "fragment"),
    [
        ({"ROS_DOMAIN_ID": "300"}, "is not a domain ID"),
        ({"ROS_DOMAIN_ID": "abc"}, "is not a domain ID"),
        ({"ROS_LOCALHOST_ONLY": "1"}, "other boards are invisible"),
        ({"ROS_AUTOMATIC_DISCOVERY_RANGE": "localhost"}, "other boards are invisible"),
        (
            {"ROS_DISCOVERY_SERVER": "10.0.0.2:11811", "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp"},
            "only applies to Fast DDS",
        ),
    ],
)
def test_risky_discovery_settings_warn(env: dict[str, str], fragment: str) -> None:
    report = scan(FakeGraph({"/audio": ([AUDIO], [RELIABLE])}), env)
    warn = [f for f in report.findings if f.level is Level.WARN]
    assert any(fragment in (f.message + (f.hint or "")) for f in warn)


def test_discovery_server_without_super_client_is_noted() -> None:
    report = scan(FakeGraph({"/audio": ([AUDIO], [RELIABLE])}), {"ROS_DISCOVERY_SERVER": "a:1"})
    assert any("ROS_SUPER_CLIENT" in f.message for f in report.findings)
    report = scan(
        FakeGraph({"/audio": ([AUDIO], [RELIABLE])}),
        {"ROS_DISCOVERY_SERVER": "a:1", "ROS_SUPER_CLIENT": "TRUE"},
    )
    assert not any("ROS_SUPER_CLIENT" in f.message for f in report.findings)
