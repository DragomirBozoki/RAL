"""ROS 2 checks: middleware settings, audio-like topics, their types and publisher QoS.

The section runs only when ``rclpy`` imports, which means a ROS 2 installation is
sourced; otherwise the report records it as skipped. The graph is reached through the
small :class:`Ros2Graph` protocol so tests can use a fake; :class:`RclpyGraph` is the real
one, built on the rclpy ``Node`` graph API (``get_topic_names_and_types`` and
``get_publishers_info_by_topic``, both in rclpy since Foxy):
https://docs.ros.org/en/humble/p/rclpy/api/node.html

Environment variables follow the ROS 2 docs on domain IDs, discovery and Fast DDS
discovery servers:
https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Domain-ID.html
https://docs.ros.org/en/jazzy/Tutorials/Advanced/Improved-Dynamic-Discovery.html
https://docs.ros.org/en/humble/Tutorials/Advanced/Discovery-Server/Discovery-Server.html
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Any, Protocol

from ral.devices.doctor.report import DoctorReport, Level, PublisherQos, TopicInfo

__all__ = ["RclpyGraph", "Ros2Graph", "Ros2Scanner", "is_audio_like"]

_STANDARD_AUDIO_PACKAGE = "audio_common_msgs/"
_AUDIO_WORDS = ("audio", "mic", "speaker", "sound", "voice")
# Domain IDs above 232 collide with ports for most DDS vendors (see the domain ID doc).
_MAX_SAFE_DOMAIN_ID = 232


class Ros2Graph(Protocol):
    """What the scanner needs from a ROS 2 graph."""

    def topics(self) -> list[tuple[str, list[str]]]:
        """Topic names with their message types."""
        ...

    def publishers(self, topic: str) -> list[PublisherQos]:
        """Publishers of ``topic`` and the QoS each offers."""
        ...


def is_audio_like(name: str, types: list[str]) -> bool:
    """True if the topic's type or name suggests audio."""
    if any("audio" in t.lower() for t in types):
        return True
    words = name.lower().replace("-", "_").replace("/", "_").split("_")
    return any(w.startswith(_AUDIO_WORDS) for w in words if w)


def _policy_name(policy: Any) -> str:
    return str(getattr(policy, "name", policy)).lower()


class RclpyGraph:
    """A short-lived rclpy node that waits for discovery, then answers graph queries.

    It uses its own rclpy context, so it does not touch an application's context.

    Args:
        wait_s: How long to let discovery run before the first query.
    """

    def __init__(self, wait_s: float = 2.0) -> None:
        import rclpy
        from rclpy.context import Context

        self._rclpy = rclpy
        self._context = Context()
        self._wait_s = wait_s
        self._node: Any = None

    def __enter__(self) -> RclpyGraph:
        self._rclpy.init(context=self._context)
        self._node = self._rclpy.create_node("ral_doctor", context=self._context)
        time.sleep(self._wait_s)  # discovery runs in the middleware's own threads
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._node is not None:
            self._node.destroy_node()
        self._rclpy.shutdown(context=self._context)

    def topics(self) -> list[tuple[str, list[str]]]:
        """Topic names with their message types."""
        return [(str(n), [str(t) for t in ts]) for n, ts in self._node.get_topic_names_and_types()]

    def publishers(self, topic: str) -> list[PublisherQos]:
        """Publishers of ``topic`` and the QoS each offers."""
        result = []
        for info in self._node.get_publishers_info_by_topic(topic):
            qos = info.qos_profile
            namespace = str(info.node_namespace).rstrip("/")
            result.append(
                PublisherQos(
                    node=f"{namespace}/{info.node_name}",
                    reliability=_policy_name(qos.reliability),
                    durability=_policy_name(qos.durability),
                    history=_policy_name(qos.history),
                    depth=int(qos.depth),
                )
            )
        return result


def _rclpy_available() -> bool:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        return False
    return True


class Ros2Scanner:
    """Middleware settings and audio-like topics on the visible ROS 2 graph.

    Args:
        graph_factory: Builds a context manager yielding a :class:`Ros2Graph`; ``None``
            uses rclpy if it imports, and skips the section if not.
        env: Environment to read; defaults to this process's.
        wait_s: Discovery time for the default rclpy graph.
    """

    area = "ros2"

    def __init__(
        self,
        graph_factory: Callable[[], Any] | None = None,
        env: Mapping[str, str] | None = None,
        wait_s: float = 2.0,
    ) -> None:
        self._graph_factory = graph_factory
        self._env = os.environ if env is None else env
        self._wait_s = wait_s

    def scan(self, report: DoctorReport) -> None:
        """Add ROS 2 findings and audio topics to ``report``, or mark the area skipped."""
        factory = self._graph_factory
        if factory is None:
            if not _rclpy_available():
                report.skipped[self.area] = (
                    "rclpy does not import; source your ROS 2 setup (for example "
                    "/opt/ros/humble/setup.bash) to include this section"
                )
                return
            wait_s = self._wait_s

            def factory() -> RclpyGraph:
                return RclpyGraph(wait_s)

        self._check_environment(report)
        with factory() as graph:
            self._check_topics(report, graph)

    def _check_environment(self, report: DoctorReport) -> None:
        env = self._env
        distro = env.get("ROS_DISTRO", "unknown")
        rmw = env.get("RMW_IMPLEMENTATION", "")
        report.add(
            self.area,
            Level.INFO,
            f"ROS 2 {distro}, RMW {rmw or 'not set (the distro default, usually Fast DDS)'}",
        )

        domain = env.get("ROS_DOMAIN_ID", "")
        if not domain:
            report.add(self.area, Level.INFO, "ROS_DOMAIN_ID not set (domain 0).")
        elif not domain.isdigit() or int(domain) > _MAX_SAFE_DOMAIN_ID:
            report.add(
                self.area,
                Level.WARN,
                f"ROS_DOMAIN_ID={domain} is not a domain ID between 0 and {_MAX_SAFE_DOMAIN_ID}.",
                "Use the same small integer on every board that should see each other.",
            )
        else:
            report.add(self.area, Level.INFO, f"ROS_DOMAIN_ID={domain}")

        local_only = env.get("ROS_LOCALHOST_ONLY") == "1"
        auto_range = env.get("ROS_AUTOMATIC_DISCOVERY_RANGE", "").upper()
        if local_only or auto_range in ("LOCALHOST", "OFF"):
            setting = (
                "ROS_LOCALHOST_ONLY=1"
                if local_only
                else f"ROS_AUTOMATIC_DISCOVERY_RANGE={auto_range}"
            )
            report.add(
                self.area,
                Level.WARN,
                f"{setting}: topics published on other boards are invisible here.",
                "Unset it when the robot's audio comes from another computer.",
            )

        server = env.get("ROS_DISCOVERY_SERVER", "")
        if server:
            report.add(self.area, Level.INFO, f"Discovery server: ROS_DISCOVERY_SERVER={server}")
            if rmw and "fastrtps" not in rmw:
                report.add(
                    self.area,
                    Level.WARN,
                    f"ROS_DISCOVERY_SERVER is set but RMW is {rmw}.",
                    "The discovery server variable only applies to Fast DDS.",
                )
            if env.get("ROS_SUPER_CLIENT", "").lower() not in ("1", "true"):
                report.add(
                    self.area,
                    Level.INFO,
                    "ROS_SUPER_CLIENT is not set; with a discovery server, graph tools such as "
                    "this one may not see every topic.",
                    "Set ROS_SUPER_CLIENT=TRUE for `ral doctor` if topics seem to be missing.",
                )
        for var in ("ROS_STATIC_PEERS", "FASTRTPS_DEFAULT_PROFILES_FILE", "CYCLONEDDS_URI"):
            if env.get(var):
                report.add(self.area, Level.INFO, f"{var}={env[var]}")

    def _check_topics(self, report: DoctorReport, graph: Ros2Graph) -> None:
        audio = [(n, ts) for n, ts in graph.topics() if is_audio_like(n, ts)]
        if not audio:
            report.add(
                self.area,
                Level.WARN,
                "No audio-like topics are visible.",
                "Check that the robot's audio node runs, that ROS_DOMAIN_ID matches, and the "
                "discovery settings above.",
            )
            return
        for name, types in audio:
            pubs = tuple(graph.publishers(name))
            report.topics.append(TopicInfo(name, tuple(types), pubs))
            type_text = ", ".join(types)
            if not pubs:
                report.add(self.area, Level.WARN, f"{name} [{type_text}] has no publishers.")
                continue
            qos = ", ".join(
                f"{p.reliability}/{p.durability}/depth {p.depth} from {p.node}" for p in pubs
            )
            report.add(self.area, Level.OK, f"{name} [{type_text}]: {qos}")
            if any(p.reliability == "best_effort" for p in pubs):
                report.add(
                    self.area,
                    Level.WARN,
                    f"{name}: a publisher is best-effort, so a reliable subscription gets nothing.",
                    "Subscribe with best-effort reliability to receive this topic.",
                )
            vendor = [
                t
                for t in types
                if "audio" in t.lower() and not t.startswith(_STANDARD_AUDIO_PACKAGE)
            ]
            if vendor:
                report.add(
                    self.area,
                    Level.INFO,
                    f"{name} uses {', '.join(vendor)}, not audio_common_msgs.",
                    "Vendor message types need a decoder plugin (planned for M2).",
                )
