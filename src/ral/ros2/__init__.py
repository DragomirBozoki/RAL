"""ROS 2 plumbing shared by the topic backend, the bridge and ``ral_ros``.

``rclpy`` comes from a sourced ROS 2 installation, not from pip, so it is imported only
here and only when a ROS 2 feature is used (CLAUDE.md, rule 6). API checked against the
rclpy sources of the ``humble`` branch (``node.py``, ``qos.py``, ``executors.py``):
https://github.com/ros2/rclpy/tree/humble/rclpy/rclpy

:class:`Ros2Runtime` owns one rclpy ``Context``, one node and one executor thread, so RAL
never touches an application's default context. Subscription callbacks run on that
executor thread; they must only copy data (CLAUDE.md, rule 4).

Time. Message stamps are ROS time (system time, or simulated time), not the RAL clock.
:class:`RosClockMapper` measures the offset ``now_ns() - ros_now`` on every message and
keeps its lower envelope, like the PortAudio clock mapper: the smallest observed offset is
the one least delayed by scheduling. Stamps from another board are only as good as the
boards' clock sync, which ``ral doctor`` checks.
"""

from __future__ import annotations

import contextlib
import threading
from types import ModuleType
from typing import Any

from ral.core.clock import now_ns
from ral.core.errors import MissingDependencyError

__all__ = ["Ros2Runtime", "RosClockMapper", "import_rclpy", "qos_profile"]

_HINT = "source your ROS 2 setup (for example: source /opt/ros/humble/setup.bash)"


def import_rclpy() -> ModuleType:
    """``rclpy``, or a clear error when ROS 2 is not sourced."""
    try:
        import rclpy
    except ImportError:
        raise MissingDependencyError(
            "rclpy", "ros2", "The ROS 2 integration", install_hint=_HINT
        ) from None
    module: ModuleType = rclpy
    return module


def import_msgs(package: str) -> ModuleType:
    """A ROS 2 message package's ``msg`` module (``audio_common_msgs``, ``std_msgs``)."""
    import importlib

    try:
        return importlib.import_module(f"{package}.msg")
    except ImportError:
        raise MissingDependencyError(
            package,
            "ros2",
            f"Messages from {package}",
            install_hint=f"install ros-<distro>-{package.replace('_', '-')} and {_HINT}",
        ) from None


def qos_profile(
    reliability: str = "reliable", depth: int = 50, transient_local: bool = False
) -> Any:
    """An rclpy ``QoSProfile``."""
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        depth=depth,
        reliability=(
            ReliabilityPolicy.BEST_EFFORT
            if reliability == "best_effort"
            else ReliabilityPolicy.RELIABLE
        ),
        durability=(
            DurabilityPolicy.TRANSIENT_LOCAL if transient_local else DurabilityPolicy.VOLATILE
        ),
    )


class RosClockMapper:
    """Maps ROS time stamps onto the RAL clock (lower envelope of the measured offset)."""

    def __init__(self, rise_shift: int = 7) -> None:
        self._offset: int | None = None
        self._rise_shift = rise_shift

    @property
    def offset_ns(self) -> int | None:
        return self._offset

    def observe(self, ros_now_ns: int, ral_now_ns: int | None = None) -> None:
        sample = (now_ns() if ral_now_ns is None else ral_now_ns) - ros_now_ns
        if self._offset is None or sample < self._offset:
            self._offset = sample
        else:
            # Let the estimate rise slowly, so a clock step (NTP) is eventually followed.
            self._offset += (sample - self._offset) >> self._rise_shift

    def to_ral_ns(self, ros_ns: int) -> int:
        assert self._offset is not None, "observe() first"
        return ros_ns + self._offset


class Ros2Runtime:
    """One rclpy context, node and executor thread.

    Args:
        node_name: Name of the node in the ROS graph.
        namespace: Node namespace.
    """

    def __init__(self, node_name: str = "ral", namespace: str = "") -> None:
        self.rclpy = import_rclpy()
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor

        self.context = Context()
        self.rclpy.init(context=self.context)
        self.node = self.rclpy.create_node(
            node_name, namespace=namespace or None, context=self.context
        )
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self._thread = threading.Thread(target=self._spin, name="ral-ros2", daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        # The context may be shut down under the executor at exit; that is normal.
        with contextlib.suppress(Exception):
            self.executor.spin()

    def ros_now_ns(self) -> int:
        return int(self.node.get_clock().now().nanoseconds)

    def close(self) -> None:
        """Stop the executor and release the context."""
        self.executor.shutdown()
        self.node.destroy_node()
        if self.context.ok():
            self.rclpy.shutdown(context=self.context)
        self._thread.join(timeout=2.0)
