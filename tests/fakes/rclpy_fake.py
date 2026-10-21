"""A fake rclpy plus the message packages RAL uses, for tests without ROS 2.

Only the API surface RAL calls (checked against rclpy's humble sources): ``init``,
``shutdown``, ``create_node``, ``context.Context``, ``qos`` policies and profile,
``executors.SingleThreadedExecutor``, node publishers, subscriptions, clock and graph.
Messages published on any node reach every subscription of the same topic and type,
delivered on the executor's own thread, as with a real executor. ROS time is wall time
(``time.time_ns``), deliberately not the RAL clock, so time mapping is exercised.
"""

from __future__ import annotations

import array
import enum
import queue
import sys
import threading
import time
import types
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


class Graph:
    def __init__(self) -> None:
        self.subs: dict[str, list[tuple[Any, Any, Any]]] = defaultdict(
            list
        )  # topic -> (type, cb, executor)
        self.pubs: dict[str, list[Any]] = defaultdict(list)
        self.latched: dict[str, list[Any]] = defaultdict(list)


GRAPH = Graph()


# Messages ------------------------------------------------------------------------------


@dataclass
class Time:
    sec: int = 0
    nanosec: int = 0


@dataclass
class Header:
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class AudioData:
    data: Any = field(default_factory=lambda: array.array("B"))


@dataclass
class AudioDataStamped:
    header: Header = field(default_factory=Header)
    audio: AudioData = field(default_factory=AudioData)


@dataclass
class AudioInfo:
    channels: int = 0
    sample_rate: int = 0
    sample_format: str = ""
    bitrate: int = 0
    coding_format: str = ""


@dataclass
class String:
    data: str = ""


# rclpy ---------------------------------------------------------------------------------


class ReliabilityPolicy(enum.IntEnum):
    RELIABLE = 1
    BEST_EFFORT = 2


class DurabilityPolicy(enum.IntEnum):
    TRANSIENT_LOCAL = 1
    VOLATILE = 2


class QoSProfile:
    def __init__(self, **kwargs: Any) -> None:
        assert "depth" in kwargs
        self.depth = kwargs["depth"]
        self.reliability = kwargs.get("reliability", ReliabilityPolicy.RELIABLE)
        self.durability = kwargs.get("durability", DurabilityPolicy.VOLATILE)


class Context:
    def __init__(self) -> None:
        self._ok = False

    def ok(self) -> bool:
        return self._ok


def init(*, args: Any = None, context: Context | None = None) -> None:
    assert context is not None
    context._ok = True


def shutdown(*, context: Context | None = None, uninstall_handlers: Any = None) -> None:
    assert context is not None
    context._ok = False


class _Clock:
    def now(self) -> Any:
        return types.SimpleNamespace(nanoseconds=time.time_ns())


class Publisher:
    def __init__(self, msg_type: Any, topic: str, qos: QoSProfile) -> None:
        self.msg_type, self.topic, self.qos = msg_type, topic, qos
        self.sent: list[Any] = []

    def publish(self, msg: Any) -> None:
        assert isinstance(msg, self.msg_type), f"{type(msg)} on a {self.msg_type} publisher"
        self.sent.append(msg)
        if self.qos.durability == DurabilityPolicy.TRANSIENT_LOCAL:
            GRAPH.latched[self.topic] = [msg]
        for msg_type, cb, executor in list(GRAPH.subs[self.topic]):
            if msg_type is self.msg_type and executor is not None:
                executor.queue.put((cb, msg))


class Subscription:
    def __init__(self, msg_type: Any, topic: str, cb: Any, node: Node) -> None:
        self.msg_type, self.topic, self.cb, self.node = msg_type, topic, cb, node


class Node:
    def __init__(
        self, name: str, namespace: str | None = None, context: Context | None = None
    ) -> None:
        self.name, self.namespace, self.context = name, namespace, context
        self.executor: SingleThreadedExecutor | None = None
        self._subs: list[Subscription] = []

    def create_publisher(
        self, msg_type: Any, topic: str, qos_profile: QoSProfile, **_: Any
    ) -> Publisher:
        pub = Publisher(msg_type, topic, qos_profile)
        GRAPH.pubs[topic].append(pub)
        return pub

    def create_subscription(
        self, msg_type: Any, topic: str, callback: Any, qos_profile: QoSProfile, **_: Any
    ) -> Subscription:
        sub = Subscription(msg_type, topic, callback, self)
        self._subs.append(sub)
        GRAPH.subs[topic].append((msg_type, callback, self.executor))
        if self.executor is not None:
            for msg in GRAPH.latched.get(topic, []):
                self.executor.queue.put((callback, msg))
        return sub

    def destroy_subscription(self, sub: Subscription) -> bool:
        GRAPH.subs[sub.topic] = [s for s in GRAPH.subs[sub.topic] if s[1] is not sub.cb]
        return True

    def destroy_publisher(self, pub: Publisher) -> bool:
        if pub in GRAPH.pubs[pub.topic]:
            GRAPH.pubs[pub.topic].remove(pub)
        return True

    def destroy_node(self) -> None:
        for sub in self._subs:
            self.destroy_subscription(sub)

    def get_clock(self) -> _Clock:
        return _Clock()

    def get_topic_names_and_types(self, no_demangle: bool = False) -> list[tuple[str, list[str]]]:
        names = {
            t: [f"audio_common_msgs/msg/{p.msg_type.__name__}"]
            for t, pubs in GRAPH.pubs.items()
            for p in pubs
            if p.msg_type.__name__.startswith("Audio")
        }
        return sorted(names.items())


def create_node(
    node_name: str, *, context: Context | None = None, namespace: str | None = None, **_: Any
) -> Node:
    return Node(node_name, namespace, context)


class SingleThreadedExecutor:
    def __init__(self, *, context: Context | None = None) -> None:
        self.context = context
        self.queue: queue.Queue[tuple[Any, Any] | None] = queue.Queue()
        self.nodes: list[Node] = []

    def add_node(self, node: Node) -> bool:
        node.executor = self
        self.nodes.append(node)
        return True

    def spin(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                return
            cb, msg = item
            cb(msg)

    def shutdown(self, timeout_sec: float | None = None) -> bool:
        self.queue.put(None)
        return True


def install(monkeypatch: Any) -> Graph:
    """Put the fake modules in ``sys.modules`` for one test; returns a fresh graph."""
    global GRAPH
    GRAPH = Graph()
    me = sys.modules[__name__]

    def module(name: str, **attrs: Any) -> types.ModuleType:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    rclpy = module("rclpy", init=init, shutdown=shutdown, create_node=create_node)
    rclpy.context = module("rclpy.context", Context=Context)  # type: ignore[attr-defined]
    rclpy.qos = module(  # type: ignore[attr-defined]
        "rclpy.qos",
        QoSProfile=QoSProfile,
        ReliabilityPolicy=ReliabilityPolicy,
        DurabilityPolicy=DurabilityPolicy,
    )
    rclpy.executors = module("rclpy.executors", SingleThreadedExecutor=SingleThreadedExecutor)  # type: ignore[attr-defined]
    module("audio_common_msgs")
    module(
        "audio_common_msgs.msg",
        AudioData=AudioData,
        AudioDataStamped=AudioDataStamped,
        AudioInfo=AudioInfo,
    )
    module("std_msgs")
    module("std_msgs.msg", String=String, Header=Header)
    _ = me
    return GRAPH


def wait_for(pred: Any, timeout: float = 3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


def hal(
    topic: str = "/hal/audio/capture",
) -> tuple[Node, Publisher, SingleThreadedExecutor, threading.Thread]:
    """A robot HAL publishing audio on ``topic`` (its own node and executor)."""
    node = Node("hal")
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
    pub = node.create_publisher(AudioDataStamped, topic, QoSProfile(depth=10))
    return node, pub, ex, th
