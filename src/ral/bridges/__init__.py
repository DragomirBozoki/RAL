"""Bridges: connections between the engine and outside stacks (DESIGN section 7).

A bridge implements the :class:`Bridge` protocol. The ``bridges`` block of a profile
names the bridges to start and their settings, for example::

    bridges:
      websocket: {host: 127.0.0.1, port: 8765}
      livekit: {url: "${LIVEKIT_URL}", room: lab}

Built-in bridges are found by name here; plugins register more through the
``ral.bridges`` entry point group. Every bridge plays agent audio through the engine's
playback service, which keeps the echo reference intact (CLAUDE.md, reference invariant).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ral.core.errors import ConfigError

if TYPE_CHECKING:
    from ral.engine.engine import Engine

__all__ = ["Bridge", "BridgeFactory", "make_bridge", "start_bridges", "stop_bridges"]

log = logging.getLogger(__name__)


@runtime_checkable
class Bridge(Protocol):
    """Connects the engine to code RAL does not own."""

    async def attach(self, engine: Engine) -> None: ...

    async def detach(self) -> None: ...


BridgeFactory = Callable[[Mapping[str, Any]], Bridge]


def _websocket(cfg: Mapping[str, Any]) -> Bridge:
    from ral.bridges.websocket import WebSocketBridge

    return WebSocketBridge.from_config(cfg)


def _livekit(cfg: Mapping[str, Any]) -> Bridge:
    from ral.bridges.livekit import LiveKitBridge

    return LiveKitBridge.from_config(cfg)


_BUILTIN: dict[str, BridgeFactory] = {"websocket": _websocket, "livekit": _livekit}


def make_bridge(name: str, cfg: Mapping[str, Any]) -> Bridge:
    """Build one bridge from its profile settings.

    Raises:
        ConfigError: If no bridge has that name.
    """
    factory = _BUILTIN.get(name)
    if factory is None:
        for ep in entry_points(group="ral.bridges"):
            if ep.name == name:
                factory = ep.load()
                break
    if factory is None:
        raise ConfigError(
            f"Unknown bridge {name!r}; built-in bridges: {', '.join(_BUILTIN)}. Plugins add "
            f"bridges through the 'ral.bridges' entry point."
        )
    return factory(cfg)


async def start_bridges(engine: Engine, config: Mapping[str, Mapping[str, Any]]) -> list[Bridge]:
    """Attach every bridge in ``config``; on failure, detach the ones already started."""
    started: list[Bridge] = []
    try:
        for name, cfg in config.items():
            bridge = make_bridge(name, cfg or {})
            await bridge.attach(engine)
            started.append(bridge)
            log.info("Bridge %s attached", name)
    except BaseException:
        await stop_bridges(started)
        raise
    return started


async def stop_bridges(bridges: list[Bridge]) -> None:
    """Detach bridges in reverse order, logging (not raising) failures."""
    for bridge in reversed(bridges):
        try:
            await bridge.detach()
        except Exception:
            log.exception("Bridge %r failed to detach", bridge)
