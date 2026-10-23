"""``PlayAudio`` and ``Say`` actions, ``stop`` and ``duck`` services (DESIGN section 7).

The types come from ``integrations/ros2/ral_interfaces``, a colcon package; RAL imports it
lazily, like every ROS 2 module. API checked against rclpy humble (``action/server.py``:
``ActionServer(node, type, name, execute_callback, *, callback_group, goal_callback,
cancel_callback)``; ``GoalResponse``/``CancelResponse``; the execute callback returns the
``Result`` after calling ``succeed``, ``abort`` or ``canceled`` on the goal handle;
``node.create_service(type, name, callback(request, response), callback_group=...)``).

Threads. rclpy runs these callbacks on the runtime's executor workers (a reentrant
callback group), never on the asyncio loop. They hand work to the loop with
``run_coroutine_threadsafe`` and wait for it there; an execute callback blocks its worker
until the audio has played, polling every 100 ms to publish feedback and to pass a cancel
request on as a fade-out. All audio still goes through the engine's playback service
(CLAUDE.md, reference invariant).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

from ral.core.errors import MissingDependencyError
from ral.core.types import AudioFrame
from ral.playback.stream import PcmStream

if TYPE_CHECKING:
    from ral.engine.engine import Engine
    from ral.ros2 import Ros2Runtime

__all__ = ["Ros2Actions", "TextToSpeechFn", "import_interfaces"]

log = logging.getLogger(__name__)

TextToSpeechFn = Callable[[str], AsyncIterator[AudioFrame]]
"""Text in, mono float32 frames at the engine rate out. TODO(M4): the ``TextToSpeech``
port of DESIGN section 4 with reference adapters replaces this callable."""

_POLL_S = 0.1


def import_interfaces() -> tuple[ModuleType, ModuleType]:
    """``ral_interfaces.action`` and ``ral_interfaces.srv``, or a clear error."""
    try:
        return (
            importlib.import_module("ral_interfaces.action"),
            importlib.import_module("ral_interfaces.srv"),
        )
    except ImportError:
        raise MissingDependencyError(
            "ral_interfaces",
            "ros2",
            "The ROS 2 actions and services",
            install_hint=(
                "build integrations/ros2/ral_interfaces with colcon and source the workspace "
                "(see its README)"
            ),
        ) from None


@dataclass
class _Goal:
    """One running goal, as seen from the loop."""

    frames: int = 0
    handle: Any = None


class Ros2Actions:
    """Serves the actions and services for one engine on a runtime's node.

    Args:
        runtime: Node, executor and reentrant callback group.
        engine: The running engine.
        loop: The engine's event loop.
        namespace: Name prefix (``/ral``).
        tts: Speaks ``Say`` goals; without it they are rejected.
    """

    def __init__(
        self,
        runtime: Ros2Runtime,
        engine: Engine,
        loop: asyncio.AbstractEventLoop,
        namespace: str = "/ral",
        tts: TextToSpeechFn | None = None,
    ) -> None:
        from rclpy.action import ActionServer

        self.engine = engine
        self.loop = loop
        self.tts = tts
        self.node = runtime.node
        action, srv = import_interfaces()
        self.action, self.srv = action, srv
        group = runtime.reentrant
        self._servers = [
            ActionServer(
                self.node,
                action.PlayAudio,
                f"{namespace}/play_audio",
                self._execute_play,
                callback_group=group,
                goal_callback=self._accept_play,
                cancel_callback=self._accept_cancel,
            ),
            ActionServer(
                self.node,
                action.Say,
                f"{namespace}/say",
                self._execute_say,
                callback_group=group,
                goal_callback=self._accept_say,
                cancel_callback=self._accept_cancel,
            ),
        ]
        self._services = [
            self.node.create_service(
                srv.Stop, f"{namespace}/stop", self._on_stop, callback_group=group
            ),
            self.node.create_service(
                srv.Duck, f"{namespace}/duck", self._on_duck, callback_group=group
            ),
        ]

    def close(self) -> None:
        for server in self._servers:
            server.destroy()
        for service in self._services:
            self.node.destroy_service(service)
        self._servers, self._services = [], []

    # Goal and cancel decisions (executor thread)

    def _accept_play(self, goal: Any) -> Any:
        from rclpy.action import GoalResponse

        ok = len(goal.data) % 2 == 0 and len(goal.data) > 0 and 0 <= goal.sample_rate <= 192_000
        if not ok:
            log.warning("PlayAudio goal rejected: give whole 16-bit samples and a valid rate.")
        return GoalResponse.ACCEPT if ok else GoalResponse.REJECT

    def _accept_say(self, goal: Any) -> Any:
        from rclpy.action import GoalResponse

        if self.tts is None:
            log.warning("Say goal rejected: the ROS 2 bridge has no TextToSpeech adapter.")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT if goal.text.strip() else GoalResponse.REJECT

    def _accept_cancel(self, _goal_handle: Any) -> Any:
        from rclpy.action import CancelResponse

        return CancelResponse.ACCEPT

    # Execution (executor thread, blocking until the audio has played)

    def _execute_play(self, goal_handle: Any) -> Any:
        req = goal_handle.request
        data = bytes(req.data)
        rate = int(req.sample_rate)

        def frames() -> AsyncIterator[AudioFrame]:
            fmt = self.engine.engine_fmt
            stream = PcmStream(rate or fmt.sample_rate, fmt)
            stream.push(data)
            stream.end()
            return aiter(stream)

        return self._execute(goal_handle, self.action.PlayAudio, frames)

    def _execute_say(self, goal_handle: Any) -> Any:
        text = str(goal_handle.request.text)
        tts = self.tts
        assert tts is not None
        return self._execute(goal_handle, self.action.Say, lambda: tts(text))

    def _execute(
        self,
        goal_handle: Any,
        action_type: Any,
        frames: Callable[[], AsyncIterator[AudioFrame]],
    ) -> Any:
        req = goal_handle.request
        goal = _Goal()
        frame_s = self.engine.engine_fmt.frame_ns / 1e9
        future = asyncio.run_coroutine_threadsafe(
            self._play(
                goal, frames, str(req.tag) or "ros2", int(req.priority), bool(req.interruptible)
            ),
            self.loop,
        )
        cancel_sent = False
        result = action_type.Result()
        while True:
            try:
                interrupted = future.result(timeout=_POLL_S)
                break
            except concurrent.futures.TimeoutError:
                pass
            except Exception as exc:  # the audio could not be played
                log.warning("ROS 2 goal failed: %s", exc)
                goal_handle.abort()
                result.reason = f"failed: {exc}"
                return result
            feedback = action_type.Feedback()
            feedback.played_s = goal.frames * frame_s
            goal_handle.publish_feedback(feedback)
            if goal_handle.is_cancel_requested and not cancel_sent and goal.handle is not None:
                cancel_sent = True
                asyncio.run_coroutine_threadsafe(self._stop_handle(goal.handle), self.loop)
        result.played_s = goal.frames * frame_s
        result.interrupted = bool(interrupted or cancel_sent)
        if cancel_sent:
            result.reason = "cancelled"
            goal_handle.canceled()
        else:
            result.reason = "interrupted" if interrupted else "finished"
            goal_handle.succeed()
        return result

    async def _play(
        self,
        goal: _Goal,
        frames: Callable[[], AsyncIterator[AudioFrame]],
        tag: str,
        priority: int,
        interruptible: bool,
    ) -> bool:
        playback = self.engine.playback
        assert playback is not None

        async def counted() -> AsyncIterator[AudioFrame]:
            async for frame in frames():
                goal.frames += 1
                yield frame

        goal.handle = playback.play(
            counted(), priority=priority, interruptible=interruptible, tag=tag
        )
        return await playback.wait(goal.handle)

    async def _stop_handle(self, handle: Any) -> None:
        assert self.engine.playback is not None
        await self.engine.playback.stop(handle)

    # Services (executor thread)

    def _on_stop(self, request: Any, response: Any) -> Any:
        response.stopped = self._call(self._stop(int(request.fade_ms), bool(request.all)))
        return response

    async def _stop(self, fade_ms: int, everything: bool) -> bool:
        playback = self.engine.playback
        assert playback is not None
        current = playback.current
        playback.interrupt(fade_ms)
        if everything and current is not None:
            await playback.stop(current, fade_ms)
        return current is not None

    def _on_duck(self, request: Any, response: Any) -> Any:
        async def duck() -> bool:
            assert self.engine.playback is not None
            self.engine.playback.duck(float(request.gain_db), int(request.duration_ms))
            return True

        response.ok = self._call(duck())
        return response

    def _call(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=5.0)
