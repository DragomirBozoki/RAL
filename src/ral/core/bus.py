"""EventBus: asyncio publish and subscribe for :class:`~ral.core.events.Event`.

The bus lives on the asyncio loop. Publishing never blocks: each subscriber has a bounded
queue, and a subscriber that falls behind loses its oldest events (counted in
``dropped``) rather than stalling the engine or other subscribers. The DSP worker thread
publishes with :meth:`EventBus.publish_threadsafe`, which hands the event to the loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from types import TracebackType

from ral.core.errors import RalError
from ral.core.events import Event

__all__ = ["EventBus", "Subscription"]

log = logging.getLogger(__name__)

Listener = Callable[[Event], None]


class Subscription:
    """An async iterator over the events a subscriber asked for.

    Use it as ``async for event in sub`` or ``await sub.get()``; close it when done (or use
    it as a context manager).
    """

    def __init__(self, bus: EventBus, types: tuple[type[Event], ...] | None, maxsize: int) -> None:
        self._bus = bus
        self._types = types
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue(maxsize)
        self.dropped = 0
        """Events lost because the queue was full."""
        self.closed = False

    def wants(self, event: Event) -> bool:
        return self._types is None or isinstance(event, self._types)

    def offer(self, event: Event | None) -> None:
        if self._queue.full():
            self._queue.get_nowait()
            self.dropped += 1
        self._queue.put_nowait(event)

    async def get(self) -> Event:
        """The next event; raises ``StopAsyncIteration`` once closed and drained."""
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    def get_nowait(self) -> Event | None:
        """The next event if one is queued, else None."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def close(self) -> None:
        """Stop receiving; a pending ``async for`` ends after the queued events."""
        if not self.closed:
            self.closed = True
            self._bus._remove(self)
            self.offer(None)

    def __aiter__(self) -> AsyncIterator[Event]:
        return self

    async def __anext__(self) -> Event:
        return await self.get()

    def __enter__(self) -> Subscription:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class EventBus:
    """Fans events out to subscribers and listeners on one asyncio loop.

    Args:
        loop: The loop to deliver on; by default the running loop when first used.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop
        self._subs: list[Subscription] = []
        self._listeners: list[Listener] = []
        self.published = 0

    def bind(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Attach to ``loop`` (default: the running loop) for :meth:`publish_threadsafe`."""
        self._loop = loop or asyncio.get_running_loop()

    def subscribe(self, *types: type[Event], maxsize: int = 1000) -> Subscription:
        """Receive events of the given types (all events if none are given)."""
        if self._loop is None:
            self.bind()
        sub = Subscription(self, types or None, maxsize)
        self._subs.append(sub)
        return sub

    def add_listener(self, listener: Listener) -> Callable[[], None]:
        """Call ``listener(event)`` on the loop for every event; returns a remover.

        Listeners run inline in :meth:`publish`, so they must be quick; an exception in
        one is logged and does not reach the publisher or other listeners.
        """
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def publish(self, event: Event) -> None:
        """Deliver ``event``; call on the loop's thread."""
        self.published += 1
        for sub in list(self._subs):
            if sub.wants(event):
                sub.offer(event)
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                log.exception("Event listener %r failed on %s", listener, type(event).__name__)

    def publish_threadsafe(self, event: Event) -> None:
        """Deliver ``event`` from any thread.

        Raises:
            RalError: If the bus is not bound to a loop yet.
        """
        if self._loop is None:
            raise RalError(
                "EventBus is not bound to an event loop; call bind() from the loop first."
            )
        self._loop.call_soon_threadsafe(self.publish, event)

    def _remove(self, sub: Subscription) -> None:
        if sub in self._subs:
            self._subs.remove(sub)
