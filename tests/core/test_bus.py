import asyncio
import threading

import pytest

from ral.core import EventBus, RalError
from ral.core import events as ev


async def test_subscribers_get_events_in_order() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    for t in range(3):
        bus.publish(ev.SpeechStarted(t, "s"))
    assert [(await sub.get()).t_ns for _ in range(3)] == [0, 1, 2]


async def test_type_filter() -> None:
    bus = EventBus()
    speech = bus.subscribe(ev.SpeechStarted, ev.SpeechEnded)
    bus.publish(ev.BargeIn(1, "s", None))
    bus.publish(ev.SpeechStarted(2, "s"))
    assert (await speech.get()).t_ns == 2
    assert speech.get_nowait() is None


async def test_slow_subscriber_loses_oldest_and_never_blocks() -> None:
    bus = EventBus()
    sub = bus.subscribe(maxsize=3)
    for t in range(10):
        bus.publish(ev.SpeechStarted(t, "s"))
    assert sub.dropped == 7
    assert [(await sub.get()).t_ns for _ in range(3)] == [7, 8, 9]


async def test_close_ends_async_for_after_queued_events() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    bus.publish(ev.SpeechStarted(1, "s"))
    sub.close()
    bus.publish(ev.SpeechStarted(2, "s"))  # not delivered after close
    assert [e.t_ns async for e in sub] == [1]


async def test_context_manager_closes() -> None:
    bus = EventBus()
    with bus.subscribe() as sub:
        pass
    assert sub.closed


async def test_publish_threadsafe_from_another_thread() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    th = threading.Thread(
        target=lambda: [bus.publish_threadsafe(ev.SpeechStarted(t, "s")) for t in range(50)]
    )
    th.start()
    got = [(await asyncio.wait_for(sub.get(), 2)).t_ns for _ in range(50)]
    th.join()
    assert got == list(range(50))


def test_publish_threadsafe_needs_a_loop() -> None:
    with pytest.raises(RalError):
        EventBus().publish_threadsafe(ev.SpeechStarted(1, "s"))


async def test_listener_errors_are_isolated(caplog: pytest.LogCaptureFixture) -> None:
    bus = EventBus()
    seen: list[int] = []

    def bad(event: ev.Event) -> None:
        raise ValueError("boom")

    bus.add_listener(bad)
    remove = bus.add_listener(lambda e: seen.append(e.t_ns))
    bus.publish(ev.SpeechStarted(1, "s"))
    remove()
    bus.publish(ev.SpeechStarted(2, "s"))
    assert seen == [1]
    assert "boom" in caplog.text
