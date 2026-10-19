import dataclasses
import json

import pytest

from ral.core import FormatError, event_from_json, event_to_json
from ral.core import events as ev

EXAMPLES: list[ev.Event] = [
    ev.SpeechStarted(1, "s"),
    ev.SpeechEnded(2, "s", t_start_ns=1),
    ev.BargeIn(3, "s", interrupted_tag="answer"),
    ev.BargeIn(3, "s", interrupted_tag=None),
    ev.PlaybackStarted(4, "s", handle_id=7, tag="tts", t_request_ns=2),
    ev.PlaybackEnded(5, "s", handle_id=7, tag="tts", interrupted=True),
    ev.WordSpoken(6, "s", word="hello", t_end_ns=9, handle_id=7),
    ev.DirectionOfArrival(7, "s", azimuth_deg=30.0),
    ev.EchoLeak(8, "s", correlation=0.8),
    ev.DeviceLost(9, "s", device_id="portaudio:wasapi:3", reason="unplugged"),
    ev.Xrun(10, "s", stream="capture", kind="overrun", count=2),
    ev.Clipping(11, "s", channel=0, share=0.01),
    ev.ProfileReloaded(12, "s", profile="laptop.yaml"),
    ev.RecordingStarted(13, "s", path="recordings/a.mcap"),
    ev.RecordingStopped(14, "s", path="recordings/a.mcap"),
]


@pytest.mark.parametrize("event", EXAMPLES, ids=lambda e: type(e).__name__)
def test_json_round_trip(event: ev.Event) -> None:
    data = event_to_json(event)
    assert data["type"] == type(event).__name__
    assert data["t_ns"] == event.t_ns
    assert event_from_json(json.loads(json.dumps(data))) == event


def test_events_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.SpeechStarted(1, "s").t_ns = 2  # type: ignore[misc]


def test_subclasses_defined_elsewhere_are_found() -> None:
    @dataclasses.dataclass(frozen=True)
    class PluginEvent(ev.Event):
        value: int

    assert event_from_json({"type": "PluginEvent", "t_ns": 1, "session_id": "s", "value": 3}) == (
        PluginEvent(1, "s", 3)
    )


@pytest.mark.parametrize(
    "data",
    [
        {"type": "Nope", "t_ns": 1, "session_id": "s"},
        {"type": "Event", "t_ns": 1, "session_id": "s"},
        {"type": "SpeechStarted", "t_ns": 1},
        {"type": "SpeechStarted", "t_ns": 1, "session_id": "s", "extra": 1},
        {"t_ns": 1, "session_id": "s"},
    ],
)
def test_bad_json_is_rejected(data: dict[str, object]) -> None:
    with pytest.raises(FormatError):
        event_from_json(data)
