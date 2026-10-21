import pytest

from ral.core import DeviceInfo
from ral.devicedb import find_entry, load_devicedb


def device(name: str, inputs: int, usb: str = "") -> DeviceInfo:
    extra = {"usb": usb} if usb else {}
    return DeviceInfo("x:1", name, "portaudio", "ALSA", inputs, 2, 16000.0, extra)


def test_database_loads_and_every_entry_cites_a_source() -> None:
    entries = load_devicedb()
    assert len({e.id for e in entries}) == len(entries)
    for entry in entries:
        assert entry.sources
        assert all(s.startswith("https://") for s in entry.sources)
        if entry.channels is not None:
            assert entry.match.input_channels in (None, len(entry.channels))


@pytest.mark.parametrize(
    ("name", "inputs", "expected"),
    [
        ("ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:1,0)", 6, "seeed-respeaker-usb-v2-6ch"),
        ("ReSpeaker 4 Mic Array (UAC1.0)", 1, "seeed-respeaker-usb-v2-1ch"),
        ("reSpeaker XVF3800 4-Mic Array", 2, "xmos-xvf3800"),
        ("Microphone Array (Realtek(R) Audio)", 2, None),
    ],
)
def test_matching_by_name_and_channels(name: str, inputs: int, expected: str | None) -> None:
    entry = find_entry(device(name, inputs))
    assert (entry.id if entry else None) == expected


def test_matching_by_usb_id_when_the_name_differs() -> None:
    entry = find_entry(device("USB Audio Device", 6, usb="2886:0018"))
    assert entry is not None
    assert entry.channels == ["processed", "mic", "mic", "mic", "mic", "reference"]


def test_xvf3800_facts_come_from_the_datasheet() -> None:
    entry = find_entry(device("XVF3800", 2))
    assert entry is not None
    assert entry.reference_channel == "left"
    assert entry.aec_tail_ms == 192
    assert entry.channels is None  # not in the public datasheet: ral probe maps it
