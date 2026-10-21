import sys

import pytest

from fakes.sounddevice import FakeSd, raw_device
from ral.core import DeviceError, MissingDependencyError
from ral.devices.doctor import DoctorReport, Level
from ral.devices.doctor.portaudio_scan import PortAudioScanner
from ral.devices.portaudio import PortAudioBackend


@pytest.fixture(autouse=True)
def windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")


def scan(sd: FakeSd) -> DoctorReport:
    report = DoctorReport()
    PortAudioScanner(lambda: PortAudioBackend(sd)).scan(report)  # type: ignore[arg-type]
    return report


def test_lists_devices_and_checks_defaults() -> None:
    report = scan(FakeSd())
    messages = [(f.level, f.message) for f in report.findings]
    assert messages[0] == (Level.OK, "PortAudio library loaded: PortAudio fake")
    assert messages[1] == (Level.INFO, "5 devices by host API: Windows WASAPI 3, MME 2")
    assert messages[2][0] is Level.OK
    assert messages[2][1].startswith("Default capture: Microphone Array (Realtek(R) Audio)")
    assert "48000 Hz (portaudio:wasapi:2)" in messages[2][1]
    assert messages[3][1].startswith("Default playback: Speakers (Realtek(R) Audio)")
    assert messages[4] == (
        Level.INFO,
        "2 device names appear under several host APIs; RAL uses Windows WASAPI.",
    )
    assert [d.id for d in report.devices][:2] == ["portaudio:wasapi:2", "portaudio:wasapi:3"]
    assert report.area_levels() == {"audio": Level.OK}


def test_missing_capture_device_points_to_ros2() -> None:
    sd = FakeSd()
    for api in sd.apis:
        api["default_input_device"] = -1
    finding = next(f for f in scan(sd).findings if "capture" in f.message)
    assert finding.level is Level.WARN
    assert finding.message == "No capture device on this host."
    assert finding.hint is not None
    assert "ROS 2" in finding.hint


def test_default_device_with_unusable_rate_is_a_warning() -> None:
    sd = FakeSd()
    sd.apis[1]["default_input_device"] = 4  # the 22050 Hz card
    finding = next(f for f in scan(sd).findings if "22050" in f.message)
    assert finding.level is Level.WARN


def test_no_devices_is_an_error() -> None:
    sd = FakeSd()
    sd.devices = []
    report = scan(sd)
    assert report.findings[-1].level is Level.ERROR
    assert report.has_errors


def test_missing_wasapi_is_a_warning() -> None:
    sd = FakeSd()
    sd.apis[1]["name"] = "Windows DirectSound"
    finding = next(f for f in scan(sd).findings if "preferred host API" in f.message)
    assert finding.level is Level.WARN


def test_missing_sounddevice_and_missing_library() -> None:
    def no_extra() -> PortAudioBackend:
        raise MissingDependencyError("sounddevice", "audio", "The PortAudio backend")

    def no_library() -> PortAudioBackend:
        raise DeviceError("The PortAudio library could not be loaded")

    report = DoctorReport()
    PortAudioScanner(no_extra).scan(report)
    PortAudioScanner(no_library).scan(report)
    first, second = report.findings
    assert (first.level, first.hint) == (Level.WARN, 'pip install "ral-audio[audio]"')
    assert second.level is Level.ERROR
    assert second.hint is not None
    assert "libportaudio2" in second.hint


def test_known_array_is_named_with_its_echo_mode_and_layout() -> None:
    sd = FakeSd()
    sd.devices.append(raw_device(5, 1, "ReSpeaker 4 Mic Array (UAC1.0)", 6, 2, 16000.0))
    report = scan(sd)
    known = [f for f in report.findings if f.area == "devices"]
    assert len(known) == 1
    assert "ReSpeaker USB Mic Array v2.0" in known[0].message
    assert known[0].hint is not None
    assert "echo.mode: hardware" in known[0].hint
    assert "processed, mic, mic, mic, mic, reference" in known[0].hint
