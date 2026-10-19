"""A fake ``sounddevice`` module: two host APIs, five devices, streams driven by hand."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class FakePortAudioError(Exception):
    pass


class FakeStream:
    def __init__(self, sd: FakeSd, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.latency = 0.020
        self.state = "open"
        self._sd = sd
        sd.streams.append(self)

    def start(self) -> None:
        if self._sd.fail_open:
            raise FakePortAudioError("Invalid sample rate [PaErrorCode -9997]")
        self.state = "started"

    def stop(self) -> None:
        self.state = "stopped"

    def close(self) -> None:
        self.state = "closed"


def raw_device(index: int, api: int, name: str, inputs: int, outputs: int, rate: float) -> dict:
    return {
        "index": index,
        "hostapi": api,
        "name": name,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
        "default_samplerate": rate,
        "default_low_input_latency": 0.003,
        "default_low_output_latency": 0.01,
    }


class FakeSd:
    PortAudioError = FakePortAudioError

    def __init__(self) -> None:
        self.streams: list[FakeStream] = []
        self.fail_open = False
        self.apis = [
            {"name": "MME", "default_input_device": 0, "default_output_device": 1},
            {"name": "Windows WASAPI", "default_input_device": 2, "default_output_device": 3},
        ]
        self.devices = [
            raw_device(0, 0, "Microphone Array (Realtek(R) Au", 2, 0, 44100.0),
            raw_device(1, 0, "Speakers (Realtek(R) Audio)", 0, 2, 44100.0),
            raw_device(2, 1, "Microphone Array (Realtek(R) Audio)", 2, 0, 48000.0),
            raw_device(3, 1, "Speakers (Realtek(R) Audio)", 0, 2, 48000.0),
            raw_device(4, 1, "Odd Card", 1, 1, 22050.0),
        ]

    def query_hostapis(self) -> list[dict]:
        return self.apis

    def query_devices(self) -> list[dict]:
        return self.devices

    def get_portaudio_version(self) -> tuple[int, str]:
        return 1246976, "PortAudio fake"

    def InputStream(self, **kwargs: Any) -> FakeStream:  # sounddevice's name
        return FakeStream(self, **kwargs)

    def OutputStream(self, **kwargs: Any) -> FakeStream:
        return FakeStream(self, **kwargs)


def time_info(current: float, adc: float = 0.0, dac: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(currentTime=current, inputBufferAdcTime=adc, outputBufferDacTime=dac)


def flags(**set_flags: bool) -> SimpleNamespace:
    return SimpleNamespace(
        input_overflow=set_flags.get("input_overflow", False),
        output_underflow=set_flags.get("output_underflow", False),
    )
