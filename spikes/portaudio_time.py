"""Spike: which clock do PortAudio callback timestamps use, and are they ever zero?

Question: DESIGN section 5 maps PortAudio's inputBufferAdcTime / outputBufferDacTime to
the RAL clock (ral.core.clock.now_ns). Is PortAudio's time base the same clock, so the
mapping is a plain unit conversion, or does it need an offset? Which host APIs report
zeros (the fallback case)?

Plays silence only (no recording): opens each host API's default output for 0.5 s and
compares callback time info with the RAL clock read inside the callback.

Run: uv run python spikes/portaudio_time.py

Result, 2026-10-19, Windows 11 x86_64, CPython 3.12.15, sounddevice 0.5.6, PortAudio
V19.7.0-devel, RAL clock = perf_counter (QPC). ~45 callbacks per host API:

    host API      currentTime - RAL clock   DAC - currentTime   stream.latency
    MME           -0.16 ms (spread 0.2)     +108 ms             100 ms
    DirectSound   -0.23 ms (spread 0.5)        0 ms (bogus)     120 ms
    WASAPI        -0.14 ms (spread 0.2)      +10 ms             22 ms
    WDM-KS        default output failed to open (-9996), not used

Conclusions:
- On Windows PortAudio's clock is QPC, the same as the RAL clock; the -0.15 ms is the
  time between PortAudio reading it and the callback reading ours.
- No host API reported zeros, but DirectSound reports DAC time == currentTime, which is
  not a DAC estimate. The fallback must treat "zero" and "equal to currentTime" alike and
  use currentTime + stream latency instead.
- Other platforms may use another time base (PortAudio's Unix util code is not
  guaranteed to use CLOCK_MONOTONIC), so the backend measures the offset
  now_ns() - currentTime in every callback and keeps its lower envelope rather than
  assuming zero. To be re-run on Linux (ALSA, PipeWire) and macOS.
"""

from __future__ import annotations

import statistics
import sys
import time

import numpy as np
import sounddevice as sd

sys.path.insert(0, "src")
from ral.core.clock import CLOCK_NAME, now_ns


def probe(device: int, rate: float, channels: int) -> dict[str, object]:
    n_max = 400
    rows = np.zeros((n_max, 4), dtype=np.float64)  # current, dac, adc, ral_clock_s
    count = [0]

    def callback(outdata, frames, t, status) -> None:  # type: ignore[no-untyped-def]
        outdata.fill(0)
        i = count[0]
        if i < n_max:
            rows[i] = (t.currentTime, t.outputBufferDacTime, t.inputBufferAdcTime, now_ns() / 1e9)
            count[0] = i + 1

    with sd.OutputStream(
        device=device,
        samplerate=rate,
        channels=channels,
        dtype="float32",
        blocksize=int(rate) // 100,
        callback=callback,
        latency="low",
    ) as stream:
        time.sleep(0.5)
        latency = stream.latency
        st0, c0 = stream.time, now_ns() / 1e9
    r = rows[: count[0]][5:]  # skip start-up callbacks
    offset_ms = (r[:, 0] - r[:, 3]) * 1e3
    return {
        "callbacks": len(r),
        "current_minus_ral_ms_median": statistics.median(offset_ms) if len(r) else None,
        "current_minus_ral_ms_spread": float(np.ptp(offset_ms)) if len(r) else None,
        "dac_minus_current_ms_median": (
            statistics.median((r[:, 1] - r[:, 0]) * 1e3) if len(r) else None
        ),
        "dac_zero": bool(len(r) and np.all(r[:, 1] == 0)),
        "current_zero": bool(len(r) and np.all(r[:, 0] == 0)),
        "stream_time_minus_ral_ms": (st0 - c0) * 1e3,
        "reported_latency_ms": latency * 1e3,
    }


def main() -> None:
    print(f"RAL clock: {CLOCK_NAME}; {sd.get_portaudio_version()[1]}")
    for api in sd.query_hostapis():
        dev = api["default_output_device"]
        if dev < 0:
            continue
        info = sd.query_devices(dev)
        try:
            res = probe(dev, info["default_samplerate"], min(2, info["max_output_channels"]))
        except Exception as exc:  # spike: report and continue
            print(f"{api['name']:22} {info['name'][:30]:30} failed: {exc}")
            continue
        print(f"{api['name']:22} {info['name'][:30]:30} {res}")


if __name__ == "__main__":
    main()
