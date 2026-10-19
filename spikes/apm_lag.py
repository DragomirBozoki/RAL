"""Spike: how far does the WebRTC APM output trail its input, per feature?

Question: the echo controller stamps clean frames with the capture time of the samples
they carry, so it needs the APM's output lag for each combination of echo cancellation
(AEC), noise suppression (NS), high-pass filter (HPF) and gain control (AGC).

Method: gated white noise through every combination at 16 kHz in 10 ms frames (silent
reverse stream when AEC is on); the lag is the shift that maximises the correlation of
output and input after the first second.

Run: uv run python spikes/apm_lag.py

Result, 2026-10-19, Windows 11 x86_64, CPython 3.12.15, livekit 1.1.20 (samples at 16 kHz):

    AEC  NS   lag
     0    0     0     (HPF and AGC add nothing in any combination)
     0    1    96     6 ms
     1    0   128     8 ms
     1    1   224    14 ms

Meaning: the lags add up (AEC 8 ms, NS 6 ms). `ral.dsp.echo.ApmProcessor` uses these
numbers, and `tests/dsp/test_echo.py` re-measures them so a livekit upgrade that changes
them fails the build.
"""

from __future__ import annotations

import itertools

import numpy as np
from livekit import rtc

FS = 16000
F = 160


def measure(aec: bool, ns: bool, hpf: bool, agc: bool, seconds: int = 4) -> int:
    rng = np.random.default_rng(0)
    n = FS * seconds
    x = (rng.standard_normal(n) * 0.1).astype(np.float32)
    x *= (np.sin(2 * np.pi * 3 * np.arange(n) / FS) > 0).astype(np.float32)
    apm = rtc.AudioProcessingModule(
        echo_cancellation=aec, noise_suppression=ns, high_pass_filter=hpf, auto_gain_control=agc
    )
    out = np.zeros(n, np.float32)
    silence = np.zeros(F, np.int16).tobytes()
    for k in range(n // F):
        if aec:
            apm.process_reverse_stream(rtc.AudioFrame(bytearray(silence), FS, 1, F))
            apm.set_stream_delay_ms(0)
        pcm = (x[k * F : (k + 1) * F] * 32767).astype(np.int16)
        frame = rtc.AudioFrame(bytearray(pcm.tobytes()), FS, 1, F)
        apm.process_stream(frame)
        out[k * F : (k + 1) * F] = np.frombuffer(frame.data, np.int16) / 32768
    a, b = x[FS:], out[FS:]
    return max(range(400), key=lambda lag: float(np.dot(a[: len(a) - lag], b[lag:])))


def main() -> None:
    for aec, ns, hpf, agc in itertools.product([False, True], repeat=4):
        print(f"aec={aec:d} ns={ns:d} hpf={hpf:d} agc={agc:d} lag={measure(aec, ns, hpf, agc)}")


if __name__ == "__main__":
    main()
