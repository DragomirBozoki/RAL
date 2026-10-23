"""Spike: does drift compensation keep AEC3 working when the speaker's clock is off?

Scene (ral.bench.drift): 60 s of a voice-like far end, the speaker's clock 500 ppm fast
against the mics (the echo slides 30 ms per minute against the reference), echo through
the synthetic path, software_ref with lead 20 ms; ERLE over the last 20 s.

Run: uv run python spikes/drift_erle.py

Result, 2026-10-23, Windows 11 x86_64, CPython 3.12.15, livekit 1.1.20:

    drift 500 ppm, last 20 s: ERLE without compensation 31.5 dB, with 43.0 dB

Meaning: AEC3's delay estimator follows a slow slide for a while but loses about 11 dB by
the end of a minute; resampling the reference onto the mic clock keeps the full ERLE.
"""

from __future__ import annotations

from ral.bench.drift import drift_erle
from ral.dsp.echo import ApmProcessor

if __name__ == "__main__":
    off = drift_erle(ApmProcessor(), compensate=False)
    on = drift_erle(ApmProcessor(), compensate=True)
    print(f"drift 500 ppm, last 20 s: ERLE without compensation {off:.1f} dB, with {on:.1f} dB")
