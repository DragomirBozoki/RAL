"""A drifting-clock scene: the speaker runs fast against the mics (DESIGN section 5).

The sink renders a voice in its own samples; its clock runs ``eps`` fast, so in mic samples
the echo is that voice slightly sped up (resampled with :class:`DriftCorrector`, -92 dB
precise). The reference buffer receives the sink's blocks with their real DAC times, and
``software_ref`` runs with or without drift compensation. Used by tests and
``spikes/drift_erle.py``.
"""

from __future__ import annotations

import numpy as np

from ral.bench.synthetic import echo_path, fft_convolve, voice
from ral.core.processing import FrameContext, Processor
from ral.core.types import AudioFormat, AudioFrame, ChannelMap
from ral.dsp.drift import DriftCorrector, DriftEstimator
from ral.metrics.echo import erle_db
from ral.playback.reference import ReferenceBuffer

__all__ = ["drift_erle"]

T0 = 10**12


def drift_erle(
    stage: Processor,
    *,
    compensate: bool,
    eps_ppm: float = 500.0,
    seconds: int = 60,
    measure_from_s: float = 40.0,
    lag_ms: int = 8,
    rate: int = 16000,
) -> float:
    """ERLE (dB) of ``stage`` from ``measure_from_s`` on, with the sink ``eps_ppm`` fast."""
    eps = eps_ppm * 1e-6
    n = rate * seconds
    fn = rate // 100
    sink = voice(n + rate, rate, 120.0, 1) * 0.1
    c = DriftCorrector(max_ppm=max(5000.0, 2 * eps_ppm))
    c.ratio = 1 / (1 + eps)
    blocks = [c.process(sink[i : i + 1600].astype(np.float32)) for i in range(0, len(sink), 1600)]
    on_mic_clock = np.concatenate(blocks)[:n].astype(np.float64)
    echo = fft_convolve(on_mic_clock, echo_path(20, rate, 3) * 0.7)
    noise = np.random.default_rng(4).standard_normal(n) * 10 ** (-65 / 20)
    mic = (echo + noise).astype(np.float32)

    fmt = AudioFormat(rate, 1, "float32")
    ref = ReferenceBuffer(fmt, rate, drift=compensate, capacity_s=5.0)
    capture = DriftEstimator(rate)
    ref.capture_estimator = capture
    cmap = ChannelMap.uniform(1)
    out = []
    rendered = 0
    frame_ns = 10_000_000
    for k in range(n // fn):
        t = T0 + k * frame_ns
        capture.observe(k * fn, t)
        while rendered * frame_ns / (1 + eps) < (k + 10) * frame_ns:  # 100 ms ahead
            block = sink[rendered * fn : (rendered + 1) * fn, None].astype(np.float32)
            ref.push(block, T0 + int(rendered * frame_ns / (1 + eps)))
            rendered += 1
        frame = AudioFrame(mic[k * fn : (k + 1) * fn, None].copy(), fmt, cmap, t, k, "mic")
        ctx = FrameContext("drift", reference=ref.frames_until(t + 20_000_000))
        out.append(stage.process(frame, ctx).data[:, 0])
    clean = np.concatenate(out)
    lag = lag_ms * rate // 1000
    frames = len(clean) // fn
    t_frames = np.arange(frames) * 0.01
    active = np.abs(echo[: frames * fn]).reshape(frames, fn).mean(axis=1) > 1e-3
    sel = (t_frames >= measure_from_s) & active
    return erle_db(mic[: len(clean) - lag], clean[lag:], sel[: (len(clean) - lag) // fn], fn)
