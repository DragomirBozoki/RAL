import numpy as np
import pytest

from ral.core import AudioFormat, FormatError
from ral.dsp.drift import DriftCorrector, DriftEstimator, drift_ppm
from ral.playback.reference import ReferenceBuffer

T0 = 5_000_000_000_000
FS = 16000


def feed(
    est: DriftEstimator, rate_hz: float, seconds: float, jitter_ms: float, seed: int = 0
) -> None:
    """Blocks of 10 ms whose observed times jitter, from a clock running at ``rate_hz``."""
    rng = np.random.default_rng(seed)
    block = est.nominal_rate // 100
    for k in range(int(seconds * 100)):
        index = k * block
        t = T0 + int(index / rate_hz * 1e9 + rng.normal(0, jitter_ms * 1e6))
        est.observe(index, t)


def test_estimator_measures_ppm_through_millisecond_jitter() -> None:
    est = DriftEstimator(48000)
    feed(est, 48000 * (1 + 150e-6), 60, jitter_ms=2.0)
    rel = est.relative()
    assert rel is not None
    assert (rel - 1) * 1e6 == pytest.approx(150, abs=10)


def test_estimator_waits_for_enough_history() -> None:
    est = DriftEstimator(16000, min_span_s=10)
    feed(est, 16000, 5, jitter_ms=0.5)
    assert est.rate_hz() is None


def test_drift_between_two_clocks() -> None:
    ref, cap = DriftEstimator(48000), DriftEstimator(16000)
    feed(ref, 48000 * (1 + 80e-6), 60, 1.0, seed=1)
    feed(cap, 16000 * (1 - 40e-6), 60, 1.0, seed=2)
    ppm = drift_ppm(ref, cap)
    assert ppm is not None
    assert ppm == pytest.approx(120, abs=10)


def run(c: DriftCorrector, x: np.ndarray, block: int = 160) -> np.ndarray:
    return np.concatenate([c.process(x[i : i + block]) for i in range(0, len(x), block)])


def tone(seconds: float = 2.0, f: float = 1000.0) -> np.ndarray:
    return (0.5 * np.sin(2 * np.pi * f * np.arange(int(FS * seconds)) / FS)).astype(np.float32)


def test_ratio_one_is_the_identity() -> None:
    x = tone()
    y = run(DriftCorrector(), x)
    np.testing.assert_allclose(y, x[: len(y)], atol=1e-6)
    assert len(x) - len(y) == 16  # the interpolator's lookahead


def test_ratio_changes_pitch_and_length_precisely() -> None:
    x = tone(4.0)
    c = DriftCorrector()
    c.ratio = 0.999
    y = run(c, x, block=97)
    assert len(y) == pytest.approx(len(x) * 0.999, abs=20)
    n = np.arange(1000, 1000 + FS)
    ideal = 0.5 * np.sin(2 * np.pi * 1000 * (n / 0.999) / FS)
    err = y[1000 : 1000 + FS] - ideal
    assert 20 * np.log10(np.sqrt(np.mean(err**2)) / 0.3536) < -80
    np.testing.assert_allclose(np.diff(c.positions), 1 / 0.999, rtol=1e-9)


def test_unreasonable_ratios_are_refused() -> None:
    with pytest.raises(FormatError, match="ppm"):
        DriftCorrector(max_ppm=2000).ratio = 1.01


def test_reference_buffer_moves_a_fast_sink_onto_the_mic_clock() -> None:
    # The sink's clock runs 300 ppm fast: its 160-sample blocks last a little less than 10 ms.
    eps = 300e-6
    fmt = AudioFormat(FS, 1, "float32")
    ref = ReferenceBuffer(fmt, drift=True, capacity_s=5.0)
    mics = DriftEstimator(FS)
    ref.capture_estimator = mics
    out = []
    for k in range(6000):  # 60 s
        t_dac = T0 + int(k * 10_000_000 / (1 + eps))
        ref.push(np.full((160, 1), 0.1, np.float32), t_dac)
        mics.observe(k * 160, T0 + k * 10_000_000)
        out.extend(ref.drain())
    assert ref.drift_ppm == pytest.approx(300, abs=15)
    late = [f.t_ns for f in out if f.t_ns > T0 + 30_000_000_000]
    spacing = np.diff(late).mean()
    # Corrected frames advance by 10 ms of mic time, not 10 / (1 + eps) ms.
    assert spacing == pytest.approx(10_000_000, rel=3e-5)


def test_without_correction_the_reference_keeps_the_sink_clock() -> None:
    eps = 300e-6
    ref = ReferenceBuffer(AudioFormat(FS, 1, "float32"), capacity_s=5.0)
    out = []
    for k in range(3000):
        ref.push(np.zeros((160, 1), np.float32), T0 + int(k * 10_000_000 / (1 + eps)))
        out.extend(ref.drain())
    # Once the sink's rate is measured (10 s), frame times follow the sink clock exactly.
    late = [f.t_ns for f in out if f.t_ns > T0 + 15_000_000_000]
    assert np.diff(late).mean() == pytest.approx(10_000_000 / (1 + eps), rel=1e-6)


def test_compensation_keeps_aec_working_under_drift() -> None:
    pytest.importorskip("livekit.rtc")
    from ral.bench.drift import drift_erle
    from ral.dsp.echo import ApmProcessor

    kw = {"eps_ppm": 800.0, "seconds": 40, "measure_from_s": 25.0}
    without = drift_erle(ApmProcessor(), compensate=False, **kw)  # type: ignore[arg-type]
    with_ = drift_erle(ApmProcessor(), compensate=True, **kw)  # type: ignore[arg-type]
    assert with_ > 35.0
    assert with_ > without + 5.0
