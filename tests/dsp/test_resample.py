import numpy as np
import pytest

from ral.core import FormatError
from ral.dsp.resample import Resampler

RATES = [8000, 16000, 22000, 24000, 32000, 44100, 48000, 96000]


def tone(rate: int, freq: float, seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(round(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)[:, None]


def run_blocks(r: Resampler, x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Feed x in random block sizes, then flush."""
    rng = np.random.default_rng(seed)
    out, i = [], 0
    while i < len(x):
        n = int(rng.integers(1, 3 * r.in_rate // 100))
        out.append(r.process(x[i : i + n]))
        i += n
    out.append(r.flush())
    return np.concatenate(out)


def rms_db(x: np.ndarray, ref: float) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) / ref))


@pytest.mark.parametrize("rate", RATES)
def test_tone_is_accurate_and_time_aligned(rate: int) -> None:
    # A 1 kHz tone must come out at the same phase as an ideal tone sampled at 16 kHz:
    # that checks gain, frequency and that output k sits at input time k * M / L.
    y = run_blocks(Resampler(rate, 16000, 1), tone(rate, 1000, 1.0))[:, 0]
    k = np.arange(len(y))
    ideal = 0.5 * np.sin(2 * np.pi * 1000 * k / 16000)
    inner = slice(800, len(y) - 800)  # ignore the zero-padded edges
    assert rms_db(y[inner] - ideal[inner], 0.5 / np.sqrt(2)) < -70


@pytest.mark.parametrize("rate", [24000, 32000, 44100, 48000, 96000])
def test_tones_above_the_output_nyquist_are_suppressed(rate: int) -> None:
    y = run_blocks(Resampler(rate, 16000, 1), tone(rate, 9000, 1.0))
    assert rms_db(y[800:-800], 0.5 / np.sqrt(2)) < -70


@pytest.mark.parametrize("rate", RATES)
def test_output_length_matches_the_ratio(rate: int) -> None:
    n_in = rate // 100 * 37
    y = run_blocks(Resampler(rate, 16000, 2), np.zeros((n_in, 2), np.float32))
    # Every output at or before the last input sample's time is emitted.
    assert len(y) == (n_in - 1) * 16000 // rate + 1


def test_block_size_does_not_change_the_output() -> None:
    x = np.random.default_rng(1).standard_normal((4410, 2)).astype(np.float32) * 0.1
    a = run_blocks(Resampler(44100, 16000, 2), x, seed=1)
    b = run_blocks(Resampler(44100, 16000, 2), x, seed=2)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_channels_are_independent() -> None:
    x = np.zeros((4800, 2), np.float32)
    x[:, 0] = tone(48000, 500, 0.1)[:, 0]
    y = run_blocks(Resampler(48000, 16000, 2), x)
    assert np.max(np.abs(y[:, 1])) == 0.0
    assert np.max(np.abs(y[:, 0])) > 0.4


def test_equal_rates_pass_samples_through_unchanged() -> None:
    x = np.random.default_rng(2).standard_normal((1000, 3)).astype(np.float32)
    r = Resampler(16000, 16000, 3)
    assert r.passthrough
    np.testing.assert_array_equal(r.process(x), x)


def test_reset_starts_a_new_stream() -> None:
    r = Resampler(48000, 16000, 1)
    x = tone(48000, 1000, 0.05)
    first = np.concatenate([r.process(x), r.flush()])
    r.process(np.ones((480, 1), np.float32))
    r.reset()
    second = np.concatenate([r.process(x), r.flush()])
    np.testing.assert_allclose(first, second, atol=1e-7)


def test_bad_arguments_are_rejected() -> None:
    with pytest.raises(FormatError):
        Resampler(0, 16000, 1)
    with pytest.raises(FormatError):
        Resampler(48000, 16000, 0)
    with pytest.raises(FormatError):
        Resampler(48000, 16000, 1).process(np.zeros((10, 2), np.float32))
