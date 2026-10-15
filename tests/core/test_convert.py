import numpy as np
import pytest

from ral.core import FormatError, float32_to_int16, int16_to_float32


def test_every_int16_value_round_trips_bit_exact() -> None:
    all_values = np.arange(-32768, 32768, dtype=np.int16)
    assert np.array_equal(float32_to_int16(int16_to_float32(all_values)), all_values)


def test_int16_to_float32_range_and_shape() -> None:
    x = np.array([[-32768, 0], [16384, 32767]], dtype=np.int16)
    y = int16_to_float32(x)
    assert y.dtype == np.float32
    assert y.shape == (2, 2)
    assert y[0, 0] == -1.0
    assert y[1, 0] == 0.5
    assert y[1, 1] < 1.0


def test_float32_to_int16_clips_out_of_range_and_infinities() -> None:
    x = np.array([1.0, 2.0, -1.0, -2.0, np.inf, -np.inf], dtype=np.float32)
    assert float32_to_int16(x).tolist() == [32767, 32767, -32768, -32768, 32767, -32768]


def test_float32_to_int16_maps_nan_to_zero() -> None:
    x = np.array([np.nan, 0.5], dtype=np.float32)
    assert float32_to_int16(x).tolist() == [0, 16384]


def test_float32_to_int16_rounds_to_nearest() -> None:
    step = 1.0 / 32768.0
    x = np.array([0.4 * step, 0.6 * step, -0.6 * step], dtype=np.float32)
    assert float32_to_int16(x).tolist() == [0, 1, -1]


def test_float32_to_int16_accepts_float64_and_keeps_input() -> None:
    x = np.array([[0.25], [-0.25]], dtype=np.float64)
    y = float32_to_int16(x)
    assert y.dtype == np.int16
    assert y.shape == (2, 1)
    assert y[:, 0].tolist() == [8192, -8192]
    assert x[0, 0] == 0.25  # input not modified in place


def test_wrong_input_dtypes_are_rejected() -> None:
    with pytest.raises(FormatError, match="int16"):
        int16_to_float32(np.zeros(4, dtype=np.int32))
    with pytest.raises(FormatError, match="floating-point"):
        float32_to_int16(np.zeros(4, dtype=np.int16))
