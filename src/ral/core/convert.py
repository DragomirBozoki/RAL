"""Sample conversion between int16 PCM and float32.

The scale is 32768 in both directions, so int16 maps onto [-1.0, 1.0) and every int16
value survives int16 -> float32 -> int16 unchanged (k / 32768 is exact in float32).
The price is that +1.0 does not fit and clips to 32767, which is the usual asymmetry of
two's-complement PCM.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ral.core.errors import FormatError

__all__ = ["float32_to_int16", "int16_to_float32"]

_SCALE = 32768.0


def int16_to_float32(samples: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """Convert int16 PCM to float32 in [-1.0, 1.0).

    Args:
        samples: An int16 array of any shape.

    Returns:
        A new float32 array of the same shape.

    Raises:
        FormatError: If ``samples`` is not int16.
    """
    if samples.dtype != np.int16:
        raise FormatError(f"int16_to_float32 needs int16 samples, got {samples.dtype}.")
    out = samples.astype(np.float32)
    out *= np.float32(1.0 / _SCALE)  # a power of two, so the scaling is exact
    return out


def float32_to_int16(samples: npt.NDArray[Any]) -> npt.NDArray[np.int16]:
    """Convert floating-point samples to int16 PCM, rounding and clipping.

    Values are rounded to the nearest integer (ties to even), clipped to the int16
    range, and NaN becomes 0, so bad input from an upstream stage cannot turn into
    loud noise at the speaker.

    Args:
        samples: A floating-point array of any shape, nominally in [-1.0, 1.0].

    Returns:
        A new int16 array of the same shape.

    Raises:
        FormatError: If ``samples`` is not a floating-point array.
    """
    if not np.issubdtype(samples.dtype, np.floating):
        raise FormatError(f"float32_to_int16 needs floating-point samples, got {samples.dtype}.")
    scaled: npt.NDArray[np.float32] = np.multiply(samples, _SCALE, dtype=np.float32)
    np.rint(scaled, out=scaled)
    np.clip(scaled, -_SCALE, _SCALE - 1.0, out=scaled)  # also maps +-inf to the limits
    np.nan_to_num(scaled, copy=False, nan=0.0)
    return scaled.astype(np.int16)
