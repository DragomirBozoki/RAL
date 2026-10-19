"""Echo metrics (DESIGN section 8, "Metrics").

ERLE (echo return loss enhancement) is how much quieter the echo is after echo control
than before it, in dB, measured over frames where only the robot is audible:
``10 log10(sum of mic power / sum of output power)``. Summing powers before the ratio
weights loud frames more, which matches what a listener (or a VAD) notices.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["erle_db", "frame_powers"]


def frame_powers(x: npt.NDArray[Any], frame_samples: int) -> npt.NDArray[np.float64]:
    """Mean power of each whole frame of a 1-D signal."""
    n = len(x) // frame_samples
    frames = np.asarray(x[: n * frame_samples], dtype=np.float64).reshape(n, frame_samples)
    return np.asarray(np.mean(frames**2, axis=1), dtype=np.float64)


def erle_db(
    mic: npt.NDArray[Any],
    out: npt.NDArray[Any],
    frames: npt.NDArray[np.bool_],
    frame_samples: int,
) -> float:
    """ERLE over the selected frames.

    Args:
        mic: The echo-control input, 1-D.
        out: The echo-control output, 1-D, aligned with ``mic`` (compensate any lag first).
        frames: Which frames to use, typically far-end-only frames after convergence.
        frame_samples: Samples per frame.
    """
    pm = frame_powers(mic, frame_samples)
    po = frame_powers(out, frame_samples)
    k = min(len(pm), len(po), len(frames))
    sel = frames[:k]
    if not np.any(sel):
        raise ValueError("No frames selected for ERLE.")
    return float(10 * np.log10(max(pm[:k][sel].sum(), 1e-20) / max(po[:k][sel].sum(), 1e-20)))
