"""The one clock every RAL timestamp comes from.

``now_ns()`` is a monotonic clock in integer nanoseconds. On Linux and macOS it is
``time.monotonic_ns()``. On Windows before CPython 3.13, ``time.monotonic()`` is backed
by ``GetTickCount64`` and only advances in 15.6 ms steps, coarser than one 10 ms frame
(measured in ``spikes/clock_resolution.py``), so there RAL uses ``time.perf_counter_ns()``,
which is ``QueryPerformanceCounter`` (100 ns, monotonic, system-wide). CPython 3.13 made
the same switch for ``time.monotonic()`` on Windows.

Reference: Microsoft, "Acquiring high-resolution time stamps",
https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquiring-high-resolution-time-stamps

Never mix ``now_ns()`` with ``time.monotonic_ns()`` in the audio path: on Windows they
have different epochs.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import Literal

__all__ = ["CLOCK_NAME", "CLOCK_RESOLUTION_NS", "now_ns"]

_FINE_ENOUGH_S = 1e-6

ClockName = Literal["monotonic", "perf_counter"]


def _pick_clock() -> tuple[Callable[[], int], ClockName]:
    if sys.platform == "win32" and time.get_clock_info("monotonic").resolution > _FINE_ENOUGH_S:
        return time.perf_counter_ns, "perf_counter"
    return time.monotonic_ns, "monotonic"


now_ns: Callable[[], int]
"""Current time of the RAL clock in integer nanoseconds."""
CLOCK_NAME: ClockName
"""Which stdlib clock backs ``now_ns``."""
now_ns, CLOCK_NAME = _pick_clock()

CLOCK_RESOLUTION_NS: int = max(1, round(time.get_clock_info(CLOCK_NAME).resolution * 1e9))
"""Resolution the platform reports for the chosen clock."""
