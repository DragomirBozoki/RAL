import sys
import time

import pytest

from ral.core import clock


def test_now_ns_is_monotonic_integer_nanoseconds() -> None:
    readings = [clock.now_ns() for _ in range(10_000)]
    assert all(isinstance(t, int) for t in readings[:10])
    assert readings == sorted(readings)


def test_clock_resolves_well_below_one_frame() -> None:
    """The clock must tick in steps far finer than 10 ms; 15.6 ms steps would fail here."""
    start = clock.now_ns()
    steps = []
    prev = start
    while len(steps) < 50 and clock.now_ns() - start < 200_000_000:
        now = clock.now_ns()
        if now != prev:
            steps.append(now - prev)
            prev = now
    assert len(steps) == 50
    assert max(steps) < 1_000_000  # 1 ms, generous for a busy CI machine
    assert clock.CLOCK_RESOLUTION_NS <= 1_000


def test_now_ns_tracks_real_time() -> None:
    t0, p0 = clock.now_ns(), time.perf_counter_ns()
    time.sleep(0.05)
    elapsed, reference = clock.now_ns() - t0, time.perf_counter_ns() - p0
    assert abs(elapsed - reference) < 2_000_000


@pytest.mark.skipif(sys.platform == "win32", reason="Windows may use perf_counter")
def test_non_windows_uses_monotonic() -> None:
    assert clock.CLOCK_NAME == "monotonic"
    assert clock.now_ns is time.monotonic_ns


def test_coarse_windows_monotonic_falls_back_to_perf_counter() -> None:
    coarse = time.get_clock_info("monotonic").resolution > 1e-6
    if sys.platform == "win32" and coarse:
        assert clock.CLOCK_NAME == "perf_counter"
        assert clock.now_ns is time.perf_counter_ns
    else:
        assert clock.CLOCK_NAME == "monotonic"
