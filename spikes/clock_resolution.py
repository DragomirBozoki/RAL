"""Spike: resolution of time.monotonic_ns() and how precisely asyncio.sleep wakes up.

Question: CLAUDE.md rule 5 stamps audio with time.monotonic_ns(). Is that clock fine
enough for 10 ms frames and millisecond-level echo alignment on every platform?

Run: uv run python spikes/clock_resolution.py

Result, 2026-10-15, Windows 11 x86_64, CPython 3.10.22 and 3.12.15:
    monotonic:    GetTickCount64(), resolution 15.625 ms; observed steps 15 and 16 ms.
    perf_counter: QueryPerformanceCounter(), resolution 100 ns.
    asyncio.sleep(0.010) measured with perf_counter: late by +7 ms on average, between
    +1.4 and +22 ms (3.12) and +4.9 to +13 ms (3.10); never early.
    Measured with monotonic_ns itself, a 10 ms wait can look like 0 or 16 ms, so a
    realtime FileSource test saw a frame "6 ms early" (the observed failure).
Conclusion: on Windows before Python 3.13, monotonic_ns() is too coarse for RAL. CPython
3.13 moved time.monotonic() on Windows to QueryPerformanceCounter. On Linux and macOS
both clocks are CLOCK_MONOTONIC (or mach_absolute_time) with ns resolution.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time


def clock_steps(clock: str, duration_s: float = 0.3) -> list[float]:
    """Distinct tick sizes in ms seen while polling ``clock`` for ``duration_s``."""
    read = getattr(time, f"{clock}_ns")
    prev = read()
    end = prev + int(duration_s * 1e9)
    steps: set[float] = set()
    while (now := read()) < end:
        if now != prev:
            steps.add(round((now - prev) / 1e6, 3))
            prev = now
    return sorted(steps)[:5]


async def sleep_errors(n: int = 50, delay_s: float = 0.010) -> list[float]:
    """Actual minus requested sleep, in ms, measured with perf_counter."""
    errors = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        await asyncio.sleep(delay_s)
        errors.append((time.perf_counter_ns() - t0) / 1e6 - delay_s * 1e3)
    return errors


def main() -> None:
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    for clock in ("monotonic", "perf_counter"):
        info = time.get_clock_info(clock)
        steps = clock_steps(clock)
        print(f"{clock:13} {info.implementation:28} res={info.resolution}  steps_ms={steps}")
    errors = asyncio.run(sleep_errors())
    print(
        f"asyncio.sleep(10 ms) error: mean {statistics.mean(errors):+.2f} ms, "
        f"min {min(errors):+.2f} ms, max {max(errors):+.2f} ms"
    )


if __name__ == "__main__":
    main()
