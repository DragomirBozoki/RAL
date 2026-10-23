"""Spike: does the GIL starve RAL's audio threads when the same process does other work?

DESIGN section 10 (M3): "engine in its own process if profiling shows the GIL starving
the DSP thread". RAL's audio path has two Python threads: the device callback
(sounddevice calls Python every block, which needs the GIL) and the DSP worker. Python
threads share one interpreter lock. A thread that gives it up to wait (a sleep, a queue,
I/O) must ask for it back, and a thread running pure Python holds it for the whole
switch interval (``sys.getswitchinterval()``, 5 ms) before handing it over: the "convoy
effect" (D. Beazley, "Understanding the Python GIL", PyCon 2010; CPython issue bpo-7946).
Native code that releases the lock (numpy, AEC3 in livekit, onnxruntime) does not
compete.

Setup, no asyncio and no file I/O, so only the GIL is measured: a device thread wakes
every 10 ms on its deadline (like a sound card callback) and queues a frame; the DSP
thread runs the real capture chain (``software_ref``: delay tracker, AEC3, noise
suppression, AGC, energy VAD) with a reference frame. Per frame: how late the device
thread woke, and how late the DSP thread finished it (now minus the end of the frame).
Busy threads run pure-Python loops (a CPU-heavy agent in the same process). 10 s per
scenario; the last row repeats 3 busy threads with a 1 ms switch interval.

Run: uv run python spikes/dsp_gil_load.py

Result, 2026-10-23, Windows 11 x86_64 laptop, CPython 3.12.15, livekit 1.1.20 ("in time":
frames the DSP thread finished before the device thread stopped):

    scenario                  device wake p99/max   DSP done late p50/p99/max  >20 ms  in time
    idle                          0.9     3.7 ms      1.2    4.3   15.9 ms    0.0 %  100.0 %
    1 busy thread                30.4    40.5 ms   8460.7 8875.7 8885.6 ms  100.0 %   11.3 %
    3 busy threads              257.7   337.6 ms        -      -      - ms  100.0 %    4.3 %
    3 busy, switch 1 ms         178.7   238.9 ms        -      -      - ms  100.0 %    4.6 %

Meaning: alone, the chain finishes a frame 1.2 ms after it ends (4.3 ms at p99). One
thread of pure Python in the same process makes the device thread wake 30 ms late (a
sound card callback that late overflows small buffers) and the DSP thread falls behind
for good: it releases the GIL around every numpy and AEC3 call and waits up to a switch
interval to get it back each time. A shorter switch interval does not save it.

Decision (DESIGN section 9, "Process model"): RAL runs in a process of its own (``ral
run``, the ``ral_ros`` node), and agents, TTS and robot code connect through the bridges,
which already exist for that. Embedding ``Engine`` in an application process is for light
control code only; ``Engine.dsp_late_ms`` and the ``dsp_late_p99_ms`` health metric show
when that rule is broken. A child process for the DSP worker inside the ``Engine`` API
would need shared-memory rings for audio both ways and is not worth it while the bridges
cover the split. TODO(M4): revisit if users need the in-process API under load.

The first version of this spike ran a whole engine with realtime file devices and found
that the file backend's pacing spun the event loop on Windows (asyncio fires timers up
to 15.6 ms early there), which itself starved the DSP thread; ``_sleep_until`` now waits
at least one clock step.
"""

from __future__ import annotations

import queue
import sys
import threading
import time

import numpy as np

from ral.core.clock import now_ns
from ral.core.processing import FrameContext
from ral.core.types import AudioFormat, AudioFrame, ChannelMap
from ral.dsp.pipeline import build_capture_chain
from ral.dsp.vad import EnergyVad

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
FMT = AudioFormat(16000, 1, "float32")
FRAME_NS = FMT.frame_ns


def busy(stop: threading.Event) -> None:
    x = 0
    while not stop.is_set():
        for i in range(2000):
            x += i * i % 7


def scenario(busy_threads: int, switch_s: float = 0.005) -> dict[str, float]:
    sys.setswitchinterval(switch_s)
    stages = build_capture_chain(
        "software_ref",
        ["aec", "noise_suppression", "agc", "vad"],
        mic=0,
        lead_ms=20,
        vad=EnergyVad(),
        delay_tracking=True,
    )
    rng = np.random.default_rng(0)
    far = (rng.standard_normal(16000 * 2) * 0.1).astype(np.float32)
    frames: queue.Queue[tuple[int, int] | None] = queue.Queue()
    wake_ms: list[float] = []
    done_ms: list[float] = []
    stop = threading.Event()
    device_done = threading.Event()  # what the DSP thread has not done by then stays undone
    t0 = now_ns() + 50_000_000
    n = int(SECONDS * 100)

    def device() -> None:
        for k in range(n):
            deadline = t0 + (k + 1) * FRAME_NS  # the block is complete at its end
            delay = deadline - now_ns()
            if delay > 0:
                time.sleep(delay / 1e9)
            wake_ms.append((now_ns() - deadline) / 1e6)
            frames.put((k, deadline - FRAME_NS))
        frames.put(None)

    def dsp() -> None:
        cmap, rmap = ChannelMap.uniform(1), ChannelMap.from_names(["reference"])
        while (item := frames.get()) is not None and not device_done.is_set():
            k, t = item
            i = (k * 160) % (len(far) - 160)
            ref = AudioFrame(far[i : i + 160, None], FMT, rmap, t - 5_000_000, k, "dac")
            mic = AudioFrame(far[i : i + 160, None] * 0.3, FMT, cmap, t, k, "mic")
            ctx = FrameContext("s", reference=[ref])
            for stage in stages:
                mic = stage.process(mic, ctx)
            done_ms.append((now_ns() - t - FRAME_NS) / 1e6)

    workers = [threading.Thread(target=busy, args=(stop,)) for _ in range(busy_threads)]
    for w in workers:
        w.start()
    threads = [threading.Thread(target=device), threading.Thread(target=dsp)]
    for th in threads:
        th.start()
    threads[0].join()
    device_done.set()
    threads[1].join()
    stop.set()
    for w in workers:
        w.join()
    sys.setswitchinterval(0.005)
    wake, done = np.array(wake_ms[100:]), np.array(done_ms[100:])  # skip the first second
    if len(done) == 0:
        done = np.array([np.inf])
    return {
        "processed": len(done_ms) / n * 100,
        "wake_p99": float(np.percentile(wake, 99)),
        "wake_max": float(wake.max()),
        "done_p50": float(np.percentile(done, 50)),
        "done_p99": float(np.percentile(done, 99)),
        "done_max": float(done.max()),
        "over_20": float(np.mean(done > 20.0) * 100),
    }


def main() -> None:
    print(f"CPython {sys.version.split()[0]} on {sys.platform}, {SECONDS:.0f} s per scenario")
    print(
        "scenario                  device wake p99/max   DSP done late p50/p99/max  >20 ms  in time"
    )
    for name, threads, switch in (
        ("idle", 0, 0.005),
        ("1 busy thread", 1, 0.005),
        ("3 busy threads", 3, 0.005),
        ("3 busy, switch 1 ms", 3, 0.001),
    ):
        r = scenario(threads, switch)
        print(
            f"{name:<25} {r['wake_p99']:7.1f} {r['wake_max']:7.1f} ms   "
            f"{r['done_p50']:6.1f} {r['done_p99']:6.1f} {r['done_max']:6.1f} ms  "
            f"{r['over_20']:5.1f} %  {r['processed']:5.1f} %",
            flush=True,
        )


if __name__ == "__main__":
    main()
