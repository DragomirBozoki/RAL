"""Where ``ral probe`` gets its recordings: real devices through the engine, or a simulation.

On devices, the sweep is played through the engine's playback service like any other
sound, so it reaches the reference buffer through the sink's render hook (CLAUDE.md,
reference invariant) and ``PlaybackStarted`` gives its DAC time. The capture side
subscribes to the engine's raw (normalized) frames, which carry ADC times; the recording
is cut so that sample 0 is the sweep's DAC time, which makes the impulse response peak
the playback-to-capture delay that ``software_ref`` cares about.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ral.bench.synthetic import fft_convolve
from ral.core.events import PlaybackStarted
from ral.core.types import ChannelRole

__all__ = ["EngineProbeIO", "ProbeIO", "SimulatedProbeIO"]

Signal = npt.NDArray[np.float64]


class ProbeIO(Protocol):
    """Audio access for the probe."""

    rate: int
    channels: int

    async def record(self, seconds: float) -> Signal:
        """``(samples, channels)`` of whatever the inputs hear now."""
        ...

    async def play_and_record(self, signal: Signal, tail_s: float) -> Signal:
        """Play ``signal`` and return the inputs from its DAC start, ``tail_s`` past its end."""
        ...


class EngineProbeIO:
    """Probe IO on a running :class:`~ral.engine.engine.Engine` (echo control off).

    Args:
        engine: Started engine whose capture channels are all mapped as ``mic``.
    """

    def __init__(self, engine: object) -> None:
        from ral.engine.engine import Engine

        assert isinstance(engine, Engine)
        self.engine = engine
        self.rate = engine.engine_fmt.sample_rate
        assert engine.graph is not None
        self.channels = engine.graph.engine_fmt.channels

    async def _collect(self, until_ns: int, sub: object) -> list[tuple[int, Signal]]:
        from ral.engine.engine import ResultSubscription

        assert isinstance(sub, ResultSubscription)
        blocks = []
        async for result in sub:
            blocks.append((result.raw.t_ns, result.raw.data.astype(np.float64)))
            if result.raw.t_ns >= until_ns:
                break
        return blocks

    async def record(self, seconds: float) -> Signal:
        from ral.core.clock import now_ns

        sub = self.engine.subscribe(maxsize=int(seconds * 100) + 200)
        try:
            blocks = await self._collect(now_ns() + int(seconds * 1e9), sub)
        finally:
            sub.close()
        return np.concatenate([b for _, b in blocks])

    async def play_and_record(self, signal: Signal, tail_s: float) -> Signal:
        from ral.playback.service import audio_frames

        playback = self.engine.playback
        assert playback is not None
        started = self.engine.bus.subscribe(PlaybackStarted)
        seconds = len(signal) / self.rate + tail_s
        sub = self.engine.subscribe(maxsize=int(seconds * 100) + 400)
        try:
            playback.play(audio_frames(signal.astype(np.float32)), tag="probe")
            event = await asyncio.wait_for(started.get(), 5.0)
            assert isinstance(event, PlaybackStarted)
            t_dac = event.t_ns
            blocks = await self._collect(t_dac + int(seconds * 1e9), sub)
        finally:
            sub.close()
            started.close()
        t0 = blocks[0][0]
        data = np.concatenate([b for _, b in blocks])
        offset = round((t_dac - t0) * self.rate / 1e9)
        if offset < 0:
            # Capture started after the DAC time (should not happen): pad the front.
            data = np.concatenate([np.zeros((-offset, data.shape[1])), data])
            offset = 0
        return np.asarray(data[offset : offset + int(seconds * self.rate)], dtype=np.float64)


class SimulatedProbeIO:
    """A robot made of impulse responses, for tests and ``ral probe --simulate``.

    Args:
        rate: Sample rate.
        roles: What each channel really is: ``mic``, ``reference``, ``processed`` or ``dead``.
        mic_irs: One impulse response per ``mic`` (and ``processed``) channel, in order.
        noise_db: Noise floor of live channels.
        residual_db: How much a ``processed`` channel keeps of its mic's echo.
        drift_ppm: Capture runs this much slower than playback: delay grows over time.
    """

    def __init__(
        self,
        rate: int,
        roles: Sequence[str],
        mic_irs: Sequence[Signal],
        *,
        noise_db: float = -70.0,
        residual_db: float = -45.0,
        drift_ppm: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.rate = rate
        self.roles = list(roles)
        self.channels = len(roles)
        self.mic_irs = list(mic_irs)
        self.noise = 10 ** (noise_db / 20)
        self.residual = 10 ** (residual_db / 20)
        self.drift_ppm = drift_ppm
        self._rng = np.random.default_rng(seed)
        self._clock_s = 0.0

    async def record(self, seconds: float) -> Signal:
        n = int(seconds * self.rate)
        self._clock_s += seconds
        return self._mix(np.zeros(n), n)

    async def play_and_record(self, signal: Signal, tail_s: float) -> Signal:
        n = len(signal) + int(tail_s * self.rate)
        # Drift: a fraction of a sample per second, rounded to whole samples.
        shift = round(self._clock_s * self.drift_ppm * 1e-6 * self.rate)
        self._clock_s += n / self.rate
        x = np.concatenate([np.zeros(shift), signal, np.zeros(n)])[:n]
        return self._mix(x, n)

    def _mix(self, x: Signal, n: int) -> Signal:
        cols = []
        irs = iter(self.mic_irs)
        for role in self.roles:
            noise = self._rng.standard_normal(n) * self.noise
            if role == "mic":
                cols.append(fft_convolve(x, next(irs)) + noise)
            elif role == "processed":
                cols.append(fft_convolve(x, next(irs)) * self.residual + noise)
            elif role == ChannelRole.REFERENCE.value:
                cols.append(np.concatenate([np.zeros(8), x])[:n])  # electrical, 0.5 ms
            else:
                cols.append(np.zeros(n))
        return np.column_stack(cols)
