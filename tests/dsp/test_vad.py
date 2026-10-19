import numpy as np
import pytest

from ral.core import AudioFormat, AudioFrame, ChannelMap
from ral.core import events as ev
from ral.core.processing import FrameContext
from ral.dsp.vad import EnergyVad, VadStateMachine

MS = 1_000_000
FMT = AudioFormat(16000, 1, "float32")
CMAP = ChannelMap.from_names(["processed"])


def bursts(spans: list[tuple[float, float]], seconds: float, level_db: float = -30.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    n = int(seconds * 16000)
    x = rng.standard_normal(n) * 10 ** (-65 / 20)
    for a, b in spans:
        seg = slice(int(a * 16000), int(b * 16000))
        x[seg] += rng.standard_normal(seg.stop - seg.start) * 10 ** (level_db / 20)
    return x.astype(np.float32)


def run(vad: EnergyVad, x: np.ndarray) -> tuple[list[ev.Event], list[float]]:
    events, probs = [], []
    for k in range(len(x) // 160):
        frame = AudioFrame(x[k * 160 : (k + 1) * 160, None].copy(), FMT, CMAP, k * 10 * MS, k, "m")
        ctx = FrameContext("s")
        vad.process(frame, ctx)
        events.extend(ctx.events)
        probs.append(ctx.info["vad_prob"])
    return events, probs


def test_state_machine_start_needs_min_speech_and_end_needs_hangover() -> None:
    sm = VadStateMachine(threshold=0.5, min_speech_ms=30, hangover_ms=50)
    probs = [0, 1, 1, 0.2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    out = []
    for k, p in enumerate(probs):
        out.extend(sm.update(p, k * 10 * MS, 10 * MS, "s"))
    started, ended = out
    assert isinstance(started, ev.SpeechStarted)
    assert started.t_ns == 40 * MS  # the dip at 30 ms cancelled the first candidate
    assert isinstance(ended, ev.SpeechEnded)
    assert ended.t_ns == 80 * MS  # end of the last speech window, before the hangover
    assert ended.t_start_ns == 40 * MS


def test_hysteresis_keeps_a_candidate_through_a_shallow_dip() -> None:
    sm = VadStateMachine(threshold=0.5, min_speech_ms=30, hangover_ms=100)
    out = []
    for k, p in enumerate([1, 0.45, 1, 1]):
        out.extend(sm.update(p, k * 10 * MS, 10 * MS, "s"))
    assert [type(e) for e in out] == [ev.SpeechStarted]
    assert out[0].t_ns == 0


def test_energy_vad_finds_bursts_with_accurate_times() -> None:
    vad = EnergyVad(min_speech_ms=50, hangover_ms=300)
    events, _ = run(vad, bursts([(1.0, 2.0), (3.0, 3.5)], 5.0))
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["SpeechStarted", "SpeechEnded", "SpeechStarted", "SpeechEnded"]
    assert events[0].t_ns == pytest.approx(1000 * MS, abs=20 * MS)
    assert events[1].t_ns == pytest.approx(2000 * MS, abs=20 * MS)
    assert events[2].t_ns == pytest.approx(3000 * MS, abs=20 * MS)


def test_energy_vad_ignores_steady_noise_and_short_clicks() -> None:
    vad = EnergyVad(min_speech_ms=100)
    events, probs = run(vad, bursts([(2.0, 2.03)], 4.0))
    assert events == []
    assert max(probs[50:190]) < 0.5


def test_energy_vad_adapts_to_a_louder_noise_floor() -> None:
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(16000 * 12) * 10 ** (-40 / 20)).astype(np.float32)
    x[: 16000 * 2] *= 10 ** (-25 / 20)  # quiet start, then the fan turns on
    events, _ = run(EnergyVad(), x)
    # The floor needs a few seconds to rise 25 dB; afterwards nothing counts as speech.
    assert all(e.t_ns < 12_000 * MS for e in events)
    assert not any(isinstance(e, ev.SpeechStarted) and e.t_ns > 20_000 * MS for e in events)
    events2, _ = run(EnergyVad(), x[16000 * 2 :])
    assert events2 == []
