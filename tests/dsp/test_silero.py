import wave
from pathlib import Path

import numpy as np
import pytest

from ral.core import AudioFormat, AudioFrame, ChannelMap, RalError
from ral.core import events as ev
from ral.core.processing import FrameContext
from ral.dsp.resample import Resampler

pytest.importorskip("onnxruntime")

from ral.dsp.silero import SileroVad
from ral.models import model_path

DATA = Path(__file__).resolve().parents[1] / "data"
MS = 1_000_000


def _model_or_skip() -> Path:
    try:
        return model_path("silero_vad")
    except RalError:
        pytest.skip("Silero model not downloaded; run: ral models fetch silero_vad")


@pytest.fixture
def speech() -> np.ndarray:
    with wave.open(str(DATA / "speech_16k.wav"), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768


def run(vad: SileroVad, x: np.ndarray, rate: int = 16000, seqs: list[int] | None = None):  # type: ignore[no-untyped-def]
    fmt = AudioFormat(rate, 1, "float32")
    n = fmt.frame_samples
    events, probs = [], []
    for k in range(len(x) // n):
        seq = seqs[k] if seqs else k
        frame = AudioFrame(
            x[k * n : (k + 1) * n, None].copy(), fmt, ChannelMap.uniform(1), seq * 10 * MS, seq, "m"
        )
        ctx = FrameContext("s")
        vad.process(frame, ctx)
        events.extend(ctx.events)
        probs.append(ctx.info["vad_prob"])
    return events, probs


def starts(events: list[ev.Event]) -> list[float]:
    return [e.t_ns / 1e9 for e in events if isinstance(e, ev.SpeechStarted)]


def ends(events: list[ev.Event]) -> list[float]:
    return [e.t_ns / 1e9 for e in events if isinstance(e, ev.SpeechEnded)]


def test_real_speech_has_the_expected_segments(speech: np.ndarray) -> None:
    events, _ = run(SileroVad(_model_or_skip()), speech)
    # The clip: speech from about 0.1 s to 2.8 s, a pause, speech again from about 5.3 s.
    assert starts(events)[0] < 0.3
    assert any(2.5 < t < 3.2 for t in ends(events))
    assert any(5.0 < t < 5.7 for t in starts(events))
    assert not any(3.2 < t < 5.0 for t in starts(events))


def test_noise_and_tones_are_not_speech() -> None:
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(16000 * 4) * 0.01).astype(np.float32)
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(16000 * 4) / 16000)).astype(np.float32)
    for x in (noise, tone):
        events, probs = run(SileroVad(_model_or_skip()), x)
        assert starts(events) == []
        assert max(probs) < 0.5


def test_48k_input_is_resampled_with_the_same_timing(speech: np.ndarray) -> None:
    up = Resampler(16000, 48000, 1)
    x48 = np.concatenate([up.process(speech[:, None]), up.flush()])[:, 0]
    e16, _ = run(SileroVad(_model_or_skip()), speech)
    e48, _ = run(SileroVad(_model_or_skip()), x48, rate=48000)
    # A pause right at the hangover limit may merge on one path, so compare boundaries.
    assert abs(starts(e48)[0] - starts(e16)[0]) < 0.07
    assert abs(starts(e48)[-1] - starts(e16)[-1]) < 0.07
    assert abs(ends(e48)[-1] - ends(e16)[-1]) < 0.07


def test_seq_gap_resets_the_window_timeline(speech: np.ndarray) -> None:
    vad = SileroVad(_model_or_skip())
    seqs = list(range(100)) + list(range(150, 150 + len(speech) // 160 - 100))
    events, _ = run(vad, speech, seqs=seqs)
    # Audio after the gap is stamped 0.5 s later, so the second segment moves with it.
    assert any(5.5 < t < 6.2 for t in starts(events))


def test_missing_model_gives_a_fetch_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAL_MODELS_DIR", str(tmp_path))
    with pytest.raises(RalError, match="ral models fetch silero_vad"):
        SileroVad()
