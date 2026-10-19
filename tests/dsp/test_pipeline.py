import pytest

from ral.core import FormatError
from ral.dsp.echo import DuplexGate, EchoOff
from ral.dsp.pipeline import build_capture_chain
from ral.dsp.vad import EnergyVad


def test_software_ref_merges_apm_features_into_one_stage() -> None:
    pytest.importorskip("livekit.rtc")
    from ral.dsp.echo import ApmProcessor

    chain = build_capture_chain("software_ref", ["aec", "noise_suppression", "agc", "vad"])
    assert [type(s) for s in chain] == [ApmProcessor, EnergyVad]
    apm = chain[0]
    assert isinstance(apm, ApmProcessor)
    assert apm.echo_cancellation
    assert apm.lag_ms == 14


def test_gate_runs_apm_features_after_the_gate() -> None:
    pytest.importorskip("livekit.rtc")
    from ral.dsp.echo import ApmProcessor

    chain = build_capture_chain("gate", ["aec", "noise_suppression"])
    assert [type(s) for s in chain] == [DuplexGate, ApmProcessor]
    assert not chain[1].echo_cancellation  # type: ignore[attr-defined]


def test_off_without_apm_features_needs_no_livekit() -> None:
    chain = build_capture_chain("off", ["aec", "vad"])
    assert [type(s) for s in chain] == [EchoOff, EnergyVad]


def test_custom_vad_is_used() -> None:
    vad = EnergyVad(hangover_ms=500)
    assert build_capture_chain("off", ["vad"], vad=vad)[-1] is vad


@pytest.mark.parametrize(
    ("mode", "pipeline"), [("hardware", ["aec"]), ("off", ["aec", "beamformer"])]
)
def test_unknown_mode_or_stage(mode: str, pipeline: list[str]) -> None:
    with pytest.raises(FormatError):
        build_capture_chain(mode, pipeline)  # type: ignore[arg-type]


def test_noise_suppression_lowers_stationary_noise() -> None:
    pytest.importorskip("livekit.rtc")
    import numpy as np

    from ral.core import AudioFormat, AudioFrame, ChannelMap
    from ral.core.processing import FrameContext

    chain = build_capture_chain("off", ["aec", "noise_suppression", "high_pass"])
    fmt = AudioFormat(16000, 1, "float32")
    x = (np.random.default_rng(0).standard_normal(16000 * 3) * 0.01).astype(np.float32)
    out = []
    for k in range(len(x) // 160):
        frame = AudioFrame(
            x[k * 160 : (k + 1) * 160, None].copy(),
            fmt,
            ChannelMap.uniform(1),
            k * 10_000_000,
            k,
            "m",
        )
        ctx = FrameContext("s")
        for stage in chain:
            frame = stage.process(frame, ctx)
        out.append(frame.data[:, 0])
    y = np.concatenate(out)[16000:]
    drop = 10 * np.log10(np.mean(x[16000:] ** 2) / np.mean(y**2))
    assert drop > 10.0
