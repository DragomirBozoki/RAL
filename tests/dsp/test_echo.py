import numpy as np
import pytest

from ral.bench.synthetic import EchoScene, echo_scene
from ral.core import AudioFormat, AudioFrame, ChannelMap, FormatError
from ral.core.processing import FrameContext, Processor
from ral.dsp.echo import APM_LAG_MS, DuplexGate, EchoOff, select_mic
from ral.metrics.echo import erle_db
from ral.playback.reference import ReferenceBuffer

T0 = 10_000_000_000
FRAME_NS = 10_000_000
ENGINE = AudioFormat(16000, 1, "float32")


def mic_frames(x: np.ndarray, roles: list[str] | None = None) -> list[AudioFrame]:
    roles = roles or ["mic"]
    fmt = AudioFormat(16000, len(roles), "float32")
    cmap = ChannelMap.from_names(roles)
    n = len(x) // 160
    data = np.repeat(x[: n * 160, None], len(roles), axis=1).astype(np.float32)
    return [
        AudioFrame(data[k * 160 : (k + 1) * 160], fmt, cmap, T0 + k * FRAME_NS, k, "mic")
        for k in range(n)
    ]


def run_software_ref(
    scene: EchoScene, proc: Processor, *, dac_error_ms: int = 0, lead_ms: int = 20
) -> tuple[np.ndarray, list[AudioFrame]]:
    """Play the far end through a reference buffer and process the mic like the engine.

    The far end's frame k reaches the DAC at T0 + k * 10 ms; its echo is inside the mic
    signal. `dac_error_ms` makes the DAC estimates late, so the reference arrives late.
    """
    ref = ReferenceBuffer(ENGINE)
    far = scene.far.astype(np.float32)
    rendered = 0
    outs = []
    for k, frame in enumerate(mic_frames(scene.mic)):
        # The sink renders 100 ms ahead of the DAC, like a real output buffer.
        while rendered < min(k + 10, len(far) // 160):
            block = far[rendered * 160 : (rendered + 1) * 160, None]
            ref.push(block, T0 + rendered * FRAME_NS + dac_error_ms * 1_000_000)
            rendered += 1
        ctx = FrameContext("s", reference=ref.frames_until(frame.t_ns + lead_ms * 1_000_000))
        outs.append(proc.process(frame, ctx))
    return np.concatenate([f.data[:, 0] for f in outs]), outs


def aligned_erle(scene: EchoScene, out: np.ndarray, lag_ms: int) -> float:
    lag = lag_ms * 16
    mic = scene.mic[: len(out) - lag]
    return erle_db(mic, out[lag:], scene.far_only_frames(after_s=3.0), 160)


@pytest.fixture(scope="module")
def apm_module():  # type: ignore[no-untyped-def]
    return pytest.importorskip("livekit.rtc")


def test_echo_off_passes_the_mic_through() -> None:
    scene = echo_scene(seconds=4)
    out, frames = run_software_ref(scene, EchoOff())
    assert abs(aligned_erle(scene, out, 0)) < 0.1
    assert frames[0].channel_map.roles[0].value == "processed"
    assert frames[5].t_ns == T0 + 5 * FRAME_NS


@pytest.mark.parametrize("delay_ms", [5, 20, 120])
def test_software_ref_cancels_echo(apm_module: object, delay_ms: int) -> None:
    from ral.dsp.echo import ApmProcessor

    scene = echo_scene(seconds=8, delay_ms=delay_ms)
    proc = ApmProcessor(lead_ms=20)
    out, frames = run_software_ref(scene, proc)
    assert proc.lag_ms == 8
    assert frames[10].t_ns == T0 + 10 * FRAME_NS - 8_000_000
    assert aligned_erle(scene, out, proc.lag_ms) > 25.0
    assert proc.reference_frames >= len(frames)


def test_late_reference_fails(apm_module: object) -> None:
    # The DAC estimate is 50 ms late, so even with a 20 ms lead the reference reaches
    # the canceller 30 ms after its echo: echo control must break down (DESIGN 5).
    from ral.dsp.echo import ApmProcessor

    scene = echo_scene(seconds=8, delay_ms=5)
    good, _ = run_software_ref(scene, ApmProcessor())
    late, _ = run_software_ref(scene, ApmProcessor(), dac_error_ms=50)
    assert aligned_erle(scene, good, 8) > 25.0
    assert aligned_erle(scene, late, 8) < 15.0


def test_near_end_survives_software_ref(apm_module: object) -> None:
    from ral.dsp.echo import ApmProcessor

    scene = echo_scene(seconds=8, far_spans=[(0, 5)], near_spans=[(5.5, 8)])
    out, _ = run_software_ref(scene, ApmProcessor())
    lag = 8 * 16
    near = scene.near[: len(out) - lag]
    seg = slice(int(5.8 * 16000), int(7.8 * 16000))
    level = 10 * np.log10(np.mean(out[lag:][seg] ** 2) / np.mean(near[seg] ** 2))
    assert level > -3.0  # the user's voice is not suppressed when the robot is silent


@pytest.mark.parametrize(
    ("aec", "ns"), [(False, True), (True, False), (True, True), (False, False)]
)
def test_apm_lag_matches_the_measured_table(apm_module: object, aec: bool, ns: bool) -> None:
    # Re-measures spikes/apm_lag.py: an upgrade of livekit that changes the lag fails here.
    from ral.dsp.echo import ApmProcessor

    rng = np.random.default_rng(0)
    x = (rng.standard_normal(16000 * 3) * 0.1).astype(np.float32)
    x *= (np.sin(2 * np.pi * 3 * np.arange(len(x)) / 16000) > 0).astype(np.float32)
    proc = ApmProcessor(echo_cancellation=aec, noise_suppression=ns)
    out = np.concatenate([proc.process(f, FrameContext("s")).data[:, 0] for f in mic_frames(x)])
    a, b = x[16000:], out[16000:]
    lag = max(range(400), key=lambda d: float(np.dot(a[: len(a) - d], b[d:])))
    expected = APM_LAG_MS["echo_cancellation"] * aec + APM_LAG_MS["noise_suppression"] * ns
    assert lag == expected * 16
    assert proc.lag_ms == expected


def test_gate_closes_while_speaking_plus_tail_and_learns_echo() -> None:
    gate = DuplexGate(attenuation_db=-30, tail_ms=100, ramp_ms=10)
    x = np.full(16000, 0.1, np.float32)
    gains, echo = [], []
    for k, frame in enumerate(mic_frames(x)):
        ctx = FrameContext("s", robot_speaking=20 <= k < 50)
        gate.process(frame, ctx)
        gains.append(ctx.info["gate_gain"])
        echo.append(ctx.info.get("echo_db"))
    assert gains[19] == 1.0
    assert gains[21] == pytest.approx(10 ** (-30 / 20))
    assert gains[59] == pytest.approx(10 ** (-30 / 20))  # tail: 100 ms after the last frame
    assert gains[62] == 1.0
    assert echo[19] is None
    assert echo[60] == pytest.approx(20 * np.log10(0.1), abs=0.1)


def test_gate_ramps_without_steps() -> None:
    gate = DuplexGate(attenuation_db=-30, ramp_ms=20)
    frames = mic_frames(np.full(1600, 0.5, np.float32))
    out = [gate.process(f, FrameContext("s", robot_speaking=k >= 3)) for k, f in enumerate(frames)]
    y = np.concatenate([f.data[:, 0] for f in out])
    assert np.max(np.abs(np.diff(y))) < 0.01


def test_select_mic() -> None:
    fmt = AudioFormat(16000, 4, "float32")
    cmap = ChannelMap.from_names(["reference", "mic", "mic", "unused"])
    data = np.tile(np.array([0.9, 0.1, 0.3, 0.7], np.float32), (160, 1))
    frame = AudioFrame(data, fmt, cmap, 0, 0, "x")
    assert select_mic(frame, 0)[0, 0] == pytest.approx(0.1)
    assert select_mic(frame, 1)[0, 0] == pytest.approx(0.3)
    assert select_mic(frame, "mean")[0, 0] == pytest.approx(0.2)
    with pytest.raises(FormatError):
        select_mic(frame, 2)
    ref_only = AudioFrame(
        data[:, :1].copy(),
        AudioFormat(16000, 1, "float32"),
        ChannelMap.from_names(["reference"]),
        0,
        0,
        "x",
    )
    with pytest.raises(FormatError):
        select_mic(ref_only, 0)


# loopback (M2): the reference arrives in the capture stream on the mics' clock.


@pytest.mark.parametrize("mic_delay_ms", [0, 5])
def test_loopback_cancels_echo_from_capture_reference(
    apm_module: object, mic_delay_ms: int
) -> None:
    from ral.bench.offline import run_scene
    from ral.dsp.echo import ApmProcessor

    scene = echo_scene(seconds=8, delay_ms=20)
    proc = ApmProcessor(reference="channels", mic_delay_ms=mic_delay_ms)
    run = run_scene(scene, [proc], software_ref=False, loopback_ms=1.0)
    assert proc.lag_ms == 8 + mic_delay_ms
    assert aligned_erle(scene, run.clean, proc.lag_ms) > 25.0


def test_loopback_without_reference_channels_explains_itself(apm_module: object) -> None:
    from ral.dsp.echo import ApmProcessor

    proc = ApmProcessor(reference="channels")
    with pytest.raises(FormatError, match="reference channels"):
        proc.process(mic_frames(np.zeros(160, np.float32))[0], FrameContext("s"))


# hardware (M2): the device cancels echo itself.


def test_hardware_passthrough_takes_the_processed_channel() -> None:
    from ral.dsp.echo import HardwareAecPassthrough

    fmt = AudioFormat(16000, 3, "float32")
    cmap = ChannelMap.from_names(["processed", "mic", "reference"])
    data = np.stack([np.full(160, 0.01), np.full(160, 0.5), np.full(160, 0.3)], axis=1).astype(
        np.float32
    )
    ctx = FrameContext("s")
    out = HardwareAecPassthrough().process(AudioFrame(data, fmt, cmap, T0, 0, "arr"), ctx)
    np.testing.assert_allclose(out.data[:, 0], 0.01)
    assert out.t_ns == T0
    assert ctx.info["mic_db"] == pytest.approx(20 * np.log10(0.5), abs=0.01)
    assert ctx.info["ref_db"] == pytest.approx(20 * np.log10(0.3), abs=0.01)
    assert ctx.info["out_db"] == pytest.approx(-40.0, abs=0.01)


def test_hardware_passthrough_needs_a_processed_channel() -> None:
    from ral.dsp.echo import HardwareAecPassthrough

    with pytest.raises(FormatError, match="processed"):
        HardwareAecPassthrough().process(
            mic_frames(np.zeros(160, np.float32))[0], FrameContext("s")
        )


def test_hardware_mode_lets_barge_in_learn_the_residual() -> None:
    # A device that leaves -40 dB of residual echo: robot-only speech must not interrupt,
    # a user talking over it must.
    from ral.bench.offline import run_scene
    from ral.dsp.bargein import BargeInDetector
    from ral.dsp.echo import HardwareAecPassthrough
    from ral.dsp.vad import EnergyVad

    scene = echo_scene(seconds=20, delay_ms=20, near_spans=[(12.0, 15.0)], seed=5)
    processed = scene.echo * 10 ** (-40 / 20) + scene.near + scene.noise
    capture = (np.stack([processed, scene.mic], axis=1), ["processed", "mic"])
    run = run_scene(
        scene,
        [HardwareAecPassthrough(), EnergyVad(), BargeInDetector("vad+dtd")],
        software_ref=True,
        capture=capture,
    )
    assert len(run.barge_ins) == 1
    assert 11.9 < run.barge_ins[0] < 12.8
