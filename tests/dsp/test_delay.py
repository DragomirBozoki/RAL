import numpy as np
import pytest

from ral.bench.synthetic import echo_path, echo_scene, fft_convolve, voice
from ral.core import AudioFormat, AudioFrame, ChannelMap
from ral.core.processing import FrameContext
from ral.dsp.delay import DelayTracker, gcc_phat
from ral.dsp.lead import LeadControl, as_lead

FS = 16000
T0 = 10**12
FMT = AudioFormat(FS, 1, "float32")


def test_gcc_phat_finds_the_lag_and_its_direction() -> None:
    y = np.random.default_rng(0).standard_normal(8000)
    later = np.concatenate([np.zeros(37), y])[:8000]
    earlier = np.concatenate([y[23:], np.zeros(23)])
    assert gcc_phat(later, y, 200)[0] == 37
    assert gcc_phat(earlier, y, 200)[0] == -23
    assert gcc_phat(later, y, 200)[1] > 50  # a sharp, prominent peak


def track(
    delay_ms: int, seconds: int = 6, level: float = 0.1, max_lead_ms: int = 200
) -> tuple[DelayTracker, LeadControl]:
    """Echo arrives ``delay_ms`` after the reference's (estimated) DAC time."""
    acoustic = max(delay_ms, 0) + 5
    far = voice(FS * seconds, FS, 120, 1) * level
    echo = fft_convolve(far, echo_path(acoustic, FS, 3))
    shift_ns = (acoustic - delay_ms) * 1_000_000  # DAC estimate this much later than truth
    lead = LeadControl(20, 20)
    tracker = DelayTracker(lead, max_lead_ms=max_lead_ms)
    mic_map, ref_map = ChannelMap.uniform(1), ChannelMap.from_names(["reference"])
    for k in range(seconds * 100):
        t = T0 + k * 10_000_000
        sl = slice(k * 160, (k + 1) * 160)
        ref = AudioFrame(far[sl, None].astype(np.float32), FMT, ref_map, t + shift_ns, k, "dac")
        mic = AudioFrame(echo[sl, None].astype(np.float32), FMT, mic_map, t, k, "mic")
        tracker.process(mic, FrameContext("s", reference=[ref]))
    return tracker, lead


@pytest.mark.parametrize("delay_ms", [35, 5])
def test_positive_delays_are_measured_and_leave_the_lead(delay_ms: int) -> None:
    tracker, lead = track(delay_ms)
    assert tracker.delay_ms == pytest.approx(delay_ms, abs=0.2)
    assert lead.ms == 20


def test_a_late_reference_raises_the_lead() -> None:
    tracker, lead = track(-45)
    assert tracker.delay_ms == pytest.approx(-45, abs=0.2)
    assert lead.ms == 55  # 45 ms late plus a 10 ms margin


def test_a_silent_robot_gives_no_estimate() -> None:
    tracker, lead = track(-45, level=0.0)
    assert tracker.delay_ms is None
    assert tracker.estimates == 0
    assert lead.ms == 20


def test_lead_is_capped() -> None:
    tracker, lead = track(-150, max_lead_ms=100)
    assert tracker.delay_ms == pytest.approx(-150, abs=0.2)
    assert lead.ms == 100
    assert as_lead(30).ms == 30


def test_tracking_rescues_echo_control_from_a_late_dac_estimate() -> None:
    pytest.importorskip("livekit.rtc")
    from ral.bench.offline import run_scene
    from ral.dsp.echo import ApmProcessor
    from ral.metrics.echo import erle_db

    scene = echo_scene(seconds=20, delay_ms=5)

    def erle(tracking: bool) -> float:
        lead = LeadControl(20, 20)
        stages = [DelayTracker(lead), ApmProcessor(lead_ms=lead)] if tracking else [ApmProcessor()]
        run = run_scene(scene, stages, software_ref=True, lead_ms=lead, dac_error_ms=50)
        lag = 8 * 16
        mic = scene.mic[: len(run.clean) - lag]
        return erle_db(mic, run.clean[lag:], scene.far_only_frames(after_s=10.0), 160)

    without, with_ = erle(False), erle(True)
    assert without < 15.0  # the reference arrives 30 ms after its echo
    assert with_ > 30.0  # the tracker raised the lead and AEC3 recovered
