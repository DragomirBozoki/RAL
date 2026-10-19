import numpy as np
import pytest

from ral.bench.offline import run_scene
from ral.bench.synthetic import echo_scene
from ral.core import AudioFormat, AudioFrame, ChannelMap
from ral.core import events as ev
from ral.core.processing import FrameContext
from ral.dsp.bargein import BargeInDetector
from ral.dsp.echo import DuplexGate, EchoOff
from ral.dsp.vad import EnergyVad
from ral.metrics.selftrigger import self_trigger_rate, self_triggers

MS = 1_000_000
FMT = AudioFormat(16000, 1, "float32")


def feed(det: BargeInDetector, infos: list[dict[str, float]], speaking: bool = True) -> list:
    out = []
    frame0 = np.zeros((160, 1), np.float32)
    for k, info in enumerate(infos):
        ctx = FrameContext("s", robot_speaking=speaking, info=dict(info))
        det.process(AudioFrame(frame0, FMT, ChannelMap.uniform(1), k * 10 * MS, k, "m"), ctx)
        out.extend(ctx.events)
    return out


def test_vad_policy_needs_min_duration_and_fires_once() -> None:
    det = BargeInDetector("vad", min_speech_ms=100, current_tag=lambda: "answer")
    events = feed(det, [{"speech": 0.0}] * 5 + [{"speech": 1.0}] * 30)
    assert len(events) == 1
    assert isinstance(events[0], ev.BargeIn)
    assert events[0].t_ns == 50 * MS
    assert events[0].interrupted_tag == "answer"


def test_short_bursts_and_long_gaps_do_not_fire() -> None:
    det = BargeInDetector("vad", min_speech_ms=100, gap_ms=20)
    pattern = ([{"speech": 1.0}] * 6 + [{"speech": 0.0}] * 4) * 5
    assert feed(det, pattern) == []


def test_nothing_fires_while_the_robot_is_silent() -> None:
    det = BargeInDetector("vad", min_speech_ms=50)
    assert feed(det, [{"speech": 1.0}] * 50, speaking=False) == []


def test_off_policy_never_fires() -> None:
    assert feed(BargeInDetector("off"), [{"speech": 1.0, "out_db": 0.0}] * 50) == []


def test_dtd_learns_leak_and_requires_residual_above_prediction() -> None:
    det = BargeInDetector("vad+dtd", min_speech_ms=50)
    learn = [{"out_db": -70.0, "ref_db": -25.0, "speech": 0.0}] * 80  # leak about -45 dB
    echo_like = [{"out_db": -68.0, "ref_db": -25.0, "speech": 1.0}] * 20
    user = [{"out_db": -35.0, "ref_db": -25.0, "speech": 1.0}] * 20
    assert feed(det, learn + echo_like) == []
    assert det.leak_db == pytest.approx(-45.0, abs=0.5)
    fired = feed(det, user)
    assert [type(e) for e in fired] == [ev.BargeIn]


def test_self_trigger_metric() -> None:
    s = 1_000_000_000
    playback = [(0, 30 * s), (40 * s, 70 * s)]  # one minute of playback
    starts = [5 * s, 10 * s, 35 * s, 50 * s]
    assert self_triggers(starts, playback, [(49 * s, 52 * s)]) == [5 * s, 10 * s]
    assert self_trigger_rate(starts, playback, [(49 * s, 52 * s)]) == pytest.approx(2.0)
    assert self_trigger_rate(starts, []) == 0.0


# End to end on synthetic echo (DESIGN section 10, M1 acceptance).


UTTERANCES = [(0.5, 3.5), (5.0, 8.0), (9.5, 12.5), (14.0, 17.0), (18.0, 19.5)]


@pytest.fixture(scope="module")
def robot_only():  # type: ignore[no-untyped-def]
    # The robot says five things with pauses between them; nobody else talks.
    return echo_scene(seconds=20, delay_ms=20, far_spans=UTTERANCES, seed=5)


def test_software_ref_lowers_the_self_trigger_rate(robot_only) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("livekit.rtc")
    from ral.dsp.echo import ApmProcessor

    off = run_scene(robot_only, [EchoOff(), EnergyVad()], software_ref=False, playback_s=UTTERANCES)
    aec = run_scene(
        robot_only, [ApmProcessor(), EnergyVad()], software_ref=True, playback_s=UTTERANCES
    )
    assert off.self_trigger_rate > 10.0  # the robot hears itself constantly
    # What remains is the canceller converging during the first utterance.
    assert aec.self_trigger_rate < off.self_trigger_rate / 4


def test_barge_in_fires_on_the_user_and_not_on_echo() -> None:
    pytest.importorskip("livekit.rtc")
    from ral.dsp.echo import ApmProcessor

    scene = echo_scene(seconds=20, delay_ms=20, near_spans=[(12.0, 15.0)], seed=5)
    run = run_scene(
        scene,
        [ApmProcessor(), EnergyVad(), BargeInDetector("vad+dtd")],
        software_ref=True,
        user_speech_s=[(12.0, 15.0)],
    )
    assert len(run.barge_ins) == 1
    assert 11.9 < run.barge_ins[0] < 12.8


def test_without_aec_the_echo_leak_guard_blocks_self_interruption(robot_only) -> None:  # type: ignore[no-untyped-def]
    naive = run_scene(
        robot_only,
        [EchoOff(), EnergyVad(), BargeInDetector("vad")],
        software_ref=True,
        playback_s=UTTERANCES,
    )
    guarded = run_scene(
        robot_only,
        [EchoOff(), EnergyVad(), BargeInDetector("vad+dtd")],
        software_ref=True,
        playback_s=UTTERANCES,
    )
    assert naive.barge_ins  # plain VAD interrupts the robot with its own echo
    assert guarded.barge_ins == []


def test_gate_mode_does_not_interrupt_itself(robot_only) -> None:  # type: ignore[no-untyped-def]
    run = run_scene(
        robot_only,
        [DuplexGate(), EnergyVad(), BargeInDetector("vad+dtd")],
        software_ref=False,
        playback_s=UTTERANCES,
    )
    assert run.barge_ins == []


def test_echo_leak_guard_blocks_residual_that_follows_the_reference() -> None:
    det = BargeInDetector("vad+dtd", min_speech_ms=250)
    rng = np.random.default_rng(0)
    learn = [{"out_db": -70.0, "ref_db": -25.0, "speech": 0.0}] * 60
    # The residual rises with the reference, 20 ms later: robot speech leaking through.
    ref = -25.0 + 10 * rng.standard_normal(80)
    leak = [
        {"out_db": float(ref[k - 2] + 5), "ref_db": float(ref[k]), "speech": 1.0}
        for k in range(2, 80)
    ]
    events = feed(det, learn + leak)
    assert not any(isinstance(e, ev.BargeIn) for e in events)
    assert any(isinstance(e, ev.EchoLeak) for e in events)
