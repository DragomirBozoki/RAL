"""Every echo mode in the same simulated room (DESIGN section 8, "Simulated rooms").

The robot's speaker sits 10 cm below its two mics; the user stands about 1.5 m away and
reaches the mics 12 dB quieter than the robot's echo. Echo at -20 dBFS, user at -32 dBFS,
RT60 0.35 s. Energy VAD (Silero needs real speech) and the default barge-in policy.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyroomacoustics")

from ral.bench.offline import SceneRun, run_scene
from ral.bench.room import RoomScene, room_scene
from ral.core.processing import Processor
from ral.dsp.bargein import BargeInDetector
from ral.dsp.echo import DuplexGate, EchoOff, HardwareAecPassthrough
from ral.dsp.vad import EnergyVad
from ral.metrics.echo import erle_db

UTTERANCES = [(0.5, 3.5), (5.0, 8.0), (9.5, 12.5), (14.0, 17.0), (18.0, 19.5)]
APM_MODES = ("software_ref", "loopback")


@pytest.fixture(scope="module")
def robot_only() -> RoomScene:
    return room_scene(far_spans=UTTERANCES)


@pytest.fixture(scope="module")
def double_talk() -> RoomScene:
    return room_scene(far_spans=[(0.5, 19.5)], near_spans=[(12.0, 15.0)])


def first_stage(mode: str) -> Processor:
    if mode in APM_MODES:
        pytest.importorskip("livekit.rtc")
        from ral.dsp.echo import ApmProcessor

        return ApmProcessor(reference="channels" if mode == "loopback" else "buffer")
    return {"off": EchoOff, "hardware": HardwareAecPassthrough, "gate": DuplexGate}[mode]()


def run(scene: RoomScene, mode: str, playback: list[tuple[float, float]], **kw: object) -> SceneRun:
    if mode == "loopback":
        capture = scene.with_loopback()
    elif mode == "hardware":
        capture = scene.hardware_array()
    else:
        capture = (scene.mics, ["mic", "mic"])
    stages = [first_stage(mode), EnergyVad(), BargeInDetector("vad+dtd")]
    return run_scene(
        scene.as_echo_scene(),
        stages,
        software_ref=mode in ("software_ref", "hardware"),
        playback_s=playback,
        capture=capture,
        **kw,  # type: ignore[arg-type]
    )


def test_levels_are_calibrated_at_the_mic(double_talk: RoomScene) -> None:
    def level(x: np.ndarray, active: np.ndarray) -> float:
        return float(10 * np.log10(np.mean(x[active] ** 2)))

    s = double_talk
    t = np.arange(len(s.far)) / s.rate
    near_active = (t >= 12.0) & (t < 15.0)
    assert level(s.echo[:, 0], s.far != 0) == pytest.approx(-20, abs=0.5)
    assert level(s.near[:, 0], near_active) == pytest.approx(-32, abs=1.0)


@pytest.mark.parametrize("mode", ["software_ref", "loopback", "hardware", "gate"])
def test_echo_control_stops_the_robot_hearing_itself(robot_only: RoomScene, mode: str) -> None:
    baseline = run(robot_only, "off", UTTERANCES)
    controlled = run(robot_only, mode, UTTERANCES)
    assert baseline.self_trigger_rate > 20.0
    assert controlled.self_trigger_rate <= 2.0
    assert controlled.barge_ins == []  # never interrupts itself


@pytest.mark.parametrize("mode", APM_MODES)
def test_room_erle(robot_only: RoomScene, mode: str) -> None:
    result = run(robot_only, mode, UTTERANCES)
    scene = robot_only.as_echo_scene()
    lag = 8 * 16
    mic = scene.mic[: len(result.clean) - lag]
    assert erle_db(mic, result.clean[lag:], scene.far_only_frames(after_s=3.0), 160) > 35.0


@pytest.mark.parametrize("mode", ["software_ref", "loopback", "hardware"])
def test_user_can_interrupt(double_talk: RoomScene, mode: str) -> None:
    result = run(double_talk, mode, [(0.5, 19.5)], user_speech_s=[(12.0, 15.0)])
    assert len(result.barge_ins) == 1
    # The user starts at 12.0 s, 12 dB below the echo; AEC3's suppressor lets them through
    # only between the robot's syllables, which costs up to about 1.2 s (measured).
    assert 11.95 < result.barge_ins[0] < 13.5


@pytest.mark.parametrize("mode", ["gate", "off"])
def test_without_full_duplex_the_robot_is_not_interrupted(
    double_talk: RoomScene, mode: str
) -> None:
    # gate: the user is quieter than the robot's own echo, so nothing exceeds the learned
    # echo level (DESIGN 5: a fallback, not full duplex). off: the echo-leak guard holds.
    result = run(double_talk, mode, [(0.5, 19.5)], user_speech_s=[(12.0, 15.0)])
    assert result.barge_ins == []
