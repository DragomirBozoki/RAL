from pathlib import Path

import pytest

from ral.core import ConfigError
from ral.profiles import builtin_profiles, load_profile, parse_profile


def test_defaults_are_a_valid_profile() -> None:
    p = parse_profile("ral: 1\n")
    assert p.echo.mode == "software_ref"
    assert p.echo.lead_ms == 20
    assert p.recording.enabled is False  # recording is opt-in
    assert p.barge_in.policy == "vad+dtd"


def test_builtin_laptop_profile_loads() -> None:
    assert "laptop" in builtin_profiles()
    p = load_profile("laptop")
    assert p.robot.model == "laptop"
    assert p.capture.backend == "portaudio"


def test_design_example_shape_parses(tmp_path: Path) -> None:
    text = """
ral: 1
robot: {model: example, validated_by: community}
capture:
  backend: file
  path: mic.wav
  channels: [mic, mic, reference]
echo: {mode: gate, mic: mean, lead_ms: 30}
pipeline: [aec, noise_suppression, agc, vad]
vad: {engine: energy, threshold: 0.6}
barge_in: {policy: vad}
bridges:
  websocket: {port: 8765}
recording: {enabled: false}
"""
    p = parse_profile(text)
    assert p.capture.channels == ["mic", "mic", "reference"]
    assert p.echo.mic == "mean"
    assert p.bridges["websocket"]["port"] == 8765


def test_env_vars_are_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LK_URL", "wss://lk.example")
    p = parse_profile("ral: 1\nbridges:\n  livekit: {url: '${LK_URL}', room: '${ROOM:-lab}'}\n")
    assert p.bridges["livekit"] == {"url": "wss://lk.example", "room": "lab"}


def test_missing_env_var_names_the_place(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOPE_NOT_SET", raising=False)
    with pytest.raises(ConfigError, match=r"profile\.bridges\.x\.url.*NOPE_NOT_SET"):
        parse_profile("bridges:\n  x: {url: '${NOPE_NOT_SET}'}\n")


@pytest.mark.parametrize(
    ("text", "where"),
    [
        ("echo: {mode: magic}", "echo.mode"),
        ("capture: {chanels: [mic]}", "capture.chanels"),
        ("capture: {channels: [mic, ears]}", "capture.channels"),
        ("capture: {backend: file}", "capture"),
        ("ral: 2", "ral"),
        ("pipeline: [aec, beamformer]", "pipeline"),
    ],
)
def test_errors_say_where(text: str, where: str) -> None:
    with pytest.raises(ConfigError) as exc:
        parse_profile(text)
    assert where in str(exc.value)


def test_bad_yaml_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not valid YAML"):
        parse_profile("a: [1, 2")
    with pytest.raises(ConfigError, match="built-in profiles: laptop"):
        load_profile(tmp_path / "nope.yaml")


def test_bare_off_is_not_a_boolean() -> None:
    p = parse_profile("echo: {mode: off}\nbarge_in: {policy: off}\n")
    assert p.echo.mode == "off"
    assert p.barge_in.policy == "off"
