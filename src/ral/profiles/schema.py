"""Profile schema, version 1 (DESIGN section 6, "Example profile").

A profile describes one robot (or laptop): where audio comes from and goes to, which
echo mode and pipeline to run, and which bridges to start. Unknown keys are errors, so a
typo does not silently fall back to a default.

Backends: ``portaudio`` and ``file``. Echo modes: ``hardware``, ``software_ref``,
``loopback``, ``gate`` and ``off``. TODO(M2): ``ros2_topic`` backends, decoder plugins and
migrations between schema versions arrive in later M2 steps.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "BargeInConfig",
    "CaptureConfig",
    "EchoConfig",
    "FormatConfig",
    "PlaybackConfig",
    "Profile",
    "RecordingConfig",
    "RobotConfig",
    "VadConfig",
]

SCHEMA_VERSION = 1
_ROLES = {"mic", "reference", "processed", "unused"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _off(v: Any) -> Any:
    # YAML 1.1 (PyYAML) reads a bare `off` as the boolean False.
    return "off" if v is False else v


class RobotConfig(_Strict):
    model: str = "unknown"
    firmware: str | None = None
    validated_by: Literal["community", "maintainer", "certified"] = "community"


class FormatConfig(_Strict):
    rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    dtype: Literal["int16", "float32"] = "float32"


class CaptureConfig(_Strict):
    backend: Literal["portaudio", "file"] = "portaudio"
    device: str | None = None
    """Device name (substring) or id; the default input device if omitted."""
    path: str | None = None
    """WAV file for the ``file`` backend."""
    realtime: bool = True
    """``file`` backend: pace by the clock (True) or run as fast as possible."""
    format: FormatConfig | None = None
    """Native format to open; the device's default rate and all channels if omitted."""
    channels: list[str] | None = None
    """Role of each channel; all ``mic`` if omitted."""

    @field_validator("channels")
    @classmethod
    def _roles(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            bad = [r for r in v if r not in _ROLES]
            if bad:
                raise ValueError(f"unknown channel roles {bad}; use {sorted(_ROLES)}")
        return v

    @model_validator(mode="after")
    def _file_needs_path(self) -> CaptureConfig:
        if self.backend == "file" and not self.path:
            raise ValueError("the file backend needs `path`")
        return self


class PlaybackConfig(_Strict):
    backend: Literal["portaudio", "file"] = "portaudio"
    device: str | None = None
    path: str | None = None
    realtime: bool = True
    format: FormatConfig | None = None
    latency_ms: int | None = None
    """Extra output latency to assume on top of what the backend reports."""

    @model_validator(mode="after")
    def _file_needs_path(self) -> PlaybackConfig:
        if self.backend == "file" and not self.path:
            raise ValueError("the file backend needs `path`")
        return self


class EchoConfig(_Strict):
    mode: Literal["hardware", "software_ref", "loopback", "gate", "off"] = "software_ref"
    engine: Literal["webrtc_aec3"] = "webrtc_aec3"
    mic: int | Literal["mean"] = 0
    lead_ms: int = Field(default=20, ge=0, le=200)
    mic_delay_ms: int = Field(default=0, ge=0, le=50)
    """``loopback``: delay the mic so the reference stays ahead of its echo."""
    drift_ppm: float = 0.0

    _mode_off = field_validator("mode", mode="before")(classmethod(lambda cls, v: _off(v)))


class VadConfig(_Strict):
    engine: Literal["silero", "energy"] = "silero"
    threshold: float = Field(default=0.5, gt=0, lt=1)
    min_speech_ms: int = Field(default=100, ge=10)
    hangover_ms: int = Field(default=300, ge=50, le=2000)


class BargeInConfig(_Strict):
    policy: Literal["off", "energy", "vad", "vad+dtd"] = "vad+dtd"
    min_speech_ms: int = Field(default=250, ge=50)

    _policy_off = field_validator("policy", mode="before")(classmethod(lambda cls, v: _off(v)))


class RecordingConfig(_Strict):
    enabled: bool = False
    path: str = "recordings"
    raw: bool = True
    """Store raw mic and reference audio, not only the processed stream and events."""


class EngineConfig(_Strict):
    rate: Literal[16000, 48000] = 16000


class Profile(_Strict):
    """A whole profile. Load it with :func:`ral.profiles.load_profile`."""

    ral: int = SCHEMA_VERSION
    robot: RobotConfig = RobotConfig()
    capture: CaptureConfig = CaptureConfig()
    playback: PlaybackConfig = PlaybackConfig()
    echo: EchoConfig = EchoConfig()
    pipeline: list[Literal["aec", "noise_suppression", "high_pass", "agc", "vad"]] = [
        "aec",
        "noise_suppression",
        "agc",
        "vad",
    ]
    vad: VadConfig = VadConfig()
    barge_in: BargeInConfig = BargeInConfig()
    bridges: dict[str, dict[str, Any]] = {}
    recording: RecordingConfig = RecordingConfig()
    engine: EngineConfig = EngineConfig()

    @field_validator("ral")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != SCHEMA_VERSION:
            raise ValueError(
                f"profile schema version {v} is not supported; this RAL reads version "
                f"{SCHEMA_VERSION}"
            )
        return v
