"""Building the capture chain from a profile's ``echo`` and ``pipeline`` settings.

The WebRTC APM does echo cancellation, noise suppression, high-pass filtering and gain
control in one module, and its lags add up, so the builder merges every APM feature the
pipeline asks for into one :class:`~ral.dsp.echo.ApmProcessor` when the echo mode is
``software_ref``. In ``gate`` and ``off`` modes the APM features (if any) run as a second
stage after the gate, with echo cancellation off.

Pipeline stage names: ``aec`` (the echo controller; always present, the mode decides what
it does), ``noise_suppression``, ``high_pass``, ``agc`` and ``vad``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ral.core.errors import FormatError
from ral.core.processing import Processor
from ral.dsp.echo import ApmProcessor, DuplexGate, EchoOff, MicChoice
from ral.dsp.vad import EnergyVad

__all__ = ["EchoMode", "build_capture_chain"]

EchoMode = Literal["software_ref", "gate", "off"]
_APM_STAGES = {
    "noise_suppression": "noise_suppression",
    "high_pass": "high_pass_filter",
    "agc": "auto_gain_control",
}
_KNOWN = {"aec", "vad", *_APM_STAGES}


def build_capture_chain(
    mode: EchoMode,
    pipeline: Sequence[str] = ("aec", "noise_suppression", "agc", "vad"),
    *,
    mic: MicChoice = 0,
    lead_ms: int = 20,
    vad: Processor | None = None,
) -> list[Processor]:
    """The stages for one engine, in order.

    Args:
        mode: Echo-control mode. ``hardware`` and ``loopback`` arrive in M2.
        pipeline: Stage names from the profile.
        mic: Mic channel for echo control.
        lead_ms: Reference lead for ``software_ref``.
        vad: The VAD stage to use for ``vad`` (default: :class:`EnergyVad`).

    Raises:
        FormatError: If a stage or mode is unknown.
    """
    unknown = [s for s in pipeline if s not in _KNOWN]
    if unknown:
        raise FormatError(f"Unknown pipeline stages {unknown}; known: {sorted(_KNOWN)}.")
    if mode not in ("software_ref", "gate", "off"):
        raise FormatError(
            f"Echo mode {mode!r} is not available yet; use software_ref, gate or off "
            f"(hardware and loopback come in M2)."
        )
    apm_flags = {flag: stage in pipeline for stage, flag in _APM_STAGES.items()}
    chain: list[Processor] = []
    if mode == "software_ref":
        chain.append(ApmProcessor(mic, echo_cancellation=True, lead_ms=lead_ms, **apm_flags))
    else:
        chain.append(DuplexGate(mic) if mode == "gate" else EchoOff(mic))
        if any(apm_flags.values()):
            chain.append(ApmProcessor(0, echo_cancellation=False, **apm_flags))
    if "vad" in pipeline:
        chain.append(vad if vad is not None else EnergyVad())
    return chain
