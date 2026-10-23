"""Building the capture chain from a profile's ``echo`` and ``pipeline`` settings.

The WebRTC APM does echo cancellation, noise suppression, high-pass filtering and gain
control in one module, and its lags add up, so the builder merges every APM feature the
pipeline asks for into one :class:`~ral.dsp.echo.ApmProcessor` when the echo mode is
``software_ref`` or ``loopback``. In ``hardware``, ``gate`` and ``off`` modes the APM
features (if any) run as a second stage after the first one, with echo cancellation off.

Pipeline stage names: ``aec`` (the echo controller; always present, the mode decides what
it does), ``noise_suppression``, ``high_pass``, ``agc`` and ``vad``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ral.core.errors import FormatError
from ral.core.processing import Processor
from ral.dsp.delay import DelayTracker
from ral.dsp.echo import ApmProcessor, DuplexGate, EchoOff, HardwareAecPassthrough, MicChoice
from ral.dsp.lead import LeadControl, as_lead
from ral.dsp.vad import EnergyVad

__all__ = ["EchoMode", "build_capture_chain"]

EchoMode = Literal["hardware", "software_ref", "loopback", "gate", "off"]
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
    lead_ms: int | LeadControl = 20,
    mic_delay_ms: int = 0,
    vad: Processor | None = None,
    delay_tracking: bool = False,
) -> list[Processor]:
    """The stages for one engine, in order.

    Args:
        mode: Echo-control mode.
        pipeline: Stage names from the profile.
        mic: Mic channel for echo control.
        lead_ms: Reference lead for ``software_ref`` (a shared control, or ms).
        delay_tracking: ``software_ref``: measure the echo delay and raise the lead when the
            reference risks arriving late (:class:`~ral.dsp.delay.DelayTracker`).
        mic_delay_ms: Mic delay for ``loopback``.
        vad: The VAD stage to use for ``vad`` (default: :class:`EnergyVad`).

    Raises:
        FormatError: If a stage or mode is unknown.
    """
    unknown = [s for s in pipeline if s not in _KNOWN]
    if unknown:
        raise FormatError(f"Unknown pipeline stages {unknown}; known: {sorted(_KNOWN)}.")
    if mode not in ("hardware", "software_ref", "loopback", "gate", "off"):
        raise FormatError(
            f"Unknown echo mode {mode!r}; use hardware, software_ref, loopback, gate or off."
        )
    ns, hpf, agc = (stage in pipeline for stage in _APM_STAGES)
    chain: list[Processor] = []
    lead = as_lead(lead_ms)
    if mode == "software_ref" and delay_tracking:
        chain.append(DelayTracker(lead, mic=mic))
    if mode in ("software_ref", "loopback"):
        chain.append(
            ApmProcessor(
                mic,
                echo_cancellation=True,
                noise_suppression=ns,
                high_pass_filter=hpf,
                auto_gain_control=agc,
                lead_ms=lead,
                reference="channels" if mode == "loopback" else "buffer",
                mic_delay_ms=mic_delay_ms,
            )
        )
    else:
        if mode == "hardware":
            chain.append(HardwareAecPassthrough())
        else:
            chain.append(DuplexGate(mic) if mode == "gate" else EchoOff(mic))
        if ns or hpf or agc:
            chain.append(
                ApmProcessor(
                    0,
                    echo_cancellation=False,
                    noise_suppression=ns,
                    high_pass_filter=hpf,
                    auto_gain_control=agc,
                )
            )
    if "vad" in pipeline:
        chain.append(vad if vad is not None else EnergyVad())
    return chain
