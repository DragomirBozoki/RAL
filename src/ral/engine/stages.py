"""Building the capture stages from a profile, shared by the engine, replay and bench."""

from __future__ import annotations

import logging
from collections.abc import Callable

from ral.core.errors import RalError
from ral.core.processing import Processor
from ral.dsp.bargein import BargeInDetector
from ral.dsp.lead import LeadControl
from ral.dsp.pipeline import build_capture_chain
from ral.dsp.vad import EnergyVad
from ral.profiles.schema import Profile, VadConfig

__all__ = ["build_stages", "make_vad"]

log = logging.getLogger(__name__)


def make_vad(v: VadConfig) -> Processor:
    """The profile's VAD; falls back to the energy VAD if Silero is unavailable."""
    if v.engine == "silero":
        try:
            from ral.dsp.silero import SileroVad

            return SileroVad(
                threshold=v.threshold, min_speech_ms=v.min_speech_ms, hangover_ms=v.hangover_ms
            )
        except RalError as exc:
            log.warning("Silero VAD unavailable (%s); using the energy VAD instead.", exc)
    return EnergyVad(
        threshold=v.threshold, min_speech_ms=v.min_speech_ms, hangover_ms=v.hangover_ms
    )


def build_stages(
    profile: Profile,
    current_tag: Callable[[], str | None] | None = None,
    lead: LeadControl | None = None,
) -> list[Processor]:
    """Echo control, noise, gain, VAD and barge-in as the profile describes them."""
    p = profile
    stages: list[Processor] = build_capture_chain(
        p.echo.mode,
        p.pipeline,
        mic=p.echo.mic,
        lead_ms=lead if lead is not None else p.echo.lead_ms,
        delay_tracking=p.echo.delay_tracking,
        mic_delay_ms=p.echo.mic_delay_ms,
        vad=make_vad(p.vad),
    )
    stages.append(
        BargeInDetector(
            p.barge_in.policy, min_speech_ms=p.barge_in.min_speech_ms, current_tag=current_tag
        )
    )
    return stages
