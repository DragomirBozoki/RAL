"""The reference lead, shared by the capture graph, echo control and delay tracking."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LeadControl", "as_lead"]


@dataclass
class LeadControl:
    """How far ahead of the mic frame the reference is fed, in ms (DESIGN 5).

    One instance is shared: the capture graph uses it to fetch reference frames, the APM
    stage to fill holes and as the stream-delay hint, and :class:`~ral.dsp.delay.DelayTracker`
    raises it when the reference is at risk of arriving after its echo.
    """

    ms: int = 20
    minimum_ms: int = 20

    @property
    def ns(self) -> int:
        return self.ms * 1_000_000


def as_lead(lead: int | LeadControl) -> LeadControl:
    """A :class:`LeadControl` from a plain value in ms, or the control itself."""
    return lead if isinstance(lead, LeadControl) else LeadControl(lead, lead)
