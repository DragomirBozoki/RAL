"""Voice activity detection: a shared state machine and the energy VAD.

Every VAD engine produces a speech probability per analysis window; :class:`VadStateMachine`
turns those into ``SpeechStarted`` and ``SpeechEnded`` events with the same rules for all
engines:

- Speech starts when the probability stays at or above ``threshold`` for
  ``min_speech_ms``; brief dips below ``threshold - 0.15`` (hysteresis) cancel a start.
  ``SpeechStarted.t_ns`` is the first speech sample, not the moment of the decision.
- Speech ends after ``hangover_ms`` without speech (200 to 500 ms; it dominates turn
  latency, DESIGN section 5). ``SpeechEnded.t_ns`` is the end of the last speech window,
  so the hangover is visible in the latency breakdown instead of hidden in timestamps.

:class:`EnergyVad` is the dependency-free fallback: the frame level against an adaptive
noise floor that falls quickly and rises slowly (a simple form of minimum statistics,
R. Martin, "Noise power spectral density estimation based on optimal smoothing and
minimum statistics", IEEE Trans. Speech and Audio Processing, 2001). It cannot tell
speech from other sounds of similar loudness; Silero VAD can.

Both write ``info["vad_prob"]`` and ``info["speech"]`` (1.0 while speech is active).
"""

from __future__ import annotations

from ral.core.events import Event, SpeechEnded, SpeechStarted
from ral.core.processing import FrameContext
from ral.core.types import AudioFrame
from ral.dsp.levels import SILENCE_DB, rms_db

__all__ = ["EnergyVad", "VadStateMachine"]

_HYSTERESIS = 0.15


class VadStateMachine:
    """Turns per-window speech probabilities into speech start and end events.

    Args:
        threshold: Probability at or above which a window counts as speech.
        min_speech_ms: Speech must last this long before it starts.
        hangover_ms: Silence that ends speech.
    """

    def __init__(
        self, threshold: float = 0.5, min_speech_ms: int = 100, hangover_ms: int = 300
    ) -> None:
        self.threshold = threshold
        self.min_speech_ns = min_speech_ms * 1_000_000
        self.hangover_ns = hangover_ms * 1_000_000
        self.reset()

    def reset(self) -> None:
        self.speaking = False
        self._candidate: int | None = None  # start of a possible speech segment
        self._start: int | None = None  # start of the current segment
        self._last_speech_end: int | None = None

    def update(self, prob: float, t_ns: int, dur_ns: int, session_id: str) -> list[Event]:
        """Feed one window (start time and duration); returns any events it completes."""
        end = t_ns + dur_ns
        is_speech = prob >= self.threshold
        events: list[Event] = []
        if not self.speaking:
            if is_speech:
                if self._candidate is None:
                    self._candidate = t_ns
                if end - self._candidate >= self.min_speech_ns:
                    self.speaking = True
                    self._start = self._candidate
                    self._last_speech_end = end
                    events.append(SpeechStarted(self._candidate, session_id))
            elif prob < self.threshold - _HYSTERESIS:
                self._candidate = None
            return events
        if is_speech:
            self._last_speech_end = end
        elif self._last_speech_end is not None and end - self._last_speech_end >= self.hangover_ns:
            assert self._start is not None
            events.append(SpeechEnded(self._last_speech_end, session_id, self._start))
            self.speaking = False
            self._candidate = None
            self._start = None
        return events


class EnergyVad:
    """Energy VAD on the clean mono stream.

    Args:
        margin_db: Level above the noise floor that means speech.
        min_level_db: Frames quieter than this are never speech.
        floor_rise_db_per_s: How fast the noise floor may rise.
        threshold, min_speech_ms, hangover_ms: See :class:`VadStateMachine`.
    """

    def __init__(
        self,
        *,
        margin_db: float = 10.0,
        min_level_db: float = -55.0,
        floor_rise_db_per_s: float = 2.0,
        threshold: float = 0.5,
        min_speech_ms: int = 100,
        hangover_ms: int = 300,
    ) -> None:
        self.margin_db = margin_db
        self.min_level_db = min_level_db
        self.floor_rise_db_per_s = floor_rise_db_per_s
        self.state = VadStateMachine(threshold, min_speech_ms, hangover_ms)
        self.reset()

    def reset(self) -> None:
        self.state.reset()
        self.floor_db: float | None = None

    def process(self, frame: AudioFrame, ctx: FrameContext) -> AudioFrame:
        level = rms_db(frame.data[:, 0])
        dt = frame.fmt.frame_ns / 1e9
        if self.floor_db is None:
            self.floor_db = level
        elif level < self.floor_db:
            self.floor_db = max(SILENCE_DB, 0.7 * self.floor_db + 0.3 * level)
        else:
            self.floor_db += self.floor_rise_db_per_s * dt
        above = level - max(self.floor_db, -90.0)
        # A soft probability: 0 at the floor, 0.5 at the margin, 1 at twice the margin.
        prob = (
            0.0 if level < self.min_level_db else min(1.0, max(0.0, above / (2 * self.margin_db)))
        )
        ctx.info["vad_prob"] = prob
        ctx.events.extend(self.state.update(prob, frame.t_ns, frame.fmt.frame_ns, ctx.session_id))
        ctx.info["speech"] = 1.0 if self.state.speaking else 0.0
        return frame
