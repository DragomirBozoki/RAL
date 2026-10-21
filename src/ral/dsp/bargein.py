"""Barge-in: deciding that a user really interrupted the robot (DESIGN section 5).

Runs after the VAD on the clean stream, only while the robot is speaking. Policies:

- ``off``: never interrupts.
- ``energy``: the residual is loud: ``out_db`` above ``energy_db``.
- ``vad``: the VAD hears speech in the residual.
- ``vad+dtd`` (default): the VAD hears speech, *and* the double-talk detector sees the
  residual clearly above the echo it predicts, *and* the residual does not follow the
  reference (echo-leak guard).

The conditions must hold for ``min_speech_ms`` in total (250 ms by default), counting
only the frames where they hold; a pause longer than ``gap_ms`` (200 ms) starts over.
Counting frames matters in double talk: AEC3's suppressor lets a quieter user through
only between the robot's syllables. ``BargeIn.t_ns`` is when the interrupting speech
began. After a barge-in the detector stays quiet until the robot has stopped speaking.

Double-talk detection. During robot-only speech the detector learns how loud the
residual is relative to the reference (``leak_db``: the echo the canceller leaves). The
predicted echo of a frame is ``leak_db`` plus the loudest reference level of the last
``tail_ms`` (the echo of a word lingers after the word, as reverberation); speech
counts as double talk when ``out_db`` exceeds that by ``dtd_margin_db``. In ``gate``
mode there is no reference
level, so the raw mic must exceed the gate's learned echo level (``echo_db``) instead.
This is a level-based variant of the classic Geigel detector (D. L. Duttweiler, "A
twelve-channel digital echo canceler", IEEE Trans. Communications, 1978), adapted to a
residual after AEC.

Echo-leak guard. If the residual's level envelope follows the reference envelope (Pearson
correlation of per-frame dB values over the last ``guard_ms``, maximised over lags of up
to ``guard_max_lag_ms``), the "speech" is the robot's own voice leaking through. Then
barge-in is suppressed and an ``EchoLeak`` event is emitted (at most once a second).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Literal

import numpy as np

from ral.core.events import BargeIn, EchoLeak
from ral.core.processing import FrameContext
from ral.core.types import AudioFrame
from ral.dsp.levels import SILENCE_DB

__all__ = ["BargeInDetector", "BargeInPolicy"]

BargeInPolicy = Literal["off", "energy", "vad", "vad+dtd"]
_POLICIES = ("off", "energy", "vad", "vad+dtd")


class BargeInDetector:
    """Barge-in stage.

    Args:
        policy: See the module docstring.
        min_speech_ms: How long the conditions must hold.
        gap_ms: Short failures of the conditions that do not restart the count.
        energy_db: Residual level for the ``energy`` policy.
        dtd_margin_db: How far above the predicted echo the residual must be.
        guard_ms: Window of the echo-leak guard.
        guard_max_lag_ms: Largest residual-behind-reference lag the guard checks.
        guard_threshold: Envelope correlation above which the residual counts as echo.
        current_tag: Returns the tag of what is playing, for ``BargeIn.interrupted_tag``.
    """

    def __init__(
        self,
        policy: BargeInPolicy = "vad+dtd",
        *,
        min_speech_ms: int = 250,
        gap_ms: int = 200,
        energy_db: float = -40.0,
        dtd_margin_db: float = 10.0,
        guard_ms: int = 600,
        guard_max_lag_ms: int = 200,
        guard_threshold: float = 0.7,
        learn_rate: float = 0.05,
        min_learn_ms: int = 300,
        tail_ms: int = 300,
        current_tag: Callable[[], str | None] | None = None,
    ) -> None:
        if policy not in _POLICIES:
            raise ValueError(f"Unknown barge-in policy {policy!r}; use one of {_POLICIES}.")
        self.policy = policy
        self.min_speech_ns = min_speech_ms * 1_000_000
        self.gap_ns = gap_ms * 1_000_000
        self.energy_db = energy_db
        self.dtd_margin_db = dtd_margin_db
        self.guard_frames = max(5, guard_ms // 10)
        self.guard_lags = guard_max_lag_ms // 10
        self.guard_threshold = guard_threshold
        self.learn_rate = learn_rate
        self.min_learn_frames = max(1, min_learn_ms // 10)
        self.tail_frames = max(1, tail_ms // 10)
        self.current_tag = current_tag or (lambda: None)
        self.reset()

    def reset(self) -> None:
        self.leak_db: float | None = None
        """Learned residual-minus-reference level during robot-only speech."""
        self._leak_frames = 0
        self._echo_frames = 0
        self._start: int | None = None
        self._last_ok: int | None = None
        self._ok_ns = 0
        self._fired = False
        self._last_leak_event: int | None = None
        self._out_env: deque[float] = deque(maxlen=self.guard_frames + self.guard_lags)
        self._ref_env: deque[float] = deque(maxlen=self.guard_frames + self.guard_lags)
        self.fired = 0
        self.leaks = 0

    def process(self, frame: AudioFrame, ctx: FrameContext) -> AudioFrame:
        info = ctx.info
        out_db = info.get("out_db", SILENCE_DB)
        ref_now = info.get("ref_db")
        self._out_env.append(out_db)
        self._ref_env.append(ref_now if ref_now is not None else SILENCE_DB)
        # The echo of a frame lingers for the room's tail, so the echo to expect now
        # follows the loudest reference of the last tail_ms, not just this frame's.
        ref_db = None
        if ref_now is not None:
            recent = list(self._ref_env)[-self.tail_frames :]
            ref_db = max(recent)
        if not ctx.robot_speaking:
            self._fired = False
            self._start = self._last_ok = None
            return frame
        if self._fired or self.policy == "off":
            return frame

        speech = info.get("speech", 0.0) >= 0.5 or info.get("vad_prob", 0.0) >= 0.5
        dtd = self._double_talk(info, out_db, ref_db)
        if self.policy == "energy":
            ok = out_db > self.energy_db
        elif self.policy == "vad":
            ok = speech
        else:
            ok = speech and dtd
        if not speech and not dtd:
            self._learn(out_db, ref_db)

        t = frame.t_ns
        if ok:
            if self._start is None or (
                self._last_ok is not None and t - self._last_ok > self.gap_ns
            ):
                self._start = t
                self._ok_ns = 0
            self._last_ok = t
            # Count the frames that pass, not the wall-clock span: during double talk the
            # canceller's suppressor lets a quieter user through only between the robot's
            # syllables, so the evidence arrives in bursts.
            self._ok_ns += frame.fmt.frame_ns
            if self._ok_ns >= self.min_speech_ns:
                span = (t - self._start) // frame.fmt.frame_ns + 1
                if self.policy == "vad+dtd" and self._leaking(span):
                    self._emit_leak(t, ctx)
                    self._start = self._last_ok = None
                    return frame
                ctx.events.append(BargeIn(self._start, ctx.session_id, self.current_tag()))
                self._fired = True
                self.fired += 1
        elif self._last_ok is not None and t - self._last_ok > self.gap_ns:
            self._start = self._last_ok = None
        return frame

    def _double_talk(self, info: dict[str, float], out_db: float, ref_db: float | None) -> bool:
        # Until the echo has been learned (min_learn_ms of robot-only speech) nothing counts
        # as double talk: an unconverged canceller's residual looks like a talker.
        if ref_db is not None and self.leak_db is not None:
            if self._leak_frames < self.min_learn_frames:
                return False
            return out_db > ref_db + self.leak_db + self.dtd_margin_db
        echo_db = info.get("echo_db")
        if echo_db is not None:  # gate mode: raw mic against the learned echo level
            self._echo_frames += 1
            if self._echo_frames < self.min_learn_frames:
                return False
            return info.get("mic_db", SILENCE_DB) > echo_db + self.dtd_margin_db
        return False

    def _learn(self, out_db: float, ref_db: float | None) -> None:
        if ref_db is None or ref_db <= SILENCE_DB + 1:
            return
        leak = out_db - ref_db
        self._leak_frames += 1
        self.leak_db = (
            leak
            if self.leak_db is None
            else (1 - self.learn_rate) * self.leak_db + self.learn_rate * leak
        )

    def _leaking(self, span: int) -> bool:
        # Correlate over the candidate speech itself (at least 10 frames, at most guard_ms).
        n = int(min(self.guard_frames, max(10, span)))
        if len(self._out_env) < n:
            return True  # too little history to rule out a leak: stay conservative
        out = np.array(self._out_env, dtype=np.float64)[-n:]
        ref_all = np.array(self._ref_env, dtype=np.float64)
        if out.std() < 1e-6:
            return False
        best = -1.0
        for lag in range(min(self.guard_lags, len(ref_all) - n) + 1):
            ref = ref_all[len(ref_all) - n - lag : len(ref_all) - lag]
            if ref.std() < 1e-6:
                continue
            best = max(best, float(np.corrcoef(out, ref)[0, 1]))
        return best > self.guard_threshold

    def _emit_leak(self, t: int, ctx: FrameContext) -> None:
        self.leaks += 1
        if self._last_leak_event is None or t - self._last_leak_event >= 1_000_000_000:
            self._last_leak_event = t
            out = np.array(self._out_env, dtype=np.float64)[-self.guard_frames :]
            ref = np.array(self._ref_env, dtype=np.float64)[-self.guard_frames :]
            corr = float(np.corrcoef(out, ref)[0, 1]) if out.std() and ref.std() else 0.0
            ctx.events.append(EchoLeak(t, ctx.session_id, correlation=corr))
