"""Echo control stages (DESIGN section 5, "Echo-control modes").

All three stages take an engine frame (float32, any number of channels with roles) and
return a mono frame: the selected microphone after echo control. In v0.1 echo control
runs on one mic (``mic``: an index among the ``mic`` channels, or ``"mean"`` for their
average); multi-channel AEC comes later. The recorder still gets the raw channels.

- :class:`EchoOff` only selects the mic. It is the baseline benchmarks compare against.
- :class:`ApmProcessor` runs the WebRTC audio processing module (AEC3, noise suppression,
  high-pass filter, gain control) from the ``livekit`` package. With echo cancellation it
  implements ``software_ref``: before each mic frame it feeds the reverse stream every
  reference frame in ``ctx.reference`` (the engine asks the reference buffer for frames
  whose DAC time is before ``t_mic + lead``), then passes ``lead`` as the stream delay.
  AEC3's own delay estimator finds the exact offset (``spikes/apm_feasibility.py``).
- :class:`DuplexGate` implements ``gate``: while the robot speaks, plus a tail, the mic is
  attenuated. It also learns the echo level so barge-in can require speech above it.

Every stage writes ``info["mic_db"]`` (the selected mic before echo control) and
``info["out_db"]`` (after); the APM stage also writes ``info["ref_db"]`` (the reference
fed for this frame). Levels are dBFS.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ral.core.convert import float32_to_int16, int16_to_float32
from ral.core.errors import FormatError, MissingDependencyError
from ral.core.processing import FrameContext
from ral.core.types import AudioFormat, AudioFrame, ChannelMap, ChannelRole
from ral.dsp.lead import LeadControl, as_lead
from ral.dsp.levels import SILENCE_DB, db_to_gain, rms_db

__all__ = [
    "APM_LAG_MS",
    "ApmProcessor",
    "DuplexGate",
    "EchoOff",
    "HardwareAecPassthrough",
    "MicChoice",
    "select_mic",
]

MicChoice = int | Literal["mean"]

# Output lag of the WebRTC APM per feature, at 16 kHz (spikes/apm_lag.py); lags add up.
APM_LAG_MS = {"echo_cancellation": 8, "noise_suppression": 6}

_OUT_MAP = ChannelMap((ChannelRole.PROCESSED,))


def select_mic(frame: AudioFrame, mic: MicChoice) -> npt.NDArray[np.float32]:
    """The chosen mic channel of a float32 frame as ``(samples, 1)``.

    Raises:
        FormatError: If the frame has no mic channel or the index is out of range.
    """
    mics = frame.channel_map.indices(ChannelRole.MIC)
    if not mics:
        # A hardware-processed channel can stand in for a mic (hardware AEC arrays).
        mics = frame.channel_map.indices(ChannelRole.PROCESSED)
    if not mics:
        raise FormatError(
            f"The frame has no mic channel (roles {[r.value for r in frame.channel_map.roles]})."
        )
    if mic == "mean":
        out: npt.NDArray[np.float32] = frame.data[:, list(mics)].mean(
            axis=1, keepdims=True, dtype=np.float32
        )
        return out
    if not 0 <= mic < len(mics):
        raise FormatError(f"echo.mic is {mic}, but the frame has {len(mics)} mic channels.")
    return frame.data[:, mics[mic] : mics[mic] + 1].astype(np.float32, copy=True)


def _mono_frame(
    frame: AudioFrame, data: npt.NDArray[np.float32], t_ns: int | None = None
) -> AudioFrame:
    fmt = AudioFormat(frame.fmt.sample_rate, 1, "float32", frame.fmt.frame_ms)
    return AudioFrame(
        data, fmt, _OUT_MAP, frame.t_ns if t_ns is None else t_ns, frame.seq, frame.clock_id
    )


class EchoOff:
    """No echo control: select the mic and pass it on (the benchmark baseline)."""

    def __init__(self, mic: MicChoice = 0) -> None:
        self.mic = mic

    def process(self, frame: AudioFrame, ctx: FrameContext) -> AudioFrame:
        data = select_mic(frame, self.mic)
        level = rms_db(data)
        ctx.info["mic_db"] = level
        ctx.info["out_db"] = level
        return _mono_frame(frame, data)

    def reset(self) -> None:
        """Nothing to reset."""


class ApmProcessor:
    """WebRTC APM through ``livekit.rtc``: AEC3, noise suppression, high-pass, AGC.

    Args:
        mic: Which mic channel to process.
        echo_cancellation: Run AEC3 against ``ctx.reference`` (``software_ref``).
        noise_suppression: Run the APM's noise suppressor.
        high_pass_filter: Remove DC and rumble.
        auto_gain_control: Run the APM's gain controller.
        lead_ms: Reference lead (and stream-delay hint) for ``software_ref``, DESIGN 5.
        reference: Where the reference comes from: ``"buffer"`` (``ctx.reference``, what RAL
            rendered; ``software_ref``) or ``"channels"`` (the frame's own ``reference``
            channels on the mics' clock; ``loopback``).
        mic_delay_ms: ``loopback`` only: delay the mic by this much so the reference is
            always ahead of its echo, and pass it as the stream delay.

    Raises:
        MissingDependencyError: If ``livekit`` is not installed.
    """

    def __init__(
        self,
        mic: MicChoice = 0,
        *,
        echo_cancellation: bool = True,
        noise_suppression: bool = False,
        high_pass_filter: bool = False,
        auto_gain_control: bool = False,
        lead_ms: int | LeadControl = 20,
        reference: Literal["buffer", "channels"] = "buffer",
        mic_delay_ms: int = 0,
    ) -> None:
        if reference not in ("buffer", "channels"):
            raise FormatError(f"reference must be 'buffer' or 'channels', got {reference!r}.")
        if mic_delay_ms < 0:
            raise FormatError(f"mic_delay_ms must not be negative, got {mic_delay_ms}.")
        try:
            from livekit import rtc
        except ImportError:
            raise MissingDependencyError("livekit", "dsp", "WebRTC echo control") from None
        self._rtc = rtc
        self.mic = mic
        self.lead = as_lead(lead_ms)
        self.reference = reference
        self.mic_delay_ms = mic_delay_ms if reference == "channels" else 0
        self._flags = {
            "echo_cancellation": echo_cancellation,
            "noise_suppression": noise_suppression,
            "high_pass_filter": high_pass_filter,
            "auto_gain_control": auto_gain_control,
        }
        self.lag_ms = (
            sum(ms for name, ms in APM_LAG_MS.items() if self._flags[name]) + self.mic_delay_ms
        )
        """How far the output trails the input (APM lag plus any mic delay); clean frames
        are stamped earlier by this."""
        self.reference_frames = 0
        """Reference frames fed to the reverse stream."""
        self.silence_frames = 0
        """Silent frames fed to keep the reverse stream continuous."""
        self.late_reference_frames = 0
        """Reference frames dropped because they arrived after their time was covered."""
        self._apm: Any = None
        self._ref_until: int | None = None
        self._silence = np.zeros((0, 1), np.float32)
        self._delay_line: npt.NDArray[np.float32] = np.zeros((0, 1), np.float32)
        self.reset()

    @property
    def lead_ms(self) -> int:
        """The current reference lead (it may be raised by delay tracking)."""
        return self.lead.ms

    @property
    def echo_cancellation(self) -> bool:
        return self._flags["echo_cancellation"]

    def reset(self) -> None:
        """Start a fresh APM (its adaptive filters and estimators restart)."""
        self._apm = self._rtc.AudioProcessingModule(**self._flags)
        self._ref_until = None
        self._delay_line = np.zeros((0, 1), np.float32)

    def process(self, frame: AudioFrame, ctx: FrameContext) -> AudioFrame:
        rate, n = frame.fmt.sample_rate, frame.fmt.frame_samples
        if frame.fmt.frame_ms != 10:
            raise FormatError("The WebRTC APM needs exactly 10 ms frames.")
        if self._silence.shape[0] != n:
            self._silence = np.zeros((n, 1), np.float32)
        data = select_mic(frame, self.mic)
        ctx.info["mic_db"] = rms_db(data)
        if self.echo_cancellation and self.reference == "channels":
            self._feed_loopback(frame, ctx, rate, n)
            self._apm.set_stream_delay_ms(self.mic_delay_ms)
            data = self._delay(data, rate)
        elif self.echo_cancellation:
            self._feed_reverse(frame, ctx, rate, n)
            self._apm.set_stream_delay_ms(self.lead_ms)
        rtc_frame = self._to_rtc(data, rate, n)
        self._apm.process_stream(rtc_frame)
        pcm = np.frombuffer(rtc_frame.data, dtype=np.int16).reshape(n, 1)
        out = int16_to_float32(pcm.copy())
        ctx.info["out_db"] = rms_db(out)
        # The output carries audio captured lag_ms earlier (spikes/apm_lag.py).
        return _mono_frame(frame, out, frame.t_ns - self.lag_ms * 1_000_000)

    def _feed_loopback(self, frame: AudioFrame, ctx: FrameContext, rate: int, n: int) -> None:
        # loopback (DESIGN 5): the reference channels share the mics' frame and clock, so
        # the reverse stream gets this frame's reference first, then the mic is processed.
        refs = frame.channel_map.indices(ChannelRole.REFERENCE)
        if not refs:
            raise FormatError(
                "Echo mode loopback needs reference channels in the capture stream; mark them "
                "as `reference` in capture.channels, or use software_ref."
            )
        ref = frame.data[:, list(refs)].mean(axis=1, keepdims=True, dtype=np.float32)
        self._apm.process_reverse_stream(self._to_rtc(ref, rate, n))
        self.reference_frames += 1
        ctx.info["ref_db"] = rms_db(ref)

    def _delay(self, data: npt.NDArray[np.float32], rate: int) -> npt.NDArray[np.float32]:
        # A short delay line on the mic: the echo reaches the canceller mic_delay_ms after
        # its reference, which keeps timing errors on the early side (DESIGN 5).
        d = self.mic_delay_ms * rate // 1000
        if d == 0:
            return data
        if self._delay_line.shape[0] != d:
            self._delay_line = np.zeros((d, 1), np.float32)
        joined = np.concatenate([self._delay_line, data])
        self._delay_line = joined[-d:]
        out: npt.NDArray[np.float32] = joined[: data.shape[0]]
        return out

    def _feed_reverse(self, frame: AudioFrame, ctx: FrameContext, rate: int, n: int) -> None:
        # Keep the reverse stream continuous in DAC time up to t_mic + lead. AEC3 behaves
        # differently without a steady render stream (its output lag even changes), so
        # holes are filled with silence; a reference frame that arrives for time already
        # covered is late anyway (its echo has passed) and is dropped and counted.
        frame_ns = frame.fmt.frame_ns
        target = frame.t_ns + self.lead_ms * 1_000_000
        powers: list[float] = []
        for ref in ctx.reference:
            if ref.fmt.sample_rate != rate:
                raise FormatError(
                    f"Reference at {ref.fmt.sample_rate} Hz but mic at {rate} Hz; "
                    f"the reference buffer must produce engine-rate frames."
                )
            if self._ref_until is not None and ref.t_ns < self._ref_until - frame_ns // 2:
                self.late_reference_frames += 1
                continue
            self._fill_silence(ref.t_ns, frame_ns, rate, n, powers)
            self._apm.process_reverse_stream(self._to_rtc(ref.data[:, :1], rate, n))
            powers.append(float(np.mean(np.square(ref.data[:, 0], dtype=np.float64))))
            self.reference_frames += 1
            self._ref_until = ref.t_ns + frame_ns
        if self._ref_until is None:
            self._ref_until = target - frame_ns
        self._fill_silence(target, frame_ns, rate, n, powers)
        if powers:
            mean = sum(powers) / len(powers)
            ctx.info["ref_db"] = 10 * float(np.log10(mean)) if mean > 1e-12 else SILENCE_DB

    def _fill_silence(
        self, until_ns: int, frame_ns: int, rate: int, n: int, powers: list[float]
    ) -> None:
        if self._ref_until is None:
            return
        while self._ref_until + frame_ns // 2 < until_ns:
            self._apm.process_reverse_stream(self._to_rtc(self._silence, rate, n))
            powers.append(0.0)
            self.silence_frames += 1
            self._ref_until += frame_ns

    def _to_rtc(self, data: npt.NDArray[np.float32], rate: int, n: int) -> Any:
        pcm = float32_to_int16(data)
        return self._rtc.AudioFrame(bytearray(pcm.tobytes()), rate, 1, n)


class DuplexGate:
    """``gate`` mode: attenuate the mic while the robot speaks, plus a tail.

    Args:
        mic: Which mic channel to pass on.
        attenuation_db: Gain applied while gated (negative).
        tail_ms: How long the gate stays closed after the robot stops (room reverb and
            output buffering).
        ramp_ms: Gain ramp length, so gating does not click.
        learn_rate: Weight of each speaking frame in the echo level estimate.

    The learned echo level is written to ``info["echo_db"]``: a peak-hold of the mic level
    while the robot speaks, decaying by ``decay_db_per_s``. An average would lag behind the
    loud start of every utterance and invite false barge-ins. Frames more than
    ``outlier_db`` above the learned level are not learned: they are likely a user, not
    echo. Barge-in in ``gate`` mode requires the raw mic (``info["mic_db"]``) clearly above
    the learned level.
    """

    def __init__(
        self,
        mic: MicChoice = 0,
        *,
        attenuation_db: float = -45.0,
        tail_ms: int = 200,
        ramp_ms: int = 10,
        decay_db_per_s: float = 3.0,
        outlier_db: float = 10.0,
        learn_ms: int = 300,
    ) -> None:
        self.mic = mic
        self.gate_gain = db_to_gain(min(attenuation_db, 0.0))
        self.tail_ns = tail_ms * 1_000_000
        self.ramp_ns = max(1, ramp_ms) * 1_000_000
        self.decay_db_per_s = decay_db_per_s
        self.outlier_db = outlier_db
        self.learn_frames = max(1, learn_ms // 10)
        self.reset()

    def reset(self) -> None:
        self._gain = 1.0
        self._last_speaking_ns: int | None = None
        self._learned = 0
        self.echo_db: float | None = None
        """Learned echo level in dBFS, or None before the robot has spoken."""

    def _learn_echo(self, level: float, frame_ns: int) -> None:
        if level <= SILENCE_DB:
            return
        self._learned += 1
        if self.echo_db is None:
            self.echo_db = level
            return
        warmed_up = self._learned > self.learn_frames
        if level > self.echo_db and not (warmed_up and level > self.echo_db + self.outlier_db):
            self.echo_db = level
        else:
            self.echo_db -= self.decay_db_per_s * frame_ns / 1e9

    def process(self, frame: AudioFrame, ctx: FrameContext) -> AudioFrame:
        data = select_mic(frame, self.mic)
        level = rms_db(data)
        ctx.info["mic_db"] = level
        if ctx.robot_speaking:
            self._last_speaking_ns = frame.t_ns
            self._learn_echo(level, frame.fmt.frame_ns)
        closed = self._last_speaking_ns is not None and (
            frame.t_ns - self._last_speaking_ns <= self.tail_ns
        )
        target = self.gate_gain if closed else 1.0
        # Linear gain ramp across the frame, at most one ramp's worth of change per frame.
        step = frame.fmt.frame_ns / self.ramp_ns * (1.0 - self.gate_gain)
        if target < self._gain:
            new = max(target, self._gain - step)
        else:
            new = min(target, self._gain + step)
        ramp = np.linspace(self._gain, new, data.shape[0], endpoint=False, dtype=np.float32)
        self._gain = new
        out = (data * ramp[:, None]).astype(np.float32)
        ctx.info["out_db"] = rms_db(out)
        ctx.info["gate_gain"] = float(new)
        if self.echo_db is not None:
            ctx.info["echo_db"] = self.echo_db
        return _mono_frame(frame, out)


class HardwareAecPassthrough:
    """``hardware`` mode: the device cancels echo itself; pass its processed channel on.

    The array (an XMOS XVF3000 or XVF3800 board, for example) runs AEC on board against
    what it plays, so RAL must play through the same device (DESIGN section 5) and takes
    its ``processed`` channel. Software AEC stays off.

    Levels for barge-in: ``mic_db`` comes from a raw ``mic`` channel if the firmware exposes
    one (otherwise the processed channel), ``ref_db`` from a ``reference`` channel in the
    frame or from the reference frames RAL rendered (``ctx.reference``), so the double-talk
    detector can learn the residual echo the device leaves.

    Args:
        processed: Which ``processed`` channel to use when there are several.

    Raises:
        FormatError: If the frame has no ``processed`` channel.
    """

    def __init__(self, processed: int = 0) -> None:
        self.processed = processed

    def reset(self) -> None:
        """Nothing to reset."""

    def process(self, frame: AudioFrame, ctx: FrameContext) -> AudioFrame:
        cmap = frame.channel_map
        processed = cmap.indices(ChannelRole.PROCESSED)
        if not processed:
            raise FormatError(
                "Echo mode hardware needs a `processed` channel in capture.channels (the "
                "device's echo-cancelled output); see `ral doctor` for known devices."
            )
        if not 0 <= self.processed < len(processed):
            raise FormatError(
                f"processed channel {self.processed} requested, the frame has {len(processed)}."
            )
        i = processed[self.processed]
        out = frame.data[:, i : i + 1].astype(np.float32, copy=True)
        mics = cmap.indices(ChannelRole.MIC)
        ctx.info["mic_db"] = rms_db(frame.data[:, mics[0]] if mics else out)
        refs = cmap.indices(ChannelRole.REFERENCE)
        if refs:
            ctx.info["ref_db"] = rms_db(frame.data[:, list(refs)])
        elif ctx.reference:
            ctx.info["ref_db"] = rms_db(np.concatenate([r.data[:, 0] for r in ctx.reference]))
        ctx.info["out_db"] = rms_db(out)
        return _mono_frame(frame, out)
