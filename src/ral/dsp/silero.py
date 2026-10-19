"""Silero VAD on ONNX Runtime, without PyTorch (DESIGN section 5).

The model (https://github.com/snakers4/silero-vad, MIT) takes windows of 512 samples at
16 kHz (256 at 8 kHz) plus a recurrent state ``[2, batch, 128]`` and the sample rate, and
returns a speech probability and the next state. Each window must be preceded by the
last 64 samples (32 at 8 kHz) of the previous one: the official Python wrapper does the
same, and without that context the model returns almost zero on real speech
(``spikes/silero_io.py``).

Engine frames are 10 ms, so the stage buffers them into windows and stamps each window
with the capture time of its first sample. Other engine rates are resampled to 16 kHz.
A ``seq`` gap resets the buffer and the model state. Inference runs on one thread so the
VAD does not compete with audio threads for cores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ral.core.errors import FormatError, MissingDependencyError
from ral.core.processing import FrameContext
from ral.core.types import AudioFrame
from ral.dsp.resample import Resampler
from ral.dsp.vad import VadStateMachine
from ral.models import model_path

__all__ = ["SileroVad"]

_WINDOW = {16000: 512, 8000: 256}
_CONTEXT = {16000: 64, 8000: 32}


class SileroVad:
    """Silero VAD stage.

    Args:
        path: Model file; default: the cached ``silero_vad`` model (``ral models fetch``).
        threshold, min_speech_ms, hangover_ms: See :class:`~ral.dsp.vad.VadStateMachine`.

    Raises:
        MissingDependencyError: If ``onnxruntime`` is not installed.
        RalError: If the model is not downloaded.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        threshold: float = 0.5,
        min_speech_ms: int = 100,
        hangover_ms: int = 300,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise MissingDependencyError("onnxruntime", "dsp", "Silero VAD") from None
        model = Path(path) if path is not None else model_path("silero_vad")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session: Any = ort.InferenceSession(
            str(model), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.state = VadStateMachine(threshold, min_speech_ms, hangover_ms)
        self._resampler: Resampler | None = None
        self._rate: int | None = None
        self.windows = 0
        self.reset()

    def reset(self) -> None:
        self.state.reset()
        self._h = np.zeros((2, 1, 128), np.float32)
        self._context: npt.NDArray[np.float32] | None = None
        self._buf: npt.NDArray[np.float32] = np.zeros(0, np.float32)
        self._buf_t: int | None = None  # capture time of _buf[0]
        self._next_seq: int | None = None
        if self._resampler is not None:
            self._resampler.reset()
        self.last_prob = 0.0

    def process(self, frame: AudioFrame, ctx: FrameContext) -> AudioFrame:
        if self._next_seq is not None and frame.seq != self._next_seq:
            self.reset()
        self._next_seq = frame.seq + 1
        rate = self._model_rate(frame.fmt.sample_rate)
        x = frame.data[:, 0].astype(np.float32, copy=False)
        if self._resampler is not None:
            # Time-aligned resampler (ral.dsp.resample): output sample k is at input time k.
            x = self._resampler.process(x[:, None])[:, 0]
        if self._buf_t is None:
            self._buf_t = frame.t_ns
        self._buf = np.concatenate([self._buf, x]) if self._buf.size else x.copy()
        window, ctx_len = _WINDOW[rate], _CONTEXT[rate]
        win_ns = window * 1_000_000_000 // rate
        while self._buf.shape[0] >= window:
            chunk = self._buf[:window]
            self._buf = self._buf[window:]
            prob = self._infer(chunk, rate, ctx_len)
            ctx.events.extend(self.state.update(prob, self._buf_t, win_ns, ctx.session_id))
            self._buf_t += win_ns
            self.windows += 1
        ctx.info["vad_prob"] = self.last_prob
        ctx.info["speech"] = 1.0 if self.state.speaking else 0.0
        return frame

    def _model_rate(self, rate: int) -> int:
        if rate in _WINDOW:
            return rate
        if self._rate != rate:
            if rate % 100:
                raise FormatError(f"Silero VAD cannot use {rate} Hz audio.")
            self._resampler = Resampler(rate, 16000, 1)
            self._rate = rate
        return 16000

    def _infer(self, chunk: npt.NDArray[np.float32], rate: int, ctx_len: int) -> float:
        if self._context is None:
            self._context = np.zeros(ctx_len, np.float32)
        inp = np.concatenate([self._context, chunk])[None, :]
        out, self._h = self._session.run(
            None, {"input": inp, "state": self._h, "sr": np.array(rate, dtype=np.int64)}
        )
        self._context = inp[0, -ctx_len:].copy()
        self.last_prob = float(out[0, 0])
        return self.last_prob
