"""Spike: Silero VAD ONNX interface, and whether the 64-sample context matters.

Question: what does the Silero VAD v6.2.3 ONNX model expect, and must each 512-sample
window be preceded by the previous window's last 64 samples (as the official wrapper
does)?

Method: run the model over the 60 s test clip from the silero-vad repository (16 kHz) in
512-sample windows, once with the 64-sample context and once without.

Run: uv run python spikes/silero_io.py <dir with silero_vad.onnx and test.wav>

Result, 2026-10-19, Windows 11 x86_64, CPython 3.12.15, onnxruntime 1.31.0:

    inputs:  input [batch, samples] float32, state [2, batch, 128] float32, sr int64 scalar
    outputs: output [batch, 1] float32 (speech probability), stateN (next state)
    with context:    mean probability 0.778, 77.4 % of windows above 0.5
    without context: mean probability 0.001,  0.0 % of windows above 0.5

Meaning: the context is required. `ral.dsp.silero.SileroVad` prepends it to every window.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main(folder: Path) -> None:
    with wave.open(str(folder / "test.wav"), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    session = ort.InferenceSession(str(folder / "silero_vad.onnx"))
    for i in session.get_inputs():
        print("input", i.name, i.shape, i.type)
    for o in session.get_outputs():
        print("output", o.name, o.shape, o.type)
    for ctx_len in (64, 0):
        state = np.zeros((2, 1, 128), np.float32)
        ctx = np.zeros((1, ctx_len), np.float32)
        probs = []
        for i in range(0, len(x) - 512, 512):
            inp = np.concatenate([ctx, x[None, i : i + 512]], axis=1)
            out, state = session.run(
                None, {"input": inp, "state": state, "sr": np.array(16000, np.int64)}
            )
            ctx = inp[:, inp.shape[1] - ctx_len :] if ctx_len else ctx
            probs.append(float(out[0, 0]))
        p = np.array(probs)
        print(f"context {ctx_len}: mean {p.mean():.3f}, above 0.5: {(p > 0.5).mean():.1%}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
