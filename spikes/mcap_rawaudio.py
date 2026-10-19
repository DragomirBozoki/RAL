"""Spike: storing audio as Foxglove RawAudio in MCAP with the mcap Python library.

Questions: does a JSON-encoded `foxglove.RawAudio` channel (schema copied from the
foxglove-sdk repository, MIT) round-trip through mcap 1.5 bit-exactly, and how large is
a minute of 16 kHz mono speech-like audio with base64 inside JSON and zstd chunks?

Run: uv run python spikes/mcap_rawaudio.py tests/data/speech_16k.wav

Result, 2026-10-19, Windows 11 x86_64, CPython 3.12.15, mcap 1.5.0 (zstandard 0.25):

    round trip: bit-exact (int16 PCM after base64 and JSON)
    real speech, 10 ms messages: 1.94 MB per minute per mono 16 kHz channel
    (raw PCM is 1.92 MB per minute; base64 adds a third, zstd takes it back)

Meaning: JSON RawAudio is good enough for v0.1 recordings (a 10-minute session with mic,
reference and clean audio is about 60 MB) and opens in Foxglove without plugins. If size
matters later, a protobuf RawAudio channel stores raw bytes without base64 (M2+).
"""

from __future__ import annotations

import base64
import io
import json
import sys
import wave

import numpy as np
from mcap.reader import make_reader
from mcap.writer import Writer

SCHEMA = {
    "title": "foxglove.RawAudio",
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
        },
        "data": {"type": "string", "contentEncoding": "base64"},
        "format": {"type": "string"},
        "sample_rate": {"type": "integer"},
        "number_of_channels": {"type": "integer"},
    },
}


def main(path: str) -> None:
    with wave.open(path, "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    buf = io.BytesIO()
    writer = Writer(buf)
    writer.start()
    sid = writer.register_schema("foxglove.RawAudio", "jsonschema", json.dumps(SCHEMA).encode())
    cid = writer.register_channel("/audio", "json", sid)
    for k in range(len(pcm) // 160):
        t = k * 10_000_000
        block = pcm[k * 160 : (k + 1) * 160]
        msg = {
            "timestamp": {"sec": t // 10**9, "nsec": t % 10**9},
            "data": base64.b64encode(block.tobytes()).decode(),
            "format": "pcm-s16",
            "sample_rate": 16000,
            "number_of_channels": 1,
        }
        writer.add_message(cid, t, json.dumps(msg).encode(), t)
    writer.finish()
    size = buf.tell()
    buf.seek(0)
    back = []
    for _, _, message in make_reader(buf).iter_messages():
        back.append(np.frombuffer(base64.b64decode(json.loads(message.data)["data"]), "<i2"))
    same = np.array_equal(np.concatenate(back), pcm[: len(back) * 160])
    minutes = len(pcm) / 16000 / 60
    print(f"round trip bit-exact: {same}; {size / minutes / 1e6:.2f} MB per minute")


if __name__ == "__main__":
    main(sys.argv[1])
