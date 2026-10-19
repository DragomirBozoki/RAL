"""Regenerate tests/data/speech_16k.wav, a 10 s real-speech excerpt for VAD tests.

Source: tests/data/test.wav of https://github.com/snakers4/silero-vad at tag v6.2.3
(MIT License, Copyright (c) 2020-present Silero Team; see LICENSE-silero-vad.txt here).
The excerpt is seconds 4.0 to 14.0: speech, a pause of about two seconds, speech.

Run: uv run python tests/data/make_speech_clip.py
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import wave
from pathlib import Path

URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/tests/data/test.wav"
SHA256 = "89f17d9c94c4b31eb320f424628bcbc920abaddbee6e2760fd868bfb1d9a2e47"
START_S, END_S = 4.0, 14.0
OUT = Path(__file__).with_name("speech_16k.wav")


def main() -> None:
    with urllib.request.urlopen(URL, timeout=60) as resp:
        raw = resp.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"Unexpected source file (sha256 {digest}); refusing to cut it.")
    with wave.open(io.BytesIO(raw), "rb") as src:
        rate, width, channels = src.getframerate(), src.getsampwidth(), src.getnchannels()
        src.setpos(int(START_S * rate))
        pcm = src.readframes(int((END_S - START_S) * rate))
    with wave.open(str(OUT), "wb") as dst:
        dst.setnchannels(channels)
        dst.setsampwidth(width)
        dst.setframerate(rate)
        dst.writeframes(pcm)
    print(f"wrote {OUT} ({len(pcm)} bytes of PCM)")


if __name__ == "__main__":
    main()
