"""Minimal WebSocket client for RAL: print events and optionally speak a WAV file.

Start RAL with the WebSocket bridge, then run this in another terminal:

    uv run ral run --profile laptop --bridge websocket
    uv run python examples/websocket_client.py --say tests/data/speech_16k.wav

Protocol: see src/ral/bridges/websocket.py (version 1).
"""

from __future__ import annotations

import argparse
import asyncio
import json

from websockets.asyncio.client import connect

from ral.core.convert import float32_to_int16
from ral.playback.clip import load_wav


async def main(url: str, say: str | None) -> None:
    async with connect(url + "?audio=0") as ws:
        hello = json.loads(await ws.recv())
        print("connected:", hello)
        if say:
            pcm = float32_to_int16(load_wav(say, hello["playback"]["rate"])).astype("<i2")
            await ws.send(json.dumps({"type": "play", "tag": "example"}))
            for start in range(0, len(pcm), 1600):  # 100 ms chunks
                await ws.send(pcm[start : start + 1600].tobytes())
            await ws.send(json.dumps({"type": "end"}))
        async for message in ws:
            if isinstance(message, str):
                print(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--say", help="WAV file to play through the robot")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.say))
