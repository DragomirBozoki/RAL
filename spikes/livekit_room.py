"""Spike: two participants exchanging 16 kHz audio and data through a LiveKit room.

Questions: with livekit (rtc) and livekit-api against a local `livekit-server --dev`, can
a "robot" participant publish 10 ms mono frames from RAL and receive another
participant's track as 16 kHz mono frames? What frame size does `rtc.AudioStream` deliver
with `frame_size_ms=10`, how late does audio arrive, and do data messages arrive?

Run: livekit-server --dev --bind 127.0.0.1   (in another terminal)
     uv run python spikes/livekit_room.py

Result, 2026-10-19, Windows 11 x86_64, CPython 3.12.15, livekit 1.1.20, livekit-api 1.2.2,
livekit-server 1.13.10 (--dev, localhost):

    robot: 232 frames, sizes [160], first audible 64 ms after send
    agent: 237 frames, sizes [160], first audible 64 ms after send
    data messages at robot: ['{"type":"hello"}']

Meaning: the bridge can publish RAL's clean 10 ms frames as they are (16 kHz mono) and
read the agent's track as 10 ms 16 kHz mono frames with `AudioStream(...,
frame_size_ms=10)`, which match the engine's playback format, so no resampling is needed
on either side. Room transport adds about 64 ms on localhost (jitter buffer included);
it belongs in the latency breakdown. Data messages carry events. At interpreter exit the
SDK prints an AssertionError from FfiHandle.__del__; it is shutdown noise in the SDK and
harmless, but the bridge disconnects explicitly so it does not appear during a run.
The dev keys trigger PyJWT's InsecureKeyLengthWarning; real deployments use long secrets.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
from livekit import api, rtc

URL = "ws://127.0.0.1:7880"
KEY, SECRET = "devkey", "secret"  # livekit-server --dev defaults
ROOM = "ral-spike"


def token(identity: str) -> str:
    grants = api.VideoGrants(room_join=True, room=ROOM)
    return api.AccessToken(KEY, SECRET).with_identity(identity).with_grants(grants).to_jwt()


async def publish_tone(room: rtc.Room, freq: float, seconds: float, start: list[float]) -> None:
    source = rtc.AudioSource(16000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await room.local_participant.publish_track(track, opts)
    await asyncio.sleep(0.5)  # let the subscriber attach
    start.append(time.perf_counter())
    n = 160
    for k in range(int(seconds * 100)):
        t = (np.arange(n) + k * n) / 16000
        pcm = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        await source.capture_frame(rtc.AudioFrame(bytearray(pcm.tobytes()), 16000, 1, n))
    await source.wait_for_playout()


async def main() -> None:
    robot, agent = rtc.Room(), rtc.Room()
    got: dict[str, list[tuple[float, int, float]]] = {"robot": [], "agent": []}
    data: list[str] = []
    readers: list[asyncio.Future[None]] = []

    def on_track(who: str):  # type: ignore[no-untyped-def]
        def handler(track: rtc.Track, *_: object) -> None:
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return

            async def read() -> None:
                stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1, frame_size_ms=10)
                async for ev in stream:
                    f = ev.frame
                    pcm = np.frombuffer(f.data, np.int16)
                    got[who].append(
                        (time.perf_counter(), f.samples_per_channel, float(np.abs(pcm).max()))
                    )

            readers.append(asyncio.ensure_future(read()))

        return handler

    robot.on("track_subscribed", on_track("robot"))
    agent.on("track_subscribed", on_track("agent"))
    robot.on("data_received", lambda pkt: data.append(pkt.data.decode()))
    await robot.connect(URL, token("robot"))
    await agent.connect(URL, token("agent"))
    starts: list[float] = []
    await asyncio.gather(
        publish_tone(robot, 440, 2.0, starts), publish_tone(agent, 660, 2.0, starts)
    )
    await agent.local_participant.publish_data('{"type":"hello"}', topic="ral.events")
    await asyncio.sleep(1.0)
    print("RESULT")
    for who, frames in got.items():
        sizes = sorted({s for _, s, _ in frames})
        loud = [t for t, _, peak in frames if peak > 3000]
        first = (loud[0] - min(starts)) * 1000 if loud else float("nan")
        print(
            f"  {who}: {len(frames)} frames, sizes {sizes}, first audible {first:.0f} ms after send"
        )
    print(f"  data messages at robot: {data}")
    await robot.disconnect()
    await agent.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
