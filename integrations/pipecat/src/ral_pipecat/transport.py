"""``RalTransport``: a robot running RAL as a Pipecat transport (DESIGN section 7).

Swap a Pipecat bot's local audio transport for this one and the bot hears the robot's
echo-cancelled microphone and speaks through the robot's playback service, so its own
voice reaches RAL's echo reference. It talks to a running RAL over the WebSocket bridge
(``ral run --bridge websocket``), which keeps RAL in its own process: the audio engine
does not compete with the bot for the event loop, and ROS 2 robots and laptops look the
same to the bot.

- Input: RAL's clean 10 ms frames become ``InputAudioRawFrame``. Pipecat's own VAD and
  turn logic still run on them. A ``BargeIn`` from RAL broadcasts an interruption, so the
  bot cancels its turn as soon as RAL has faded the speech out. Every RAL event also goes
  to the ``on_ral_event`` handler.
- Output: ``OutputAudioRawFrame`` audio is sent as one RAL utterance (``play`` with the
  frame's sample rate; RAL resamples), ended when Pipecat reports the bot stopped
  speaking (after a TTS's ``TTSStoppedFrame``) or after ``end_after_ms`` without audio. An
  ``InterruptionFrame`` from Pipecat sends ``interrupt`` so RAL fades out at once. Writes
  are paced like a sound card with a ``lead_ms`` buffer (Pipecat's WebSocket transport
  paces the same way, with no buffer), so Pipecat's speaking state follows real time and
  RAL never runs dry between chunks.

Pipecat API as of pipecat-ai 1.13 (``transports/base_input.py``, ``base_output.py``,
``local/audio.py``, ``websocket/server.py``).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

__all__ = ["RalTransport", "RalTransportParams"]


class RalTransportParams(TransportParams):
    """Where RAL listens and how the bot's speech is played."""

    url: str = "ws://127.0.0.1:8765"
    tag: str = "pipecat"
    priority: int = 50
    end_after_ms: int = 300
    lead_ms: int = 100
    """How far ahead of real time audio is sent, like a sound card's buffer. RAL plays from
    its own queue; with less lead, network jitter becomes gaps in the robot's speech."""


class _Connection:
    """One WebSocket to RAL shared by the input and the output."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.ws: Any = None
        self.hello: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        async with self._lock:
            if self.ws is not None:
                return
            from websockets.asyncio.client import connect

            self.ws = await connect(self.url, compression=None)
            self.hello = json.loads(await self.ws.recv())
            if self.hello.get("type") != "hello" or self.hello.get("protocol") != 1:
                raise RuntimeError(f"{self.url} is not a RAL WebSocket bridge (protocol 1).")

    async def send(self, message: bytes | str) -> None:
        if self.ws is not None:
            await self.ws.send(message)

    async def close(self) -> None:
        async with self._lock:
            if self.ws is not None:
                await self.ws.close()
                self.ws = None


class RalInputTransport(BaseInputTransport):
    """RAL's clean microphone into the pipeline."""

    def __init__(self, transport: RalTransport, conn: _Connection, params: RalTransportParams):
        super().__init__(params)
        self._transport = transport
        self._conn = conn
        self._reader: asyncio.Task[None] | None = None

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self._conn.open()
        if self._reader is None:
            self._reader = self.create_task(self._read())
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame) -> None:
        await super().stop(frame)
        await self._stop_reader()

    async def cancel(self, frame: CancelFrame) -> None:
        await super().cancel(frame)
        await self._stop_reader()

    async def _stop_reader(self) -> None:
        if self._reader is not None:
            await self.cancel_task(self._reader)
            self._reader = None
        await self._conn.close()

    async def _read(self) -> None:
        rate = int(self._conn.hello.get("capture", {}).get("rate", 16000))
        async for message in self._conn.ws:
            if isinstance(message, bytes):
                await self.push_audio_frame(
                    InputAudioRawFrame(audio=message, sample_rate=rate, num_channels=1)
                )
                continue
            event = json.loads(message)
            if event.get("type") == "BargeIn":
                # RAL already faded the robot out; the bot must drop its turn too.
                await self.broadcast_interruption()
            await self._transport.call_ral_event(event)


class RalOutputTransport(BaseOutputTransport):
    """The bot's speech out through RAL's playback service."""

    def __init__(self, conn: _Connection, params: RalTransportParams):
        super().__init__(params)
        self._conn = conn
        self._params: RalTransportParams = params
        self._open_utterance = False
        self._last_write = 0.0
        self._next_send = 0.0
        self._idle: asyncio.Task[None] | None = None

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self._conn.open()
        if self._idle is None:
            self._idle = self.create_task(self._end_idle_utterances())
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame) -> None:
        await super().stop(frame)
        await self._finish()

    async def cancel(self, frame: CancelFrame) -> None:
        await super().cancel(frame)
        await self._finish()

    async def _finish(self) -> None:
        if self._idle is not None:
            await self.cancel_task(self._idle)
            self._idle = None
        if self._open_utterance:
            await self._conn.send(json.dumps({"type": "end"}))
            self._open_utterance = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, InterruptionFrame) and self._open_utterance:
            self._open_utterance = False
            await self._conn.send(json.dumps({"type": "interrupt"}))
        await super().process_frame(frame, direction)

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        # The base output announces BotStoppedSpeaking once the TTS's audio, flushed by its
        # TTSStoppedFrame, has been written: end the utterance right there, so RAL plays
        # the last (resampled) frame without waiting for the idle timeout.
        if isinstance(frame, BotStoppedSpeakingFrame) and self._open_utterance:
            self._open_utterance = False
            await self._conn.send(json.dumps({"type": "end"}))
        await super().push_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if self._conn.ws is None:
            return False
        if not self._open_utterance:
            await self._conn.send(
                json.dumps(
                    {
                        "type": "play",
                        "tag": self._params.tag,
                        "priority": self._params.priority,
                        "rate": frame.sample_rate,
                    }
                )
            )
            self._open_utterance = True
        await self._conn.send(frame.audio)
        self._last_write = time.monotonic()
        # Pace like a sound card with a buffer of lead_ms: Pipecat's speaking state follows
        # the writes, and RAL always holds enough audio to play without gaps.
        duration = len(frame.audio) / (2 * frame.num_channels * frame.sample_rate)
        now = time.monotonic()
        self._next_send = max(self._next_send, now) + duration
        await asyncio.sleep(max(0.0, self._next_send - now - self._params.lead_ms / 1000))
        return True

    async def _end_idle_utterances(self) -> None:
        while True:
            await asyncio.sleep(0.05)
            idle = time.monotonic() - self._last_write
            if self._open_utterance and idle * 1000 > self._params.end_after_ms:
                self._open_utterance = False
                await self._conn.send(json.dumps({"type": "end"}))


class RalTransport(BaseTransport):
    """Pipecat transport for a robot running RAL.

    Args:
        params: Connection and playback settings; ``audio_in_enabled`` and
            ``audio_out_enabled`` are switched on.
    """

    def __init__(self, params: RalTransportParams | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._params = params or RalTransportParams()
        self._params.audio_in_enabled = True
        self._params.audio_out_enabled = True
        self._conn = _Connection(self._params.url)
        self._input: RalInputTransport | None = None
        self._output: RalOutputTransport | None = None
        self._register_event_handler("on_ral_event")

    def input(self) -> FrameProcessor:
        if self._input is None:
            self._input = RalInputTransport(self, self._conn, self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if self._output is None:
            self._output = RalOutputTransport(self._conn, self._params)
        return self._output

    async def call_ral_event(self, event: dict[str, Any]) -> None:
        await self._call_event_handler("on_ral_event", event)
