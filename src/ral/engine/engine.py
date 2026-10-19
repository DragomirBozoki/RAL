"""Engine: one per robot. Opens devices, runs the capture graph and the playback service.

Threading (DESIGN section 9): device callbacks only copy into ring buffers (inside the
backends); an asyncio task pulls native frames from the source into a bounded queue; one
DSP worker thread runs :class:`~ral.engine.capture.CaptureGraph` frame by frame; the
asyncio loop runs playback, bridges and subscribers. The worker hands results back with
``call_soon_threadsafe``, so subscribers and the event bus are only touched on the loop.

The engine reacts to its own events in one place: a ``BargeIn`` fades out what is
playing (``PlaybackQueue.interrupt``) before bridges tell the agent to cancel its turn.

Everything RAL plays goes through :attr:`Engine.playback`, whose sink feeds the
reference buffer through its render hook (CLAUDE.md, reference invariant).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import threading
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ral.core.bus import EventBus
from ral.core.errors import ConfigError, RalError
from ral.core.events import BargeIn, Event
from ral.core.processing import Processor
from ral.core.protocols import AudioSink, AudioSource
from ral.core.types import AudioFormat, AudioFrame, ChannelMap
from ral.dsp.bargein import BargeInDetector
from ral.dsp.pipeline import build_capture_chain
from ral.dsp.vad import EnergyVad
from ral.engine.capture import CaptureGraph, CaptureResult
from ral.playback.reference import ReferenceBuffer
from ral.playback.service import PlaybackQueue
from ral.playback.writer import SinkWriter
from ral.profiles.schema import CaptureConfig, PlaybackConfig, Profile

__all__ = ["Engine", "ResultSubscription"]

log = logging.getLogger(__name__)

_INPUT_QUEUE_FRAMES = 200  # 2 s of native frames between the capture task and the worker


class ResultSubscription:
    """Async iterator over :class:`CaptureResult` objects (raw, clean, context).

    A slow subscriber loses its oldest results (``dropped``) instead of stalling the
    engine. ``clean()`` iterates only the clean frames.
    """

    def __init__(self, engine: Engine, maxsize: int) -> None:
        self._engine = engine
        self._queue: asyncio.Queue[CaptureResult | None] = asyncio.Queue(maxsize)
        self.dropped = 0
        self.closed = False

    def offer(self, item: CaptureResult | None) -> None:
        if self._queue.full():
            self._queue.get_nowait()
            self.dropped += 1
        self._queue.put_nowait(item)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._engine._subs.discard(self)
            self.offer(None)

    def __aiter__(self) -> AsyncIterator[CaptureResult]:
        return self

    async def __anext__(self) -> CaptureResult:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def clean(self) -> AsyncIterator[AudioFrame]:
        """Only the clean (echo-controlled) frames."""
        async for result in self:
            yield result.clean


class Engine:
    """Runs one profile.

    Args:
        profile: What to run.
        session_id: Stamped on every event; a random id by default.
        portaudio: A ``sounddevice``-compatible module for the PortAudio backend (tests).
    """

    def __init__(
        self, profile: Profile, *, session_id: str | None = None, portaudio: Any = None
    ) -> None:
        self.profile = profile
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.engine_fmt = AudioFormat(profile.engine.rate, 1, "float32")
        self.bus = EventBus()
        self._sd = portaudio
        self.source: AudioSource | None = None
        self.sink: AudioSink | None = None
        self.reference: ReferenceBuffer | None = None
        self.playback: PlaybackQueue | None = None
        self.graph: CaptureGraph | None = None
        self._subs: set[ResultSubscription] = set()
        self._in_q: queue.Queue[AudioFrame | None] = queue.Queue(_INPUT_QUEUE_FRAMES)
        self._worker: threading.Thread | None = None
        self._capture_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._remove_listener: Any = None
        self.capture_done = asyncio.Event()
        """Set when the source ended (a file ran out) and every frame was processed."""
        self.input_overruns = 0
        """Native frames dropped because the DSP worker fell behind."""
        self.worker_error: BaseException | None = None

    # Lifecycle

    async def start(self) -> None:
        """Open devices and start capture, processing and playback.

        Raises:
            ConfigError: If the profile cannot be realised on this host.
            RalError: If a device cannot be opened.
        """
        try:
            await self._start()
        except BaseException:
            await self.stop()  # close whatever was already opened
            raise

    async def _start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.bus.bind(self._loop)
        self.capture_done = asyncio.Event()
        p = self.profile

        sink = self._open_sink(p.playback)
        self.sink = sink
        self.reference = ReferenceBuffer(sink.fmt, self.engine_fmt.sample_rate)
        sink.on_rendered(self.reference.push)
        writer = SinkWriter(sink, self.engine_fmt, realtime=self._realtime(p.playback))
        self.playback = PlaybackQueue(writer, self.bus, self.session_id)

        source = self._open_source(p.capture)
        self.source = source
        playback = self.playback
        stages: list[Processor] = build_capture_chain(
            p.echo.mode, p.pipeline, mic=p.echo.mic, lead_ms=p.echo.lead_ms, vad=self._make_vad()
        )
        stages.append(
            BargeInDetector(
                p.barge_in.policy,
                min_speech_ms=p.barge_in.min_speech_ms,
                current_tag=lambda: playback.current.tag if playback.current else None,
            )
        )
        self.graph = CaptureGraph(
            source.fmt,
            source.channel_map,
            stages,
            reference=self.reference,
            lead_ms=p.echo.lead_ms,
            speaking=playback.speaking_at,
            session_id=self.session_id,
            engine_rate=self.engine_fmt.sample_rate,
        )
        self._remove_listener = self.bus.add_listener(self._on_event)

        await sink.start()
        await playback.start()
        self._worker = threading.Thread(target=self._dsp_loop, name="ral-dsp", daemon=True)
        self._worker.start()
        await source.start()
        self._capture_task = asyncio.create_task(self._capture(source), name="ral-capture")
        log.info(
            "Engine %s started: %s -> %s, echo %s",
            self.session_id,
            source.fmt,
            sink.fmt,
            p.echo.mode,
        )

    async def stop(self) -> None:
        """Stop capture, processing and playback, and close devices."""
        if self._capture_task is not None:
            self._capture_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._capture_task
            self._capture_task = None
        if self.source is not None:
            with contextlib.suppress(Exception):
                await self.source.stop()
        if self._worker is not None:
            self._put(None)
            await asyncio.get_running_loop().run_in_executor(None, self._worker.join, 5.0)
            self._worker = None
        if self.playback is not None:
            await self.playback.close()
        if self.sink is not None:
            with contextlib.suppress(Exception):
                await self.sink.stop()
        if self._remove_listener is not None:
            self._remove_listener()
            self._remove_listener = None
        for sub in list(self._subs):
            sub.close()

    async def __aenter__(self) -> Engine:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def subscribe(self, maxsize: int = 200) -> ResultSubscription:
        """Receive every processed frame (raw, clean and its context)."""
        sub = ResultSubscription(self, maxsize)
        self._subs.add(sub)
        return sub

    # Device setup

    @staticmethod
    def _realtime(cfg: CaptureConfig | PlaybackConfig) -> bool:
        return cfg.backend != "file" or cfg.realtime

    def _portaudio(self) -> Any:
        from ral.devices.portaudio import PortAudioBackend

        return PortAudioBackend(self._sd)

    def _open_source(self, cfg: CaptureConfig) -> AudioSource:
        if cfg.backend == "file":
            from ral.devices.file import FileBackend

            assert cfg.path is not None
            backend = FileBackend(sources={"mic": cfg.path}, realtime=cfg.realtime)
            (dev,) = backend.discover()
            rate = int(dev.default_sample_rate or 0)
            dtype = cfg.format.dtype if cfg.format else "int16"
            fmt = AudioFormat(rate, dev.max_input_channels, dtype)
            return backend.open_source(dev, fmt, self._channel_map(cfg, fmt.channels))
        pa = self._portaudio()
        dev = pa.find(cfg.device, "input") if cfg.device else pa.default_device("input")
        if cfg.format is not None:
            fmt = AudioFormat(cfg.format.rate, cfg.format.channels, cfg.format.dtype)
        else:
            channels = len(cfg.channels) if cfg.channels else None
            fmt = pa.native_format(dev, "input", "float32", channels)
        source: AudioSource = pa.open_source(dev, fmt, self._channel_map(cfg, fmt.channels))
        return source

    def _open_sink(self, cfg: PlaybackConfig) -> AudioSink:
        if cfg.backend == "file":
            from ral.devices.file import FileBackend

            assert cfg.path is not None
            Path(cfg.path).parent.mkdir(parents=True, exist_ok=True)
            backend = FileBackend(sinks={"speaker": cfg.path}, realtime=cfg.realtime)
            (dev,) = backend.discover()
            f = cfg.format
            fmt = AudioFormat(f.rate, f.channels, f.dtype) if f else self.engine_fmt
            return backend.open_sink(dev, fmt)
        pa = self._portaudio()
        dev = pa.find(cfg.device, "output") if cfg.device else pa.default_device("output")
        if cfg.format is not None:
            fmt = AudioFormat(cfg.format.rate, cfg.format.channels, cfg.format.dtype)
        else:
            fmt = pa.native_format(dev, "output", "float32", min(2, dev.max_output_channels))
        sink: AudioSink = pa.open_sink(dev, fmt)
        return sink

    @staticmethod
    def _channel_map(cfg: CaptureConfig, channels: int) -> ChannelMap:
        if cfg.channels is None:
            return ChannelMap.uniform(channels)
        if len(cfg.channels) != channels:
            raise ConfigError(
                f"capture.channels lists {len(cfg.channels)} roles but the device is opened "
                f"with {channels} channels; list one role per channel."
            )
        return ChannelMap.from_names(cfg.channels)

    def _make_vad(self) -> Processor:
        v = self.profile.vad
        if v.engine == "silero":
            try:
                from ral.dsp.silero import SileroVad

                return SileroVad(
                    threshold=v.threshold, min_speech_ms=v.min_speech_ms, hangover_ms=v.hangover_ms
                )
            except RalError as exc:
                log.warning("Silero VAD unavailable (%s); using the energy VAD instead.", exc)
        return EnergyVad(
            threshold=v.threshold, min_speech_ms=v.min_speech_ms, hangover_ms=v.hangover_ms
        )

    # Capture and DSP

    def _put(self, item: AudioFrame | None) -> None:
        while True:
            try:
                self._in_q.put_nowait(item)
                return
            except queue.Full:
                # The worker fell behind: drop the oldest frame (the Normalizer sees the gap).
                with contextlib.suppress(queue.Empty):
                    self._in_q.get_nowait()
                    self.input_overruns += 1

    async def _capture(self, source: AudioSource) -> None:
        # A live device cannot wait, so a full queue drops its oldest frame; an offline
        # file can, so it waits for the DSP worker instead (no audio is lost).
        lossless = not self._realtime(self.profile.capture)
        try:
            async for frame in source.frames():
                while lossless and self._in_q.full():
                    await asyncio.sleep(0.002)
                self._put(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Capture failed; the engine keeps playback running")
        # End of stream: the worker flushes and reports done. The marker must not evict
        # a frame either.
        while lossless and self._in_q.full():
            await asyncio.sleep(0.002)
        self._put(None)

    def _dsp_loop(self) -> None:
        graph = self.graph
        assert graph is not None
        assert self._loop is not None
        loop = self._loop
        try:
            while True:
                frame = self._in_q.get()
                results = graph.flush() if frame is None else graph.process(frame)
                for result in results:
                    loop.call_soon_threadsafe(self._deliver, result)
                if frame is None:
                    loop.call_soon_threadsafe(self.capture_done.set)
                    return
        except BaseException as exc:  # surface worker failures instead of dying silently
            self.worker_error = exc
            log.exception("DSP worker failed")
            loop.call_soon_threadsafe(self.capture_done.set)

    def _deliver(self, result: CaptureResult) -> None:
        for event in result.ctx.events:
            self.bus.publish(event)
        for sub in list(self._subs):
            sub.offer(result)

    def _on_event(self, event: Event) -> None:
        if isinstance(event, BargeIn) and self.playback is not None:
            self.playback.interrupt()
