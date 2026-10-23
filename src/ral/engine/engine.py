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
from ral.core.clock import now_ns
from ral.core.errors import ConfigError
from ral.core.events import BargeIn, Event, RecordingStarted, RecordingStopped
from ral.core.protocols import AudioSink, AudioSource
from ral.core.types import AudioFormat, AudioFrame, ChannelMap, ChannelRole
from ral.dsp.drift import DriftEstimator
from ral.dsp.lead import LeadControl
from ral.engine.capture import CaptureGraph, CaptureResult
from ral.engine.stages import build_stages
from ral.metrics.health import HealthMonitor
from ral.metrics.latency import LatencyTracker
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
        self,
        profile: Profile,
        *,
        session_id: str | None = None,
        portaudio: Any = None,
        record: bool | None = None,
    ) -> None:
        self.profile = profile
        self.record = profile.recording.enabled if record is None else record
        self.recorder: Any = None
        self._record_task: asyncio.Task[None] | None = None
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
        self._remove_latency: Any = None
        self._ros2: Any = None
        self.capture_estimator: DriftEstimator | None = None
        self.health: HealthMonitor | None = None
        self._health_tasks: list[asyncio.Task[None]] = []
        self.latency = LatencyTracker()
        """Per-turn latency; replaced with a publishing tracker on start."""
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
        self.reference = ReferenceBuffer(
            sink.fmt,
            self.engine_fmt.sample_rate,
            drift=p.echo.mode == "software_ref" and p.echo.drift_compensation,
            initial_ppm=p.echo.drift_ppm,
        )
        sink.on_rendered(self.reference.push)
        writer = SinkWriter(sink, self.engine_fmt, realtime=self._realtime(p.playback))
        self.playback = PlaybackQueue(writer, self.bus, self.session_id)

        source = self._open_source(p.capture)
        self.source = source
        if p.echo.mode == "hardware":
            self._check_hardware(source)
        framer = getattr(source, "framer", None)
        if framer is not None:
            # The capture clock's real rate, for drift compensation and health metrics.
            self.capture_estimator = DriftEstimator(source.fmt.sample_rate)
            framer.observer = self.capture_estimator.observe
            framer.rate_source = self.capture_estimator.rate_hz
            self.reference.capture_estimator = self.capture_estimator
        playback = self.playback
        self.lead = LeadControl(p.echo.lead_ms, p.echo.lead_ms)
        stages = build_stages(
            p,
            current_tag=lambda: playback.current.tag if playback.current else None,
            lead=self.lead,
        )
        self.graph = CaptureGraph(
            source.fmt,
            source.channel_map,
            stages,
            reference=self.reference,
            lead_ms=self.lead,
            speaking=playback.speaking_at,
            session_id=self.session_id,
            engine_rate=self.engine_fmt.sample_rate,
        )
        self._remove_listener = self.bus.add_listener(self._on_event)
        self._start_health(source)
        loop = self._loop

        def publish_later(event: Event) -> None:
            loop.call_soon(self.bus.publish, event)  # not from inside the bus's own publish

        self.latency = LatencyTracker(publish=publish_later)
        self._remove_latency = self.bus.add_listener(self.latency.on_event)
        if self.record:
            self._start_recorder()

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
        try:
            await self._stop()
        finally:
            await self._stop_recorder()

    async def _stop(self) -> None:
        for task in self._health_tasks:
            task.cancel()
        for task in self._health_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._health_tasks = []
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
        if self._remove_latency is not None:
            self._remove_latency()
            self._remove_latency = None
        for sub in list(self._subs):
            sub.close()
        if self._ros2 is not None:
            self._ros2.close()
            self._ros2 = None

    async def __aenter__(self) -> Engine:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    @property
    def drift_ppm(self) -> float | None:
        """Sink clock minus capture clock in ppm, once measured (or the profile's value)."""
        return None if self.reference is None else self.reference.drift_ppm

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

    def _ros2_runtime(self) -> Any:
        if self._ros2 is None:
            from ral.ros2 import Ros2Runtime

            self._ros2 = Ros2Runtime("ral_engine")
        return self._ros2

    def _open_source(self, cfg: CaptureConfig) -> AudioSource:
        if cfg.backend == "ros2_topic":
            from ral.devices.ros2_topic import Ros2TopicBackend

            assert cfg.topic is not None
            assert cfg.format is not None
            ros = Ros2TopicBackend(
                self._ros2_runtime(), decoder=cfg.decoder, reliability=cfg.reliability
            )
            fmt = AudioFormat(cfg.format.rate, cfg.format.channels, "int16")
            ros_source: AudioSource = ros.open_source(
                ros.device(cfg.topic, "input"), fmt, self._channel_map(cfg, fmt.channels)
            )
            return ros_source
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
        if cfg.backend == "ros2_topic":
            from ral.devices.ros2_topic import Ros2TopicBackend

            assert cfg.topic is not None
            ros = Ros2TopicBackend(
                self._ros2_runtime(), reliability=cfg.reliability, latency_ms=cfg.latency_ms or 0
            )
            f = cfg.format
            fmt = (
                AudioFormat(f.rate, f.channels, "int16")
                if f
                else AudioFormat(self.engine_fmt.sample_rate, 1, "int16")
            )
            ros_sink: AudioSink = ros.open_sink(ros.device(cfg.topic, "output"), fmt)
            return ros_sink
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

    def _check_hardware(self, source: AudioSource) -> None:
        # hardware mode (DESIGN 5): the device cancels echo against what it plays itself.
        if not source.channel_map.indices(ChannelRole.PROCESSED):
            raise ConfigError(
                "Echo mode hardware needs the device's echo-cancelled channel marked as "
                "`processed` in capture.channels. `ral doctor` shows the layout of known devices."
            )
        cap, play = self.profile.capture, self.profile.playback
        if cap.backend != play.backend or (cap.device or "") != (play.device or ""):
            log.warning(
                "Echo mode hardware: playback (%s %s) is not the capture device (%s %s); the "
                "device only cancels what it plays itself.",
                play.backend,
                play.device or "default",
                cap.backend,
                cap.device or "default",
            )

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

    # Health (DESIGN section 8)

    def _start_health(self, source: AudioSource) -> None:
        loop = asyncio.get_running_loop()

        def publish(event: Event) -> None:
            loop.call_soon(self.bus.publish, event)

        self.health = HealthMonitor(
            self.session_id,
            publish=publish,
            counters=self._xrun_counters,
            device_id=str(getattr(source, "clock_id", "") or self.profile.capture.backend),
            extra=self._health_extra,
        )
        health = self.health
        frames = self.subscribe(maxsize=400)

        async def observe() -> None:
            async for result in frames:
                health.observe(result.raw.data, result.raw.t_ns, result.raw.fmt.frame_ns)

        async def poll() -> None:
            while True:
                await asyncio.sleep(1.0)
                if not self.capture_done.is_set():  # a file that ended is not a lost device
                    health.poll()

        self._health_tasks = [
            asyncio.create_task(observe(), name="ral-health-observe"),
            asyncio.create_task(poll(), name="ral-health-poll"),
        ]

    def _xrun_counters(self) -> dict[str, int]:
        out: dict[str, int] = {"engine.dropped": self.input_overruns}
        framer = getattr(self.source, "framer", None)
        if framer is not None:
            out["capture.overrun"] = framer.ring.overruns
            out["capture.lost_frames"] = framer.lost_frames
        for name, obj, attr in (
            ("capture.overflow", self.source, "input_overflows"),
            ("playback.underflow", self.sink, "output_underflows"),
        ):
            value = getattr(obj, attr, None)
            if isinstance(value, int):
                out[name] = value
        if self.reference is not None:
            out["reference.overrun"] = self.reference.overruns
        return out

    def _health_extra(self) -> dict[str, float | None]:
        delay = next(
            (
                s.delay_ms
                for s in (self.graph.stages if self.graph else [])
                if hasattr(s, "delay_ms")
            ),
            None,
        )
        stats = {s.name: s.median_ms for s in self.latency.summary()}
        return {
            "drift_ppm": self.drift_ppm,
            "echo_delay_ms": delay,
            "lead_ms": float(self.lead.ms) if hasattr(self, "lead") else None,
            "dsp_us_per_frame": self.graph.us_per_frame if self.graph else None,
            "turn_latency_median_ms": stats.get("total"),
        }

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
        if self.recorder is not None:
            self.recorder.write_event(event)

    # Recording (opt-in; DESIGN section 8, "Privacy")

    def _start_recorder(self) -> None:
        from ral.record.mcap import McapRecorder

        cfg = self.profile.recording
        path = Path(cfg.path) / f"ral-{self.session_id}.mcap"
        self.recorder = McapRecorder(
            path,
            session_id=self.session_id,
            profile=self.profile.model_dump(mode="json"),
            raw=cfg.raw,
        )
        self.recorder.start()
        sub = self.subscribe(maxsize=1000)

        async def pump() -> None:
            async for result in sub:
                self.recorder.write_result(result.raw, result.clean, result.ctx.reference)

        self._record_task = asyncio.create_task(pump(), name="ral-recorder-pump")
        self.bus.publish(RecordingStarted(now_ns(), self.session_id, path=str(path)))
        log.info("Recording to %s", path)

    async def _stop_recorder(self) -> None:
        if self.recorder is None:
            return
        if self._record_task is not None:
            await asyncio.wait({self._record_task}, timeout=2.0)
            self._record_task.cancel()
            self._record_task = None
        self.recorder.write_event(
            RecordingStopped(now_ns(), self.session_id, path=str(self.recorder.path))
        )
        await asyncio.get_running_loop().run_in_executor(None, self.recorder.stop)
        if self.recorder.dropped:
            log.warning("Recorder dropped %d messages (disk too slow)", self.recorder.dropped)
