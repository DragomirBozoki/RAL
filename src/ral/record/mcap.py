"""Session recording to MCAP, readable in Foxglove (DESIGN section 8).

Recording is off by default (robots record people) and is switched on per profile
(``recording.enabled``) or per run (``ral run --record``). It publishes
``RecordingStarted`` and ``RecordingStopped`` and writes only to a local file.

Channels (log time = RAL clock, ``ral.core.clock.now_ns``):

- ``/ral/mic``: the normalized capture frame, every channel, before echo control.
- ``/ral/reference``: the reference frames fed to echo control for each mic frame.
- ``/ral/clean``: the clean stream after echo control, noise suppression and gain.
- ``/ral/events``: every event as JSON (``{"type": ..., ...}``).

Audio uses Foxglove's ``foxglove.RawAudio`` JSON schema (``pcm-s16``, base64 data), so a
recording plays in Foxglove without plugins (``spikes/mcap_rawaudio.py``: bit-exact,
about 2 MB per minute per mono channel). Channel roles are stored in the channel
metadata. Metadata records hold the profile, host information and the mapping from the
RAL clock to wall time at the start (only for humans; the audio path never uses it).

File writes happen on a dedicated thread, so a slow disk never stalls the event loop;
if the disk falls far behind, the oldest pending audio is dropped and counted.
"""

from __future__ import annotations

import base64
import json
import logging
import platform
import queue
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ral import __version__
from ral.core.clock import CLOCK_NAME, now_ns
from ral.core.convert import float32_to_int16
from ral.core.errors import FormatError, MissingDependencyError
from ral.core.events import Event, event_from_json, event_to_json
from ral.core.types import AudioFrame

__all__ = ["RAW_AUDIO_SCHEMA", "McapRecorder", "Recording", "read_recording"]

log = logging.getLogger(__name__)

# foxglove.RawAudio, JSON schema from https://github.com/foxglove/foxglove-sdk (MIT).
RAW_AUDIO_SCHEMA: dict[str, Any] = {
    "title": "foxglove.RawAudio",
    "description": "A single block of an audio bitstream",
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "title": "time",
            "properties": {
                "sec": {"type": "integer", "minimum": 0},
                "nsec": {"type": "integer", "minimum": 0, "maximum": 999999999},
            },
        },
        "data": {"type": "string", "contentEncoding": "base64"},
        "format": {"type": "string"},
        "sample_rate": {"type": "integer", "minimum": 0},
        "number_of_channels": {"type": "integer", "minimum": 0},
    },
    "required": ["timestamp", "data", "format", "sample_rate", "number_of_channels"],
}
EVENT_SCHEMA: dict[str, Any] = {
    "title": "ral.Event",
    "type": "object",
    "properties": {"type": {"type": "string"}, "t_ns": {"type": "integer"}},
    "required": ["type", "t_ns"],
}
TOPICS = ("/ral/mic", "/ral/reference", "/ral/clean")
_QUEUE_ITEMS = 3000  # about 10 s of all three audio channels


def _mcap() -> Any:
    try:
        from mcap import writer
    except ImportError:
        raise MissingDependencyError("mcap", "record", "Session recording") from None
    return writer


def _raw_audio(frame: AudioFrame) -> bytes:
    data = frame.data
    pcm = float32_to_int16(data) if data.dtype != np.int16 else data
    t = max(0, frame.t_ns)
    return json.dumps(
        {
            "timestamp": {"sec": t // 1_000_000_000, "nsec": t % 1_000_000_000},
            "data": base64.b64encode(np.ascontiguousarray(pcm, "<i2").tobytes()).decode(),
            "format": "pcm-s16",
            "sample_rate": frame.fmt.sample_rate,
            "number_of_channels": frame.fmt.channels,
        },
        separators=(",", ":"),
    ).encode()


class McapRecorder:
    """Writes one session to an MCAP file.

    Args:
        path: The ``.mcap`` file to create.
        session_id: Stored in the metadata.
        profile: The active profile as a plain dict, stored in the metadata.
        raw: Also store ``/ral/mic`` and ``/ral/reference`` (not only clean audio and events).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        session_id: str = "",
        profile: dict[str, Any] | None = None,
        raw: bool = True,
    ) -> None:
        self.path = Path(path)
        self.session_id = session_id
        self.profile = profile or {}
        self.raw = raw
        self._queue: queue.Queue[tuple[str, int, bytes, dict[str, str]] | None] = queue.Queue(
            _QUEUE_ITEMS
        )
        self._thread: threading.Thread | None = None
        self.dropped = 0
        """Messages dropped because the disk fell behind."""
        self.messages = 0
        self.error: BaseException | None = None

    def start(self) -> None:
        """Create the file and start the writer thread."""
        writer_mod = _mcap()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("wb")
        writer = writer_mod.Writer(stream)
        writer.start(profile="ral", library=f"ral-audio/{__version__}")
        writer.add_metadata(
            "ral.session",
            {
                "session_id": self.session_id,
                "ral_version": __version__,
                "clock": CLOCK_NAME,
                "clock_ns_at_start": str(now_ns()),
                "wall_time_at_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "host_os": platform.platform(),
                "python": sys.version.split()[0],
            },
        )
        writer.add_metadata("ral.profile", {"yaml_json": json.dumps(self.profile, default=str)})
        self._thread = threading.Thread(
            target=self._run, args=(writer, stream), name="ral-recorder", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Flush everything and close the file."""
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout=10.0)
        self._thread = None

    # Called on the event loop; never blocks.

    def write_result(self, raw: AudioFrame, clean: AudioFrame, reference: list[AudioFrame]) -> None:
        """Record one processed frame (its raw input, reference and clean output)."""
        if self.raw:
            self._offer("/ral/mic", raw.t_ns, _raw_audio(raw), raw)
            for ref in reference:
                self._offer("/ral/reference", ref.t_ns, _raw_audio(ref), ref)
        self._offer("/ral/clean", clean.t_ns, _raw_audio(clean), clean)

    def write_event(self, event: Event) -> None:
        """Record one event."""
        data = json.dumps(event_to_json(event), separators=(",", ":")).encode()
        self._put(("/ral/events", event.t_ns, data, {}))

    def _offer(self, topic: str, t_ns: int, data: bytes, frame: AudioFrame) -> None:
        roles = ",".join(r.value for r in frame.channel_map.roles)
        self._put((topic, t_ns, data, {"roles": roles, "clock_id": frame.clock_id}))

    def _put(self, item: tuple[str, int, bytes, dict[str, str]]) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    # Writer thread.

    def _run(self, writer: Any, stream: Any) -> None:
        channels: dict[str, int] = {}
        audio = writer.register_schema(
            "foxglove.RawAudio", "jsonschema", json.dumps(RAW_AUDIO_SCHEMA).encode()
        )
        events = writer.register_schema(
            "ral.Event", "jsonschema", json.dumps(EVENT_SCHEMA).encode()
        )
        try:
            while (item := self._queue.get()) is not None:
                topic, t_ns, data, meta = item
                if topic not in channels:
                    schema = events if topic == "/ral/events" else audio
                    channels[topic] = writer.register_channel(topic, "json", schema, meta)
                writer.add_message(channels[topic], max(0, t_ns), data, max(0, t_ns))
                self.messages += 1
        except BaseException as exc:
            self.error = exc
            log.exception("Recorder failed; recording stopped")
        finally:
            writer.finish()
            stream.close()


@dataclass
class Recording:
    """A session read back from MCAP: audio per topic and the events."""

    audio: dict[str, list[tuple[int, npt.NDArray[np.int16]]]] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    metadata: dict[str, dict[str, str]] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)

    def signal(self, topic: str) -> npt.NDArray[np.int16]:
        """All blocks of one audio topic concatenated, ``(samples, channels)``."""
        blocks = [b for _, b in self.audio.get(topic, [])]
        if not blocks:
            return np.zeros((0, 1), np.int16)
        return np.concatenate(blocks)


def _iter(path: Path) -> Iterator[tuple[Any, Any, Any]]:
    try:
        from mcap.reader import make_reader
    except ImportError:
        raise MissingDependencyError("mcap", "record", "Reading recordings") from None
    with path.open("rb") as f:
        reader = make_reader(f)
        yield from ((None, {"metadata": m}, None) for m in reader.iter_metadata())
        yield from reader.iter_messages()


def read_recording(path: str | Path) -> Recording:
    """Load a recording made by :class:`McapRecorder`.

    Raises:
        FormatError: If a message is not what RAL writes.
    """
    rec = Recording()
    for _schema, channel, message in _iter(Path(path)):
        if message is None:
            meta = channel["metadata"]
            rec.metadata[meta.name] = dict(meta.metadata)
            continue
        payload = json.loads(message.data)
        if channel.topic == "/ral/events":
            rec.events.append(event_from_json(payload))
            continue
        if payload.get("format") != "pcm-s16":
            raise FormatError(
                f"Unexpected audio format {payload.get('format')!r} on {channel.topic}."
            )
        pcm = np.frombuffer(base64.b64decode(payload["data"]), "<i2").astype(np.int16)
        block = pcm.reshape(-1, int(payload["number_of_channels"]))
        rec.audio.setdefault(channel.topic, []).append((message.log_time, block))
        rec.roles[channel.topic] = channel.metadata.get("roles", "")
    return rec
