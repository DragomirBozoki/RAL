# Progress

## Current milestone: M2, robots and measurement (v0.2)

See `docs/DESIGN.md` section 10. Approved 2026-10-21: all steps in a row with commit, push
and CI after each; Pipecat as a separate package under `integrations/pipecat` (it needs
Python 3.11+ and pulls LGPL `soxr`); `pyroomacoustics` as a dev dependency with a marker (no
wheel for 3.10 on Linux aarch64; those tests skip there); ROS 2 written against the rclpy
docs and tested with fakes, to be run on Ubuntu by the maintainer. The order changed from
the M1 proposal so the room simulation can cover every echo mode.

| Step | What | Status |
|---|---|---|
| 1 | `loopback` echo mode | done |
| 2 | `hardware` echo mode and device database | done |
| 3 | Simulated rooms in CI for every echo mode | done |
| 4 | Replay of recordings and `ral bench` | done |
| 5 | `ral probe` | done (hardware run pending) |
| 6 | Profile migrations and registry | done |
| 7 | ROS 2 backend and `ral_ros` | next |
| 8 | Pipecat transport (`integrations/pipecat`) | |

### M2 done

- **Step 1.** `loopback` mode: `ApmProcessor(reference="channels", mic_delay_ms=...)` feeds
  the frame's own `reference` channels (mean, same clock) to the reverse stream before the
  mic, with an optional mic delay line whose length is passed as the stream delay and added
  to the lag. Profile `echo.mode: loopback`, `echo.mic_delay_ms`; `run_scene` takes a capture
  matrix with roles or `loopback_ms`. Synthetic ERLE 44 dB at 5, 20 and 120 ms echo delay,
  with or without a 5 ms mic delay.
- **Step 2.** `hardware` mode (`HardwareAecPassthrough`: processed channel out, mic and
  reference levels for barge-in; engine refuses it without a `processed` channel and warns
  when playback is on another device) and the device database `ral.devicedb`
  (`devices.yaml` in the package, pydantic-validated, matching by USB id, name and channel
  count; `ral doctor` reports known devices with echo mode and layout). Entries: ReSpeaker
  USB Mic Array v2.0 (XVF3000) 6- and 1-channel firmware, XMOS XVF3800; only documented
  facts, `validated_by: docs`.
- **Step 3.** `ral.bench.room` (pyroomacoustics shoebox: speaker under two mics, user at
  1.5 m, levels calibrated at the mic; captures for every mode) and `tests/bench/test_rooms.py`
  (12 tests). Numbers: self-trigger 36/min without echo control, 0 with `software_ref`,
  `loopback`, `hardware` and `gate`; ERLE 47 dB; barge-in at 12.0 s (`hardware`) and 13.1 to
  13.2 s (APM modes) for a user starting at 12.0 s, 12 dB below the echo. Changes found by
  the room: barge-in now counts passing frames (gap 200 ms) because AEC3 chops a quieter
  user; the gate attenuates 45 dB (was 30) and learns the echo as a peak-hold (was an
  average that lagged at utterance starts and caused false barge-ins).
- **Step 4.** `ral.record.replay` (`load_session`, `RecordedReference`, `replay`),
  `ral.engine.stages.build_stages` (shared by engine, replay and bench), `ral.bench.report`
  (`bench_rows`, `write_scene_recording`) and `ral bench` (`--echo`, `--config`, `--vad`,
  `--demo`, `--json`). Recordings now keep each channel's sample rate when read back.
  Demo: off 31 starts/min and 0.5 dB, `software_ref` 0 starts/min and 52.6 dB.
- **Step 5.** `ral.probe` (`signals`: exponential sweep and inverse filter; `analyze`: noise
  floor, roles, delay, gain, drift, mode suggestion; `io`: `EngineProbeIO` through the
  playback service and `SimulatedProbeIO`; `run`: profile YAML with a measurement header and
  an HTML report) and `ral probe` (`--input`, `--output`, `--level`, `--drift-s`, `--quiet`,
  `--yes`, `--simulate`). Simulated arrays are mapped correctly (mic, reference, processed,
  dead), delays within 0.2 ms, drift within 30 %. A `hardware` test runs it on real devices.
- Found in CI after step 3: `test_higher_priority_preempts_with_fade` failed on Windows (an
  underrun gap fooled the fade search; fixed in 21f90db).
- **Step 6.** `ral.profiles.migrations` (registered dict-to-dict steps, refusal of newer
  schemas; v1 is current, so the chain is empty and tested with a pretend history),
  `ral.profiles.registry` (path, `$RAL_PROFILES_PATH`, user folder, built-ins; user files
  override built-ins for roll-out and roll-back), `ral profile list|show|validate|migrate`,
  and the built-in `respeaker-usb-v2` profile from the device database (`hardware` mode).

### M2 decisions

- 2026-10-21 (step 5): the probe plays through the engine's playback service, like all
  RAL audio, so its delay is measured from the DAC time RAL itself estimates, which is the
  delay `software_ref` has to cover.
- 2026-10-21 (step 3): the simulated array's residual echo is -45 dB, an assumption until
  `ral probe` and `ral bench` measure a real one; at -35 dB the energy VAD self-triggers.
- 2026-10-21 (step 2): the device database lives in `src/ral/devicedb/` (not top-level
  `devicedb/`) so it ships with the package. The XVF3800's USB output channel layout and USB
  ids are not in the public datasheet, so the entry leaves them out; `ral probe` maps them.
- 2026-10-21 (step 1): in `loopback` the reference is the mean of the `reference` channels;
  stereo loopbacks of a mono speaker feed are the common case. The mic delay defaults to 0
  because the loopback already leads its acoustic echo by the speaker-to-mic path.

## M1, echo-free conversation on a laptop (v0.1): code complete, laptop demo pending

See `docs/DESIGN.md` section 10. Approved 2026-10-19: all 13 steps in a row (commit and
push after each), every new dependency in the plan, `websockets` for the WebSocket bridge,
and `ral models fetch` for the Silero model.

| Step | What | Status |
|---|---|---|
| 1 | Normalizer and resampler | done |
| 2 | `ReferenceBuffer` | done |
| 3 | Events and `EventBus` | done |
| 4 | Playback service | done |
| 5 | Echo controllers (`software_ref`, `gate`) | done |
| 6 | Noise suppression, high-pass, AGC; energy VAD | done |
| 7 | Silero VAD | done |
| 8 | Barge-in and self-trigger metric | done |
| 9 | `Engine` and `ral run` | done |
| 10 | MCAP recorder | done |
| 11 | WebSocket bridge | done |
| 12 | LiveKit bridge | done |
| 13 | Latency report and demo recording | done (demo recording waits for the maintainer) |

#### M1 done

- **Step 1.** `ral.dsp.resample.Resampler` (streaming polyphase FIR, time-aligned, about
  -85 dB error and -90 dB aliasing, about 100 us per 10 ms frame at 48 kHz) and
  `ral.dsp.normalizer.Normalizer` (float32, engine rate, `unused` channels dropped, output
  frames inherit input `t_ns` and `seq`, gaps flush the resampler).
- **Step 2.** `ReferenceProvider` protocol in `ral.core`; `ral.playback.ReferenceBuffer`
  (render hook copies into a Framer; DSP side reframes, downmixes to mono and resamples to
  the engine rate; `frames_until(t)` hands out frames by DAC time).
- **Step 3.** `ral.core.events` (15 frozen event types, JSON round trip, plugin subclasses
  found automatically) and `ral.core.bus.EventBus` (bounded per-subscriber queues that drop
  the oldest, type filters, sync listeners, `publish_threadsafe`).
- **Step 4.** `PlaybackHandle` and `PlaybackService` protocols; `ral.playback.SinkWriter`
  (resample, upmix, dtype, realtime pacing with `ahead_ms`, DAC-time cursor) and
  `ral.playback.PlaybackQueue` (priorities, preemption, raised-cosine fades, ducking,
  `interrupt`, `wait`, `speaking_at`, events at DAC time); `audio_frames()` helper.
- **Step 5.** `ral.core.processing` (`FrameContext` with `info`, `Processor`);
  `ral.dsp.echo` (`EchoOff`, `ApmProcessor` for `software_ref` with a continuous reverse
  stream and lag-corrected `t_ns`, `DuplexGate` for `gate` with a learned echo level);
  `ral.bench.synthetic` (echo scenes) and `ral.metrics.echo.erle_db`. CI test numbers
  (8 s scene, lead 20 ms): ERLE 44 / 50 / 49 / 37 dB at 5 / 20 / 50 / 120 ms echo delay;
  with the reference 30 ms late, 10 / 16 dB at 5 / 20 ms (the test requires below 15 at
  5 ms). Spike `spikes/apm_lag.py`: APM lag AEC 8 ms, NS 6 ms, additive.
- **Step 6.** `ral.dsp.vad` (`VadStateMachine` shared by all VAD engines, `EnergyVad` with
  an adaptive noise floor) and `ral.dsp.pipeline.build_capture_chain` (merges NS, high-pass
  and AGC into the AEC APM in `software_ref`, or runs them after the gate). APM noise
  suppression lowers stationary white noise by more than 10 dB in the test.
- **Step 7.** `ral.models` and `ral models list|fetch` (pinned URL and SHA-256, user
  cache, `$RAL_MODELS_DIR`); `ral.dsp.silero.SileroVad` (onnxruntime, 512-sample windows
  with 64 samples of context, one thread, about 120 us per 10 ms frame); real speech clip
  `tests/data/speech_16k.wav` (silero-vad test clip excerpt, MIT, with its script and
  licence). CI downloads the model before the tests. Spike `spikes/silero_io.py`.
- **Step 8.** `ral.engine.capture.CaptureGraph` (normalizer, reference by DAC time,
  speaking state, stages; no threads); `ral.dsp.bargein.BargeInDetector` (`off`, `energy`,
  `vad`, `vad+dtd` with learned leak, tail-aware echo prediction and the echo-leak guard);
  `ral.metrics.selftrigger`; `ral.bench.offline.run_scene`. CI scene: self-trigger 31/min
  without AEC, 4.4/min with `software_ref`; barge-in at 12.0 s for a user starting at
  12.0 s; plain VAD barge-in interrupts itself on echo, `vad+dtd` does not.
- **Fixes during step 8.** CI was red from M1 step 1 to 7 (mypy with numpy 2.2 on 3.10;
  fixed in c2fab32; I had not checked CI after each push, now I do). Synthetic echo tail
  normalised (it clipped the mic). Playback: `PlaybackEnded` uses the end of the last frame
  written; items start after a 30 ms jitter buffer; the writer lead covers four asyncio
  timer ticks (Windows before 3.13 has 15.6 ms ticks). The local check now runs 3.10 too.
- **Step 9.** `ral.profiles` (pydantic schema v1, YAML, `${VAR}` expansion, built-in
  `laptop` profile, errors with the path of every problem); `ral.engine.Engine` (devices from
  the profile, reference buffer on the sink's render hook, DSP worker thread around
  `CaptureGraph`, result subscriptions, barge-in fades playback); `ral run` (`--play`,
  `--json`, `--duration`); `load_wav`. Found on the way: offline capture lost frames (now
  waits instead of dropping, including the end marker); `PlaybackQueue.wait()` now returns
  at the DAC end time; a stalled producer ends at once when stopped; YAML `off` is a bool.
- **Step 10.** `ral.record.mcap` (`McapRecorder` with a writer thread, `foxglove.RawAudio`
  JSON channels `/ral/mic`, `/ral/reference`, `/ral/clean`, plus `/ral/events` and session,
  host and profile metadata; `read_recording`); engine and `ral run --record` (off by
  default, `RecordingStarted` and `RecordingStopped`). `record` extra: `mcap` (MIT, with lz4
  and zstandard, BSD). Spike `spikes/mcap_rawaudio.py`: bit-exact, 1.94 MB per minute per
  mono channel. Fixed a flaky M0 PortAudio test (fake clock taken before the loop).
- **Step 11.** `ral.bridges` (`Bridge` protocol, registry with the `ral.bridges` entry
  point, `start_bridges` and `stop_bridges`); `ral.bridges.websocket.WebSocketBridge`
  (protocol v1, localhost by default); `ral.playback.stream.PcmStream` (pushed PCM of any
  chunk size as one playback item); `ral run --bridge NAME`; `examples/websocket_client.py`.
  `server` extra: `websockets` (BSD-3).
- **Step 12.** `ral.bridges.livekit.LiveKitBridge` (clean mic track out, agent audio in as
  utterances through the playback service, events as data, control messages in, token
  minting) and `UtteranceSegmenter`; `livekit` extra adds `livekit-api` (Apache-2.0; its
  dependencies are MIT, BSD and Apache-2.0). Integration test against `livekit-server
  --dev` 1.13.10 passed locally (`RAL_TEST_LIVEKIT_URL`, skipped in CI). Spike
  `spikes/livekit_room.py`: 10 ms frames both ways, about 64 ms through a local room.
  Example profile and instructions in `examples/livekit_agent/`.
- **Step 13.** `ral.metrics.latency.LatencyTracker` and the `TurnLatency` event (total =
  hangover + agent + playback, from bus events only); the engine publishes one per turn and
  `ral run` prints median and p90 per stage at the end; `ral run --echo MODE` overrides the
  profile for A/B listening.

#### M1 acceptance (DESIGN section 10)

| Criterion | State |
|---|---|
| A LiveKit agent talks through `ral run` on a laptop with speaker and mic without answering itself | bridge tested against `livekit-server --dev` with a scripted agent; a real agent on the laptop waits for the maintainer |
| Synthetic-echo tests in CI show `software_ref` lowering the self-trigger rate against no AEC and report ERLE | yes: 31 vs 4.4 self-triggers per minute; ERLE 44 dB at 5 to 120 ms delay; a late reference fails as it must |
| Every turn produces a latency breakdown | yes (`TurnLatency`, tested end to end on the file backend) |
| A 2-minute demo recording exists | waits for the maintainer (commands below) |

#### Maintainer tasks (laptop with speaker and mic)

1. M0 hardware tests: `uv run pytest -m hardware`. Expect 3 passed (discovery, a short
   recording, a short tone).
2. Echo control without an agent, two runs, wearing no headphones:

   ```bash
   uv sync --all-extras
   uv run ral models fetch silero_vad
   uv run ral run --profile laptop --play tests/data/speech_16k.wav --duration 12 --echo off
   uv run ral run --profile laptop --play tests/data/speech_16k.wav --duration 12
   ```

   Expected: with `--echo off` the robot hears its own clip (`SpeechStarted` lines while it
   plays, possibly a `BargeIn` that cuts it off). With the default `software_ref` there
   should be no `SpeechStarted` while it plays unless you talk; talking over it gives one
   `BargeIn` and `PlaybackEnded ... interrupted=True`.
3. Demo with a LiveKit agent (see `examples/livekit_agent/README.md`), recorded:

   ```bash
   uv run ral run --profile examples/livekit_agent/laptop_livekit.yaml --record --duration 120
   ```

   The recording lands in `recordings/ral-<session>.mcap` (open it in Foxglove); the turn
   latency table prints at the end. Recordings are ignored by git.

#### M1 decisions

- 2026-10-19 (step 1): no `samplerate` dependency. It has no CPython 3.10 wheel for Linux
  aarch64 in any release (checked on PyPI), which is the Jetson and Humble target, and the
  engine only needs fixed ratios. RAL has its own numpy polyphase resampler; DESIGN
  section 9 updated. Drift compensation (M3) revisits this.
- 2026-10-19 (step 9): `AudioSource` stays async-only (closes the M0 open question): the
  engine bridges to the DSP thread with a bounded queue, lossy for devices, lossless for
  files.
- 2026-10-19 (step 9): an interrupt fades only audio already buffered; if the producer has
  stalled, the item ends at once (the sink is already playing silence).
- 2026-10-19 (step 7): Silero VAD v6.2.3 instead of v5 (latest release, same interface,
  MIT). Test speech comes from the silero-vad repository (MIT), which answers the open
  question about real speech clips for now.

## M0, Foundations (done 2026-10-19; CI green)

Two acceptance items wait for the maintainer: the PortAudio record and play hardware tests, and the ROS 2 part of `ral doctor`.

| Step | What | Status |
|---|---|---|
| 1 | Scaffolding, tooling, CI | done |
| 2 | Core types, errors, conversions | done |
| 3 | RingBuffer and Framer | done |
| 4 | DeviceBackend protocol and FileBackend | done |
| 5 | PortAudioBackend | done (hardware record and play tests to be run by the maintainer) |
| 6 | `ral doctor` v0 | done (ROS 2 part to be verified on Ubuntu) |
| 7 | Spike: WebRTC APM standalone, ERLE on synthetic echo | done |

### Done

- **Step 1.** `pyproject.toml` (hatchling, src layout, extras `audio`, `dsp`, `dev`),
  `ral --version`, LICENSE, NOTICE, README, CONTRIBUTING, ruff, mypy and pytest config
  (hardware marker skipped by default), pre-commit, GitHub Actions CI with a licence check.
- **Step 2.** `ral.core`: `AudioFormat`, `ChannelRole`, `ChannelMap`, `AudioFrame` (validated on
  construction), `DeviceInfo`, the `RalError` hierarchy (`FormatError`, `DeviceError`,
  `DeviceNotFoundError`, `MissingDependencyError`), int16 and float32 conversion.
- **Step 3.** `ral.core.RingBuffer` (preallocated, absolute sample indices, block anchors,
  `drop_oldest` and `drop_newest` overrun policies, overrun and underrun counters) and
  `ral.core.Framer` (exact 10 ms frames, `t_ns` from the sample count, jitter-tolerant
  re-anchoring, `seq` gaps for lost audio). Tests with a fake clock and a real producer thread.
- **Step 4.** `AudioSource`, `AudioSink`, `DeviceBackend` protocols (`ral.core.protocols`);
  `ral.core.clock.now_ns()`; `ral.devices.FileBackend` with `FileSource` and `FileSink`
  (16-bit PCM WAV, fast or realtime, render hooks, silence on underrun, `drop_newest`).
  Round trip bit-exact for int16 and float32 frames. Spike `spikes/clock_resolution.py`.
- **Step 5.** `ral.devices.portaudio`: `PortAudioBackend` (discovery with the preferred host
  API first, `default_device`, `find`, `native_format`), `PortAudioSource` (callback pushes
  into a `Framer`), `PortAudioSink` (`drop_newest` buffer, render hooks with DAC time),
  `PaClockMapper`. Fake-`sounddevice` tests plus three `hardware` tests. Spike
  `spikes/portaudio_time.py`.
- **Step 6.** `ral doctor` v0 (`ral.devices.doctor`, `ral.cli.doctor`): `HostScanner` (OS,
  Python, RAL clock, Linux `/dev/snd` access, PipeWire or PulseAudio), `PortAudioScanner`
  (library, host APIs, default devices, devices listed under several host APIs),
  `Ros2Scanner` (RMW, domain ID, localhost-only and discovery-server variables, audio-like
  topics with types and publisher QoS). Rich tables or `--json`; exit 1 on errors.
  Shared test fakes in `tests/fakes/` (pytest `pythonpath = ["tests"]`).
- **Step 7.** `spikes/apm_feasibility.py`: LiveKit APM standalone at 16 kHz / 10 ms on a
  synthetic scene. 36-42 dB ERLE for 5-120 ms echo delays with the reference on time; the
  stream-delay hint changes ERLE by under 1 dB; a reference 30 ms late drops ERLE to 7 dB
  (5 ms echo); APM output lags its input by 8 ms (14 ms with NS); 150-250 us per frame.
  Double talk costs the near-end talker 7-10 dB. Full table and meaning in the header.

### M0 acceptance (DESIGN section 10)

| Criterion | State |
|---|---|
| Installs with pip and uv on Linux, Windows and macOS | yes (CI run 39236835319, 2026-10-19: uv on all four, pip on Linux, Windows and macOS) |
| CI green on Linux x86_64, Linux arm64 and Windows | yes (same run, Python 3.10 and 3.12, plus macOS 3.12 and the licence check) |
| Framer tests pass for random block sizes | yes |
| `FileBackend` round trip is bit-exact | yes (int16 and float32, 1 to 6 channels) |
| On a laptop, `ral doctor` lists devices | yes, on the maintainer's Windows laptop |
| PortAudio hardware test records and plays | discovery test run; record and play tests waiting for the maintainer |
| With ROS 2 sourced, doctor lists audio-like topics with types and QoS | tested with a fake graph; needs a ROS 2 machine |
| The APM spike reports ERLE for several delays | yes |

### Decisions

- 2026-10-14: Distribution name `ral-audio` (`ral` on PyPI is taken by an empty project);
  import name and CLI stay `ral`.
- 2026-10-14: `DESIGN.md` moved to `docs/`.
- 2026-10-14: Core dependencies are added only when used. M0 core: numpy, typer, rich.
  samplerate (M1), pydantic and PyYAML (when profiles arrive) come later.
- 2026-10-14: `dev` is an optional extra (as in DESIGN section 9), so both
  `uv sync --all-extras` and `pip install -e ".[dev]"` work.
- 2026-10-14: CI adds a macOS job (3.12) because M0 acceptance requires installing on macOS.
- 2026-10-14: Copyright holder in NOTICE is "The RAL Authors".
- Agreed in step 0, to apply in later steps (each lands in DESIGN.md with its step):
  frames before the Normalizer are 10 ms in the device's native rate and dtype; sample rates
  must be divisible by 100; PortAudio opens devices at their native rate (WASAPI shared mode);
  render hooks must be real-time safe; a full sink buffer drops the frame and counts an
  overrun; the last partial WAV frame is zero-padded; PortAudio timestamp smoothing is M1.
- 2026-10-15: int16 and float32 conversion scales by 32768 both ways, rounds half to even,
  clips, and maps NaN to 0. Every int16 value round-trips bit-exact.
- 2026-10-15: `AudioFormat.dtype` is typed `Literal["int16", "float32"]` and checked at runtime;
  added `frame_ns` and `numpy_dtype` properties, and `ChannelMap.from_names`, `uniform`,
  `indices`. `DeviceInfo.extra` is left out of the hash so `DeviceInfo` stays hashable.
- 2026-10-15 (step 3): `RingBuffer` uses a short `threading.Lock`, because `drop_oldest`
  lets the producer move the read index. Anchors live in a fixed FIFO (default 1024).
- 2026-10-15 (step 3): `t_ns` uses integer nanoseconds; frames sit on a grid of the absolute
  sample index and `seq` is the grid index plus frames skipped for forward time jumps of at
  least one frame (nearest whole frame, ties down). Re-anchor after 3 consecutive
  observations beyond 2 ms, or at once after a gap. Underruns count only partial reads
  (`read_available`, for sinks); a Framer waiting for data is not an underrun.
- 2026-10-15 (step 4): all timestamps come from `ral.core.clock.now_ns()`; on Windows before
  Python 3.13 it is `perf_counter_ns` because `monotonic` has 15.6 ms steps there (spike).
  CLAUDE.md rule 5 and DESIGN section 5 updated. Option B (keep `monotonic_ns`) rejected:
  Windows timing numbers would be meaningless.
- 2026-10-15 (step 4): `FileBackend` takes `{name: path}` maps for sources and sinks; device
  ids are `file:in:<name>` and `file:out:<name>`. Source `fmt` must match the file's rate and
  channels; `dtype` may be `int16` (unchanged) or `float32` (converted). Realtime sources
  deliver a frame once its last sample is due; realtime sinks render every 10 ms from a
  `drop_newest` ring buffer. Render hooks get samples in the sink's dtype.
- 2026-10-19 (step 6): doctor levels are ok/info/warn/error, shown as green/amber/red per
  area; `info` never changes an area's colour. When `rclpy` does not import, the ROS 2
  area is listed as skipped, not as a finding. The device table shows the preferred host
  API only unless `--all-devices`. Echo-mode suggestion waits for the device database (M2).
- 2026-10-19 (step 6): the rclpy graph code (`RclpyGraph`) is written from the rclpy docs but
  not run yet; it uses its own `Context` so it never touches an application's rclpy state.
- 2026-10-19 (step 5): PortAudio time is mapped through a measured offset (lower envelope
  of `now_ns() - currentTime`, rising by 1/128 per callback), not assumed equal to the RAL
  clock. ADC/DAC times that are 0 or equal to `currentTime` fall back to `currentTime` -/+
  stream latency. Streams use `blocksize=0` and `latency="low"`; `frames()` polls the Framer
  every 5 ms because waking the event loop from a callback is not real-time safe.
- 2026-10-19 (step 5): PortAudio input overflow and output underflow flags are counted
  (`input_overflows`, `output_underflows`); the sink's `buffer.underruns` also counts idle
  silence until the M1 playback service can tell idle from starved.
- 2026-10-19 (step 7): default `lead` is 20 ms and the stream-delay hint stays `lead`
  (DESIGN section 5). The echo controller must shift clean-frame `t_ns` by the APM's output
  lag (8 ms with AEC at 16 kHz, 14 ms with NS); DESIGN section 5 updated.
- 2026-10-15: Python 3.10 tests run locally in a separate env
  (`UV_PROJECT_ENVIRONMENT=.venv310 uv run --python 3.10 --all-extras pytest`), so the main
  `.venv` stays on 3.12. `.venv*/` is ignored.
- 2026-10-15: Errors that match a builtin also subclass it (`FormatError` is a `ValueError`,
  `MissingDependencyError` an `ImportError`), so generic handlers still catch them.

### Open questions

- M1: if the timeline re-anchors backwards by a frame or more, `t_ns` can go backwards. Rare
  (a very late first callback), and the M1 delay-locked loop should remove it; until then
  it is counted in `reanchors` only.
- Watch: CI failed once on `test (ubuntu-24.04-arm, py3.12)` for the LiveKit commit
  (28b44b1) and passed on the next commit; logs need a login, so failing tests are now also
  printed as `::error` annotations, which the public run page shows.
- Watch: one local test run hung once in step 9 and could not be reproduced in 6 more
  runs. `faulthandler_timeout = 120` dumps every thread if it happens again; CI jobs time
  out after 15 minutes.
- ROS 2 part of `ral doctor` can only be verified on a machine with ROS 2 (Ubuntu dual boot).
- Re-run `spikes/portaudio_time.py` on Linux (ALSA, PipeWire) and macOS to confirm the clock
  offset mapping there.
- M1: Silero VAD model distribution (bundle the MIT-licensed ONNX file, about 2 MB, or an
  explicit `ral models fetch`), and which real speech clips may go into `tests/data`.

