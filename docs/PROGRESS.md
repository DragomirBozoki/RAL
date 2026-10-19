# Progress

## Current milestone: M1, echo-free conversation on a laptop (v0.1)

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
| 7 | Silero VAD | next |
| 8 | Barge-in and self-trigger metric | |
| 9 | `Engine` and `ral run` | |
| 10 | MCAP recorder | |
| 11 | WebSocket bridge | |
| 12 | LiveKit bridge | |
| 13 | Latency report and demo recording | |

### M1 done

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

### M1 decisions

- 2026-10-19 (step 1): no `samplerate` dependency. It has no CPython 3.10 wheel for Linux
  aarch64 in any release (checked on PyPI), which is the Jetson and Humble target, and the
  engine only needs fixed ratios. RAL has its own numpy polyphase resampler; DESIGN
  section 9 updated. Drift compensation (M3) revisits this.

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
- M1: should `AudioSource` gain a blocking `read(timeout)` for the DSP worker thread, or stay
  async-only (DESIGN section 4 vs section 9 threading model)?
- ROS 2 part of `ral doctor` can only be verified on a machine with ROS 2 (Ubuntu dual boot).
- Re-run `spikes/portaudio_time.py` on Linux (ALSA, PipeWire) and macOS to confirm the clock
  offset mapping there.
- M1: Silero VAD model distribution (bundle the MIT-licensed ONNX file, about 2 MB, or an
  explicit `ral models fetch`), and which real speech clips may go into `tests/data`.

