# Progress

## Current milestone: M0, Foundations

See `docs/DESIGN.md` section 10.

| Step | What | Status |
|---|---|---|
| 1 | Scaffolding, tooling, CI | done |
| 2 | Core types, errors, conversions | done |
| 3 | RingBuffer and Framer | done |
| 4 | DeviceBackend protocol and FileBackend | done |
| 5 | PortAudioBackend | done (hardware record and play tests to be run by the maintainer) |
| 6 | `ral doctor` v0 | next |
| 7 | Spike: WebRTC APM standalone, ERLE on synthetic echo | |

## Done

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

## Decisions

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
- 2026-10-19 (step 5): PortAudio time is mapped through a measured offset (lower envelope
  of `now_ns() - currentTime`, rising by 1/128 per callback), not assumed equal to the RAL
  clock. ADC/DAC times that are 0 or equal to `currentTime` fall back to `currentTime` -/+
  stream latency. Streams use `blocksize=0` and `latency="low"`; `frames()` polls the Framer
  every 5 ms because waking the event loop from a callback is not real-time safe.
- 2026-10-19 (step 5): PortAudio input overflow and output underflow flags are counted
  (`input_overflows`, `output_underflows`); the sink's `buffer.underruns` also counts idle
  silence until the M1 playback service can tell idle from starved.
- 2026-10-15: Python 3.10 tests run locally in a separate env
  (`UV_PROJECT_ENVIRONMENT=.venv310 uv run --python 3.10 --all-extras pytest`), so the main
  `.venv` stays on 3.12. `.venv*/` is ignored.
- 2026-10-15: Errors that match a builtin also subclass it (`FormatError` is a `ValueError`,
  `MissingDependencyError` an `ImportError`), so generic handlers still catch them.

## Open questions

- M1: if the timeline re-anchors backwards by a frame or more, `t_ns` can go backwards. Rare
  (a very late first callback), and the M1 delay-locked loop should remove it; until then
  it is counted in `reanchors` only.
- M1: should `AudioSource` gain a blocking `read(timeout)` for the DSP worker thread, or stay
  async-only (DESIGN section 4 vs section 9 threading model)?
- ROS 2 part of `ral doctor` can only be verified on a machine with ROS 2 (Ubuntu dual boot).

## Next

Step 6: `ral doctor` v0.
