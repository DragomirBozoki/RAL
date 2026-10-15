# Progress

## Current milestone: M0, Foundations

See `docs/DESIGN.md` section 10.

| Step | What | Status |
|---|---|---|
| 1 | Scaffolding, tooling, CI | done |
| 2 | Core types, errors, conversions | in review |
| 3 | RingBuffer and Framer | next |
| 4 | DeviceBackend protocol and FileBackend | |
| 5 | PortAudioBackend | |
| 6 | `ral doctor` v0 | |
| 7 | Spike: WebRTC APM standalone, ERLE on synthetic echo | |

## Done

- **Step 1.** `pyproject.toml` (hatchling, src layout, extras `audio`, `dsp`, `dev`),
  `ral --version`, LICENSE, NOTICE, README, CONTRIBUTING, ruff, mypy and pytest config
  (hardware marker skipped by default), pre-commit, GitHub Actions CI with a licence check.
- **Step 2.** `ral.core`: `AudioFormat`, `ChannelRole`, `ChannelMap`, `AudioFrame` (validated on
  construction), `DeviceInfo`, the `RalError` hierarchy (`FormatError`, `DeviceError`,
  `DeviceNotFoundError`, `MissingDependencyError`), int16 and float32 conversion.

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
- 2026-10-15: Errors that match a builtin also subclass it (`FormatError` is a `ValueError`,
  `MissingDependencyError` an `ImportError`), so generic handlers still catch them.

## Open questions

- M1: should `AudioSource` gain a blocking `read(timeout)` for the DSP worker thread, or stay
  async-only (DESIGN section 4 vs section 9 threading model)?
- ROS 2 part of `ral doctor` can only be verified on a machine with ROS 2 (Ubuntu dual boot).

## Next

Step 3: `RingBuffer` and `Framer` in `ral.core`.
