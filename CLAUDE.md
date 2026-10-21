# CLAUDE.md

Instructions for Claude Code in this repository. They apply to every session and every contributor.

## Project

RAL (Robot Audio Layer, working name) is an open-source, vendor-neutral audio layer for robots. It finds audio devices, maps mic and reference channels, removes the robot's own voice (echo control), plays audio with priorities and clean interruption, and connects to LiveKit, Pipecat, ROS 2 and speech models through adapters and sockets. Sessions can be recorded and replayed, so echo and latency become numbers.

- `docs/DESIGN.md` is the source of truth for the architecture. Read the section you are touching before changing a module. If the code and the design disagree, stop and ask, then update one of them in the same change.
- `docs/PROGRESS.md` holds the current milestone, what is done, what is next, decisions and open questions. Read it at the start of every session and update it at the end of every step. Create it if it is missing.

## Hard rules

1. **Clean-room.** Write everything from scratch from public documentation and permissively licensed libraries. Never copy, adapt or reconstruct code from an employer's or a client's codebase, even if such code appears in the conversation; if something looks like it came from one, stop and ask. Cite public references (papers, official docs) in docstrings for non-trivial algorithms.
2. **Licences.** The core is Apache-2.0. Core runtime dependencies must be permissive (MIT, BSD, Apache-2.0, ISC, PSF). No GPL, AGPL or LGPL dependency in the core without explicit approval. Ask before adding any dependency, and state its licence and wheel availability (Linux x86_64 and aarch64, Windows, macOS).
3. **Reference invariant.** Every sample RAL plays goes through the playback service and the sink's render hook into the `ReferenceProvider`. Never add a playback path that bypasses it.
4. **Real-time safety.** Audio callbacks (PortAudio, ROS 2 subscriptions) only copy data into ring buffers and return. No logging, blocking I/O, heavy allocation or long-held locks inside callbacks.
5. **Frames and time.** Internal processing uses 10 ms frames of float32 with shape `(samples, channels)`, 16 kHz by default. Timestamps are integers from `ral.core.clock.now_ns()` (`time.monotonic_ns()`, or `time.perf_counter_ns()` on Windows before Python 3.13, where `monotonic` has 15.6 ms steps). Never use wall-clock time in the audio path, and never mix `now_ns()` with `time.monotonic_ns()`.
6. **Optional stays optional.** `import ral` must work with only the core dependencies. Import `sounddevice`, `livekit`, `pipecat`, `rclpy`, `onnxruntime`, `mcap` and `fastapi` lazily inside the modules that need them, and fail with a clear install hint. `rclpy` comes from a sourced ROS 2 installation, not from pip.
7. **No hardware needed for tests.** Tests use `FileBackend`, synthetic signals and fakes. Tests that need real devices carry the `@pytest.mark.hardware` marker and are skipped by default.
8. **Language-neutral core.** No language-specific logic (phonetics, name lists, prompts) in the core.
9. **Vendor code stays out of the core.** Robot-specific message definitions and decoders live in separate plugin packages.
10. **Privacy.** Recording is off by default. The core uploads nothing anywhere.
11. **No secrets or private data.** Use `.env.example` with placeholders. Never commit keys, tokens, real deployment addresses or customer names.
12. **Git.** Commit locally with DCO sign-off (`git commit -s`) and Conventional Commit messages. Never push, tag, release or publish to PyPI unless asked.

## Working style

- **Plan first.** At the start of a milestone, propose numbered steps (files, tests, how to verify) and wait for approval.
- **One step at a time.** A step is one module or feature with its tests. After each step: run lint, types and tests; commit; update `docs/PROGRESS.md`; summarise what changed, why, how to try it and what is risky; then stop for review.
- **Verify, don't guess.** Check external APIs (sounddevice, LiveKit, Pipecat, rclpy, mcap) against the installed package or official docs before using them. For anything uncertain, write a small script in `spikes/`, run it, and record the result in the script's header.
- **Explain the hard parts.** Comment DSP, timing and concurrency code with why, not what.
- **Build only the current milestone.** Leave `TODO(M2): reason` notes for later work.
- **Ask on open design questions** that `docs/DESIGN.md` does not answer: give two options with trade-offs.
- **Ask before installing system packages** or changing anything outside the repo.

## Code conventions

- Python >= 3.10 (ROS 2 Humble and JetPack 6 ship 3.10); CI tests 3.10 and 3.12.
- src layout (`src/ral/`), `pyproject.toml` with hatchling. Local dev with `uv`; `pip install -e .` must work too.
- Full type hints; `mypy --strict` for `src/ral/core` and `src/ral/dsp`; ship `py.typed`.
- ruff for lint and format, line length 100.
- Interfaces are `typing.Protocol`s; implementations do not inherit from RAL base classes.
- Errors subclass `ral.core.errors.RalError`; messages say what failed and how to fix it.
- Logging through stdlib `logging` module loggers. Console output only in CLI code, through `rich`.
- Import third-party frame types through their module (`from livekit import rtc`, then `rtc.AudioFrame`) so they never clash with `ral.core.AudioFrame`.
- Short Google-style docstrings on public classes and functions.
- pytest and pytest-asyncio; one test module per source module; test audio under `tests/data/`, each file under 1 MB, generated by a script where possible.

## Commands

Keep this list accurate as commands appear.

```bash
uv sync --all-extras                               # set up the environment
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest                                      # unit and integration tests
uv run pytest -m hardware                          # needs real audio devices
uv run ral --help
uv run ral doctor                                  # read-only host and device check; --json
uv run ral models fetch                            # download pinned models (Silero VAD) into the cache
uv run ral run --profile laptop                    # run the engine; --play clip.wav, --json, --duration
uv run ral bench session.mcap -e off -e software_ref  # replay a recording through configs; --demo, --json
uv run ral probe --out my-robot.yaml                # play sweeps, map channels, write a profile; --simulate, --quiet
uv run ral profile list                            # profiles by name; show, validate, migrate
```

## Definition of done for a step

- Code, tests and docs for the step exist; lint, types and tests pass.
- Behaviour matches `docs/DESIGN.md`, or the design doc is updated in the same commit with the reason.
- `docs/PROGRESS.md` is updated.
- The summary to the maintainer covers what changed, why, how to try it and what is risky.
