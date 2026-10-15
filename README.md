# RAL: Robot Audio Layer

> Working name. Status: **pre-alpha**. Nothing here is usable yet.

RAL is an open-source, vendor-neutral audio layer for robots. It sits between a robot's
microphones and speakers and whatever speech stack you use. It finds audio devices, maps
microphone and reference channels, removes the robot's own voice (echo control), plays audio
with priorities and clean interruption, and connects to LiveKit, Pipecat, ROS 2 and speech
models through adapters and sockets. Sessions can be recorded and replayed, so echo and latency
become numbers.

RAL handles the robot side and leaves every AI choice to you.

- Architecture: [docs/DESIGN.md](docs/DESIGN.md)
- Current milestone and progress: [docs/PROGRESS.md](docs/PROGRESS.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)

## Install (development)

```bash
uv sync --all-extras
uv run ral --version
```

`pip install -e ".[dev]"` works too. The `audio` extra uses PortAudio; on Debian and Ubuntu
install the system library first: `sudo apt install libportaudio2`.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
