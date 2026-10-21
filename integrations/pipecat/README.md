# ral-pipecat

A [Pipecat](https://github.com/pipecat-ai/pipecat) transport for robots running RAL. Swap
your bot's local audio transport for `RalTransport`: the bot hears the robot's
echo-cancelled microphone and speaks through RAL's playback service, so the robot does
not answer itself and a user can interrupt it.

```python
from ral_pipecat import RalTransport, RalTransportParams

transport = RalTransport(RalTransportParams(url="ws://127.0.0.1:8765"))
pipeline = Pipeline([transport.input(), stt, llm, tts, transport.output()])

@transport.event_handler("on_ral_event")
async def on_ral_event(transport, event):
    print(event["type"])          # SpeechStarted, BargeIn, PlaybackStarted, ...
```

Start RAL with its WebSocket bridge on the robot (or laptop) first:

```bash
ral run --profile laptop --bridge websocket
```

- Input: RAL's clean 16 kHz mono frames as `InputAudioRawFrame`; Pipecat's VAD and turn
  logic still run. A barge-in detected by RAL interrupts the pipeline.
- Output: the bot's audio at any sample rate (RAL resamples), sent up to 100 ms ahead of
  real time; an interruption from Pipecat fades RAL's playback out.

Separate from `ral-audio` because pipecat-ai needs Python 3.11+ (RAL supports 3.10 for
ROS 2 Humble) and depends on `soxr`, which is LGPL-2.1. Apache-2.0.

## Development

```bash
cd integrations/pipecat
uv sync --python 3.12 --extra dev
uv run pytest
```
