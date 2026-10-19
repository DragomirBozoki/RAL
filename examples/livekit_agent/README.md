# RAL with a LiveKit agent

RAL joins a LiveKit room as the robot: it publishes the echo-cancelled microphone and
plays the agent's audio through its playback service. The agent needs no changes; it
talks to the participant `robot` like to any user.

1. Put your LiveKit credentials in `.env` (copy `.env.example`): `LIVEKIT_URL`,
   `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and optionally `LIVEKIT_ROOM`.
2. Start your agent (LiveKit Agents or any other) so it joins the same room.
3. Run RAL on the laptop with a speaker and a microphone:

   ```bash
   uv sync --extra audio --extra dsp --extra livekit
   uv run ral models fetch silero_vad
   uv run ral run --profile examples/livekit_agent/laptop_livekit.yaml
   ```

To try without a cloud project, run a local server (`livekit-server --dev`, keys
`devkey` and `secret`) and set `LIVEKIT_URL=ws://127.0.0.1:7880`.

Events (speech, barge-in, playback) arrive at the agent as data messages on topic
`ral.events`; it can send `{"type": "interrupt"}` or `{"type": "duck", ...}` on topic
`ral.control`.
