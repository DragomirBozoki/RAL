# ral_interfaces

ROS 2 actions and services served by RAL's ROS 2 bridge (`bridges: {ros2: {}}`).

| Name | Kind | What it does |
|---|---|---|
| `/ral/play_audio` | `ral_interfaces/action/PlayAudio` | Plays 16-bit mono PCM through RAL's playback service; feedback and result report the seconds played |
| `/ral/say` | `ral_interfaces/action/Say` | Speaks text with the TextToSpeech adapter given to the bridge; goals are rejected without one |
| `/ral/stop` | `ral_interfaces/srv/Stop` | Fades out the playing item and drops the queue (`all: true` includes non-interruptible items) |
| `/ral/duck` | `ral_interfaces/srv/Duck` | Lowers the output by `gain_db` for `duration_ms` |

Cancelling a goal fades its audio out. Build it in a colcon workspace:

```bash
mkdir -p ~/ral_ws/src && cp -r integrations/ros2/ral_interfaces ~/ral_ws/src/
cd ~/ral_ws && source /opt/ros/humble/setup.bash && colcon build --packages-select ral_interfaces
source install/setup.bash
ros2 interface show ral_interfaces/action/PlayAudio
```

The bridge finds the package when the workspace is sourced; without it the topics
(`play`, `control`) still carry the same requests.
