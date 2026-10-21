# ral_ros

RAL as a ROS 2 node (Humble, Jazzy). Not tested on a ROS 2 machine yet; see
`docs/PROGRESS.md` for the commands to verify it.

```bash
source /opt/ros/humble/setup.bash
sudo apt install ros-humble-audio-common-msgs
pip install "ral-audio[dsp]"          # into the Python ROS 2 uses
colcon build --packages-select ral_ros && source install/setup.bash
ros2 launch ral_ros robot.launch.py profile:=my-robot.yaml
```

Topics (namespace `/ral` by default): `clean` and `clean_info` (clean audio and its
format), `events` (JSON), `play` (audio to speak, 16-bit mono at the engine rate) and
`control` (JSON: `interrupt`, `stop`, `duck`, `end`). A robot whose HAL publishes audio
uses a profile with `capture.backend: ros2_topic`.
