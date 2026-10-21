"""A stand-in robot HAL: publishes a WAV file as audio_common_msgs/AudioDataStamped.

    source /opt/ros/humble/setup.bash
    python examples/ros2/fake_hal.py tests/data/speech_16k.wav [--topic /hal/audio/capture]

Publishes 20 ms chunks in real time, each stamped with the capture time of its first
sample, and loops the file.
"""

from __future__ import annotations

import argparse
import array
import time
import wave

import rclpy
from audio_common_msgs.msg import AudioDataStamped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    parser.add_argument("--topic", default="/hal/audio/capture")
    args = parser.parse_args()
    with wave.open(args.wav, "rb") as w:
        rate, pcm = w.getframerate(), w.readframes(w.getnframes())
    chunk = rate // 50 * 2  # 20 ms of 16-bit mono
    rclpy.init()
    node = rclpy.create_node("fake_hal")
    pub = node.create_publisher(AudioDataStamped, args.topic, 50)
    t0 = time.monotonic()
    ros0 = node.get_clock().now().nanoseconds
    sent = 0
    try:
        while rclpy.ok():
            for i in range(0, len(pcm) - chunk + 1, chunk):
                stamp = ros0 + sent * 1_000_000_000 // rate
                msg = AudioDataStamped()
                msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
                msg.audio.data = array.array("B", pcm[i : i + chunk])
                pub.publish(msg)
                sent += chunk // 2
                delay = t0 + sent / rate - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
