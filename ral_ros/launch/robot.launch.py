"""ros2 launch ral_ros robot.launch.py profile:=my-robot.yaml [namespace:=/ral] [record:=false]"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    profile = LaunchConfiguration("profile")
    namespace = LaunchConfiguration("namespace")
    record = LaunchConfiguration("record")
    common = {"package": "ral_ros", "executable": "ral_node", "name": "ral", "output": "screen"}
    return LaunchDescription(
        [
            DeclareLaunchArgument("profile", default_value="laptop"),
            DeclareLaunchArgument("namespace", default_value="/ral"),
            DeclareLaunchArgument("record", default_value="false"),
            Node(
                **common,
                arguments=["--profile", profile, "--namespace", namespace],
                condition=UnlessCondition(record),
            ),
            Node(
                **common,
                arguments=["--profile", profile, "--namespace", namespace, "--record"],
                condition=IfCondition(record),
            ),
        ]
    )
