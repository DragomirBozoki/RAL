"""``ral_node``: run a RAL profile with the ROS 2 bridge.

    ros2 launch ral_ros robot.launch.py profile:=my-robot.yaml
    ros2 run ral_ros ral_node --profile my-robot.yaml --namespace /ral

The profile decides where audio comes from (``ros2_topic`` for a robot HAL, or a sound
card) and the node always adds the ROS 2 bridge (``<namespace>/clean``, ``events``,
``play``, ``control``; see ``ral.bridges.ros2``).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys


def main(argv: list[str] | None = None) -> None:
    from rclpy.utilities import remove_ros_args

    parser = argparse.ArgumentParser(prog="ral_node")
    parser.add_argument("--profile", default="laptop", help="Profile file or name.")
    parser.add_argument("--namespace", default="/ral", help="Topic namespace of the bridge.")
    parser.add_argument("--record", action="store_true", help="Record the session (MCAP).")
    args = parser.parse_args(remove_ros_args(argv if argv is not None else sys.argv)[1:])
    asyncio.run(_run(args.profile, args.namespace, args.record))


async def _run(profile: str, namespace: str, record: bool) -> None:
    from ral.bridges import start_bridges, stop_bridges
    from ral.engine.engine import Engine
    from ral.profiles import load_profile

    prof = load_profile(profile)
    prof.bridges.setdefault("ros2", {"namespace": namespace})
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):  # not on Windows
            loop.add_signal_handler(sig, stop.set)
    async with Engine(prof, record=record or None) as engine:
        bridges = await start_bridges(engine, prof.bridges)
        try:
            await stop.wait()
        finally:
            await stop_bridges(bridges)


if __name__ == "__main__":
    main()
