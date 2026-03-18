import argparse
import time

from online.config import DemoConfig, RobotEndpoint
from online.robot_io import RobotClient


SEND_HZ = 20.0


def _endpoint_for_name(config: DemoConfig, robot: str) -> RobotEndpoint:
    if robot == "chaser":
        return config.chaser_robot
    if robot == "evader":
        return config.evader_robot
    raise ValueError(f"Unknown robot '{robot}'")


def _send_for_duration(
    client: RobotClient, left: float, right: float, duration_s: float
) -> None:
    deadline = time.monotonic() + duration_s
    period_s = 1.0 / SEND_HZ
    while time.monotonic() < deadline:
        client.send(left, right)
        time.sleep(period_s)
    client.stop()
    time.sleep(0.25)


def run_sequence(
    endpoint: RobotEndpoint,
    power: float,
    duration_s: float,
    pause_s: float,
) -> None:
    steps = [
        ("forward", power, power),
        ("reverse", -power, -power),
        ("spin_left", -power, power),
        ("spin_right", power, -power),
        ("left_wheel_only", power, 0.0),
        ("right_wheel_only", 0.0, power),
    ]

    print(f"Connecting to {endpoint.name} at {endpoint.ip}:{endpoint.port}")
    print("Watch the robot and note what each step actually does.")
    print("Expected behavior:")
    print("  - forward: drives straight forward")
    print("  - reverse: drives straight backward")
    print("  - spin_left: rotates counterclockwise in place")
    print("  - spin_right: rotates clockwise in place")
    print("  - left_wheel_only/right_wheel_only: gentle arc")
    print()

    client = RobotClient(endpoint)
    try:
        client.stop()
        time.sleep(0.5)
        for index, (name, left, right) in enumerate(steps, start=1):
            print(
                f"[{index}/{len(steps)}] {name}: left={left:+.2f} right={right:+.2f} for {duration_s:.1f}s"
            )
            _send_for_duration(client, left, right, duration_s)
            print(f"Paused for {pause_s:.1f}s")
            time.sleep(pause_s)
        print("Done. Robot stopped.")
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed drive diagnostic sequence on a robot."
    )
    parser.add_argument(
        "robot",
        choices=("chaser", "evader"),
        help="Which robot to test.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.12,
        help="Wheel command magnitude in [-1, 1]. Default: 0.12",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="How long to run each motion step in seconds. Default: 1.0",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Pause between steps in seconds. Default: 1.0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DemoConfig()
    endpoint = _endpoint_for_name(config, args.robot)
    power = max(0.0, min(1.0, abs(args.power)))
    run_sequence(endpoint, power, args.duration, args.pause)


if __name__ == "__main__":
    main()
