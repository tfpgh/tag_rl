from __future__ import annotations

import argparse

import uvicorn

from online.game.config import GameRuntimeConfig
from online.game.dashboard import build_dashboard_app
from online.game.runtime import TagGameRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live two-robot tag game")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--chaser-ip", default="192.168.1.5")
    parser.add_argument("--evader-ip", default="192.168.1.6")
    parser.add_argument("--udp-port", type=int, default=8888)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-udp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GameRuntimeConfig(checkpoint_path=args.checkpoint)
    config.chaser.robot_ip = args.chaser_ip
    config.evader.robot_ip = args.evader_ip
    config.chaser.udp_port = args.udp_port
    config.evader.udp_port = args.udp_port
    config.tracker.arena.chaser_tag_id = config.chaser.tag_id
    config.tracker.arena.evader_tag_id = config.evader.tag_id
    config.tracker.teleop.robot_ip = config.chaser.robot_ip
    config.tracker.teleop.udp_port = config.chaser.udp_port
    config.tracker.teleop.robot_tag_id = config.chaser.tag_id
    config.dashboard_host = args.host
    config.dashboard_port = args.port

    runtime = TagGameRuntime(config, udp_enabled=not args.no_udp)
    runtime.start()
    app = build_dashboard_app(runtime)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
