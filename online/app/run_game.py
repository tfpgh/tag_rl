from __future__ import annotations

import os
from dataclasses import replace


def main() -> None:
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import uvicorn

    from online.game.config import GameRuntimeConfig
    from online.game.dashboard import build_dashboard_app
    from online.game.runtime import TagGameRuntime

    config = GameRuntimeConfig()
    config.tracker.arena = replace(
        config.tracker.arena,
        chaser_tag_id=config.chaser.tag_id,
        evader_tag_id=config.evader.tag_id,
    )
    config.tracker.teleop.robot_ip = config.chaser.robot_ip
    config.tracker.teleop.udp_port = config.chaser.udp_port
    config.tracker.teleop.robot_tag_id = config.chaser.tag_id
    runtime = TagGameRuntime(config, udp_enabled=True)
    runtime.start()
    app = build_dashboard_app(runtime)
    try:
        uvicorn.run(
            app,
            host=config.dashboard_host,
            port=config.dashboard_port,
            log_level="info",
        )
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
