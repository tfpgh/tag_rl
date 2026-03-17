from __future__ import annotations

import signal
from contextlib import suppress

import uvicorn

from online.camera import CameraWorker
from online.config import DemoConfig
from online.control_loop import ControlWorker
from online.detection import DetectionWorker
from online.observation import ObservationBuilder
from online.policy_runner import PolicyRunner
from online.robot_io import RobotClient
from online.runtime_state import RuntimeState
from online.server import create_app


def main() -> None:
    config = DemoConfig()
    state = RuntimeState()
    state.mutate_snapshot(
        lambda snapshot: setattr(
            snapshot.operator,
            "control_enabled",
            config.control.control_enabled_on_start,
        )
    )

    policy = PolicyRunner(config)
    config.env = policy.env_config
    config.tracking.obstacle_size_m = policy.env_config.obstacle_width
    observation_builder = ObservationBuilder(policy.env_config)
    chaser_robot = RobotClient(config.chaser_robot)
    evader_robot = RobotClient(config.evader_robot)

    workers = [
        CameraWorker(config.camera, state),
        DetectionWorker(config, state),
        ControlWorker(
            config,
            state,
            policy,
            observation_builder,
            chaser_robot,
            evader_robot,
        ),
    ]

    def shutdown(*_args) -> None:  # type: ignore[no-untyped-def]
        state.request_stop()
        chaser_robot.close()
        evader_robot.close()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for worker in workers:
        worker.start()

    app = create_app(config, state)
    server = uvicorn.Server(
        uvicorn.Config(
            app, host=config.gui.host, port=config.gui.port, log_level="info"
        )
    )
    try:
        server.run()
    finally:
        shutdown()
        for worker in workers:
            with suppress(RuntimeError):
                worker.join(timeout=1.0)


if __name__ == "__main__":
    main()
