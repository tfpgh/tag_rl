from __future__ import annotations

from dataclasses import dataclass, field

from online.core.config import TrackerConfig


@dataclass(slots=True)
class RobotEndpointConfig:
    name: str
    robot_ip: str
    udp_port: int
    tag_id: int


@dataclass(frozen=True, slots=True)
class RuntimeModeConfig:
    idle: str = "idle"
    armed: str = "armed"
    running: str = "running"
    estop: str = "estop"


@dataclass(frozen=True, slots=True)
class RuntimePhaseConfig:
    waiting: str = "waiting"
    ready: str = "ready"
    active: str = "active"
    tagged: str = "tagged"
    time_up: str = "time_up"
    stopped: str = "stopped"


@dataclass(frozen=True, slots=True)
class RuntimeActionConfig:
    arm: str = "arm"
    disarm: str = "disarm"
    start: str = "start"
    stop: str = "stop"
    reset: str = "reset"
    estop: str = "estop"
    clear_estop: str = "clear_estop"
    record_off: str = "record_off"
    record_train: str = "record_train"
    record_eval: str = "record_eval"


@dataclass(slots=True)
class GameRuntimeConfig:
    checkpoint_path: str = "runs/slow_checkpoint.pkl"
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    chaser: RobotEndpointConfig = field(
        default_factory=lambda: RobotEndpointConfig(
            name="chaser",
            robot_ip="192.168.1.5",
            udp_port=8888,
            tag_id=4,
        )
    )
    evader: RobotEndpointConfig = field(
        default_factory=lambda: RobotEndpointConfig(
            name="evader",
            robot_ip="192.168.1.4",
            udp_port=8888,
            tag_id=5,
        )
    )
    control_hz: float | None = None
    match_duration_s: float | None = None
    action_output_scale: float = 0.5
    stale_frame_timeout_s: float = 0.20
    target_loss_timeout_s: float = 0.20
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8000
    render_hz: float = 20.0
    telemetry_hz: float = 20.0
    dashboard_frame_width: int = 1280
    jpeg_quality: int = 80
    recording_root: str = "data"
    recording_games_dirname: str = "games"
    modes: RuntimeModeConfig = field(default_factory=RuntimeModeConfig)
    phases: RuntimePhaseConfig = field(default_factory=RuntimePhaseConfig)
    actions: RuntimeActionConfig = field(default_factory=RuntimeActionConfig)
