from __future__ import annotations

from dataclasses import dataclass, field

from online.core.config import TrackerConfig


@dataclass(slots=True)
class RobotEndpointConfig:
    name: str
    robot_ip: str
    udp_port: int
    tag_id: int


@dataclass(slots=True)
class GameRuntimeConfig:
    checkpoint_path: str
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
            robot_ip="192.168.1.6",
            udp_port=8888,
            tag_id=5,
        )
    )
    control_hz: float | None = None
    match_duration_s: float | None = None
    stale_frame_timeout_s: float = 0.20
    target_loss_timeout_s: float = 0.20
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8000
    jpeg_quality: int = 80
