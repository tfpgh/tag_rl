from __future__ import annotations

import sys
from math import pi
from dataclasses import dataclass, field
from pathlib import Path

from environment.config import EnvironmentConfig


def _default_camera_api_preference() -> int | None:
    if sys.platform == "darwin":
        return 1200  # cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("linux"):
        return 200  # cv2.CAP_V4L2
    return None


def _default_env_config() -> EnvironmentConfig:
    config = EnvironmentConfig()
    config.pipeline_randomization.enabled = False
    config.dynamics_randomization.enabled = False
    config.curriculum.enabled = False
    return config


@dataclass(slots=True)
class CameraConfig:
    device_index: int = 0
    api_preference: int | None = field(default_factory=_default_camera_api_preference)
    width: int = 1920
    height: int = 1080
    fps: int = 30
    buffersize: int = 1
    auto_exposure: int = 1
    exposure: int = 17
    mjpg_fourcc: str = "MJPG"


@dataclass(slots=True)
class DetectionConfig:
    families: str = "tagStandard41h12"
    nthreads: int = 7
    quad_decimate: float = 1.5
    quad_sigma: float = 0.0
    refine_edges: int = 1
    decode_sharpening: float = 0.25
    latest_only_sleep_s: float = 0.002
    roi_padding_px: int = 120
    use_dynamic_roi: bool = True


@dataclass(slots=True)
class ArenaCalibrationConfig:
    corner_tag_ids: tuple[int, int, int, int] = (0, 1, 2, 3)
    auxiliary_tag_ids: tuple[int, ...] = ()
    stable_frames_required: int = 15
    drop_tolerance_s: float = 0.5
    tag_size_m: float = 0.1
    mat_width_m: float = 2.4384
    mat_height_m: float = 1.2192


@dataclass(slots=True)
class TrackingConfig:
    chaser_tag_id: int = 4
    evader_tag_id: int = 5
    obstacle_tag_min_id: int = 6
    obstacle_size_m: float = 0.20
    obstacle_hold_timeout_s: float = 0.35
    robot_pose_timeout_s: float = 0.20
    robot_heading_offset_rad: float = pi / 2
    obstacle_heading_offset_rad: float = 0.0
    position_alpha: float = 0.7
    yaw_alpha: float = 0.7


@dataclass(slots=True)
class ControlConfig:
    hz: float = 20.0
    enable_chaser_freeze: bool = True
    max_command_step: float = 0.15
    world_state_timeout_s: float = 0.20
    control_enabled_on_start: bool = False


@dataclass(slots=True)
class GuiConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    websocket_rate_hz: float = 20.0
    mjpeg_rate_hz: float = 60.0
    jpeg_quality: int = 85
    frontend_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "gui"
    )


@dataclass(slots=True)
class RobotEndpoint:
    name: str
    ip: str
    port: int = 8888


@dataclass(slots=True)
class DemoConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    calibration: ArenaCalibrationConfig = field(default_factory=ArenaCalibrationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
    env: EnvironmentConfig = field(default_factory=_default_env_config)
    chaser_robot: RobotEndpoint = field(
        default_factory=lambda: RobotEndpoint(name="chaser", ip="192.168.1.3")
    )
    evader_robot: RobotEndpoint = field(
        default_factory=lambda: RobotEndpoint(name="evader", ip="192.168.1.5")
    )
    checkpoint_path: Path = Path("checkpoints/1.pkl")
