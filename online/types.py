from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import numpy as np


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_serializable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(slots=True)
class Pose2D:
    x: float
    y: float
    yaw: float
    timestamp: float


@dataclass(slots=True)
class TagDetection:
    tag_id: int
    center_px: tuple[float, float]
    corners_px: list[tuple[float, float]]
    decision_margin: float
    hamming: int


@dataclass(slots=True)
class FrameState:
    frame_id: int = 0
    timestamp: float = 0.0
    width: int = 0
    height: int = 0
    age_s: float = 0.0


@dataclass(slots=True)
class CalibrationState:
    status: str = "uncalibrated"
    stable_count: int = 0
    last_update_s: float = 0.0
    source_tag_ids: list[int] = field(default_factory=list)
    homography: list[list[float]] | None = None
    arena_corners_world: list[tuple[float, float]] = field(default_factory=list)


@dataclass(slots=True)
class TrackedBody:
    tag_id: int
    label: str
    visible: bool
    stale: bool
    age_s: float
    raw_pose: Pose2D | None
    filtered_pose: Pose2D | None


@dataclass(slots=True)
class TrackedObstacle:
    tag_id: int
    visible: bool
    stale: bool
    age_s: float
    pose: Pose2D
    size_m: float


@dataclass(slots=True)
class WorldState:
    timestamp: float = 0.0
    ready: bool = False
    frame_id: int = 0
    chaser: TrackedBody | None = None
    evader: TrackedBody | None = None
    obstacles: list[TrackedObstacle] = field(default_factory=list)


@dataclass(slots=True)
class PolicyActionState:
    timestamp: float = 0.0
    enabled: bool = False
    episode_progress: float = 0.0
    chaser_action: list[float] = field(default_factory=lambda: [0.0, 0.0])
    evader_action: list[float] = field(default_factory=lambda: [0.0, 0.0])
    observation_size: int = 0
    reset_count: int = 0
    last_reason: str = "idle"


@dataclass(slots=True)
class RobotCommandState:
    name: str
    timestamp: float = 0.0
    left: float = 0.0
    right: float = 0.0
    packets_sent: int = 0
    watchdog_stop: bool = False


@dataclass(slots=True)
class RuntimeStats:
    capture_fps: float = 0.0
    detection_fps: float = 0.0
    detection_ms: float = 0.0
    control_hz: float = 0.0
    control_ms: float = 0.0
    frame_age_s: float = 0.0
    world_age_s: float = 0.0
    detections: int = 0
    last_error: str = ""


@dataclass(slots=True)
class OperatorState:
    control_enabled: bool = False
    emergency_stop: bool = False
    pause_detection: bool = False
    show_debug: bool = True


@dataclass(slots=True)
class Snapshot:
    frame: FrameState = field(default_factory=FrameState)
    detections: list[TagDetection] = field(default_factory=list)
    calibration: CalibrationState = field(default_factory=CalibrationState)
    world: WorldState = field(default_factory=WorldState)
    policy: PolicyActionState = field(default_factory=PolicyActionState)
    chaser_command: RobotCommandState = field(
        default_factory=lambda: RobotCommandState(name="chaser")
    )
    evader_command: RobotCommandState = field(
        default_factory=lambda: RobotCommandState(name="evader")
    )
    stats: RuntimeStats = field(default_factory=RuntimeStats)
    operator: OperatorState = field(default_factory=OperatorState)

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(self)
