from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PoseSample:
    monotonic_time: float
    x_m: float
    y_m: float
    yaw_rad: float
    visible: bool


@dataclass(frozen=True, slots=True)
class CommandSample:
    monotonic_time: float
    left: float
    right: float
    left_int16: int
    right_int16: int


@dataclass(frozen=True, slots=True)
class TrackerSample:
    monotonic_time: float
    capture_ms: float
    detector_ms: float
    tracking_ms: float
    loop_ms: float
    visible_tags: int
    calibration_valid: bool


@dataclass(frozen=True, slots=True)
class RunData:
    run_dir: Path
    label: str
    target_robot_tag_id: int | None
    metadata: dict[str, Any]
    samples: list[dict[str, Any]]
    commands: list[dict[str, Any]]
    target_pose: list[PoseSample]
    command_timeline: list[CommandSample]
    tracker_timeline: list[TrackerSample]


@dataclass(frozen=True, slots=True)
class TimeSegment:
    start_time: float
    end_time: float
    label: str
    sample_count: int


@dataclass(frozen=True, slots=True)
class AlignedTrajectory:
    times_s: list[float]
    left: list[float]
    right: list[float]
    x_m: list[float]
    y_m: list[float]
    yaw_rad: list[float]


@dataclass(frozen=True, slots=True)
class ReplayMetrics:
    position_rmse_m: float
    yaw_rmse_rad: float
    endpoint_position_error_m: float
    endpoint_yaw_error_rad: float
    score: float
    sample_count: int
