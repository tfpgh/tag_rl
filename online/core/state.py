from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TagDetection:
    tag_id: int
    center_px: np.ndarray
    corners_px: np.ndarray
    decision_margin: float
    hamming: int


@dataclass(slots=True)
class Pose2D:
    x_mm: float
    y_mm: float
    yaw_rad: float
    visible: bool
    last_seen_timestamp: float | None


@dataclass(slots=True)
class ObstacleState:
    tag_id: int
    pose: Pose2D
    size_mm: float


@dataclass(slots=True)
class ArenaCalibration:
    valid: bool
    samples_collected: int
    required_samples: int
    image_to_warp: np.ndarray | None
    image_to_world: np.ndarray | None
    warp_size_px: tuple[int, int]
    pixels_per_mm: float


@dataclass(slots=True)
class TrackingStats:
    fps: float
    frame_width: int
    frame_height: int
    capture_ms: float
    detector_ms: float
    tracking_ms: float
    loop_ms: float
    visible_tags: int


@dataclass(slots=True)
class BoardState:
    timestamp: float
    calibration: ArenaCalibration
    chaser: Pose2D | None
    evader: Pose2D | None
    obstacles: list[ObstacleState]
    detections: list[TagDetection]
    stats: TrackingStats
