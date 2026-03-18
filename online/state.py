from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


CORNER_TAG_IDS = (0, 1, 2, 3)
CHASER_TAG_ID = 4
EVADER_TAG_ID = 5
DEFAULT_OBSTACLE_TAG_IDS = (6,)

TAG_CENTER_W_MM = 2338.4
TAG_CENTER_H_MM = 1119.2
TAG_SIZE_MM = 100.0
MAT_W_MM = TAG_CENTER_W_MM + TAG_SIZE_MM
MAT_H_MM = TAG_CENTER_H_MM + TAG_SIZE_MM


@dataclass(slots=True)
class CameraConfig:
    device_index: int = 0
    backend: int | None = None
    frame_width: int = 1920
    frame_height: int = 1080
    fps: int = 30
    mjpg: bool = True
    buffer_size: int = 1
    auto_exposure: float | None = 1.0
    exposure: float | None = 17.0


@dataclass(slots=True)
class ArenaViewConfig:
    frame_width: int = 1920
    stats_height: int = 35
    buffer_fraction: float = 0.01


@dataclass(slots=True)
class TrackerConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    arena_view: ArenaViewConfig = field(default_factory=ArenaViewConfig)
    detector_threads: int = 8
    detector_quad_decimate: float = 2.0
    detector_family: str = "tagStandard41h12"
    calibration_frames: int = 30
    smoothing_alpha: float = 0.35
    chaser_tag_id: int = CHASER_TAG_ID
    evader_tag_id: int = EVADER_TAG_ID
    obstacle_tag_ids: tuple[int, ...] = DEFAULT_OBSTACLE_TAG_IDS
    obstacle_size_mm: float = 100.0


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
    detector_ms: float
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
