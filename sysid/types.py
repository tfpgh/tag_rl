from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class WindowMetadata:
    split: str
    run_name: str
    start_time: float
    score_start_time: float
    end_time: float
    sample_count: int
    command_mean_abs: float
    command_max_abs: float
    translation_distance_m: float
    yaw_change_rad: float
    maneuver: str


@dataclass(frozen=True, slots=True)
class WindowBatch:
    split: str
    relative_sample_times_s: np.ndarray
    target_xy: np.ndarray
    target_yaw: np.ndarray
    sample_mask: np.ndarray
    score_mask: np.ndarray
    command_substeps: np.ndarray
    initial_pose: np.ndarray
    initial_velocity: np.ndarray
    initial_command: np.ndarray
    metadata: list[WindowMetadata]


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    train: WindowBatch
    eval: WindowBatch


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    total: float
    position: float
    yaw: float
    linear_velocity: float
    angular_velocity: float
    endpoint: float


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    score: ScoreBreakdown
    metrics: dict[str, float]
    maneuver_scores: dict[str, float]
    windows: int
