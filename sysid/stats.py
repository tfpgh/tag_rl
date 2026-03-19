from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any

from sysid.segments import extract_command_segments, filter_pose_samples
from sysid.types import RunData


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return pstdev(values) if len(values) >= 2 else 0.0


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def tracker_summary(run: RunData) -> dict[str, float]:
    tracker = run.tracker_timeline
    sample_times = [sample.monotonic_time for sample in tracker]
    sample_periods = [
        b - a for a, b in zip(sample_times, sample_times[1:], strict=False)
    ]
    return {
        "mean_capture_ms": _mean([sample.capture_ms for sample in tracker]),
        "mean_detector_ms": _mean([sample.detector_ms for sample in tracker]),
        "mean_tracking_ms": _mean([sample.tracking_ms for sample in tracker]),
        "mean_loop_ms": _mean([sample.loop_ms for sample in tracker]),
        "sample_period_mean_s": _mean(sample_periods),
        "sample_period_std_s": _std(sample_periods),
    }


def visibility_summary(run: RunData) -> dict[str, float]:
    target_robot = [sample.get("target_robot") for sample in run.samples]
    visible = sum(
        1
        for pose in target_robot
        if pose is not None and bool(pose.get("visible", True))
    )
    return {
        "visible_fraction": visible / len(target_robot) if target_robot else 0.0,
        "sample_count": float(len(target_robot)),
    }


def stationary_pose_noise(run: RunData) -> dict[str, float]:
    segments = extract_command_segments(run.command_timeline)
    stationary_segments = [segment for segment in segments if segment.label == "idle"]
    x_residuals: list[float] = []
    y_residuals: list[float] = []
    yaw_residuals: list[float] = []

    for segment in stationary_segments:
        poses = filter_pose_samples(
            run.target_pose, segment.start_time, segment.end_time
        )
        if len(poses) < 4:
            continue
        mean_x = _mean([pose.x_m for pose in poses])
        mean_y = _mean([pose.y_m for pose in poses])
        reference_yaw = poses[0].yaw_rad
        yaw_offsets = [_wrap_angle(pose.yaw_rad - reference_yaw) for pose in poses]
        mean_yaw_offset = _mean(yaw_offsets)
        x_residuals.extend(pose.x_m - mean_x for pose in poses)
        y_residuals.extend(pose.y_m - mean_y for pose in poses)
        yaw_residuals.extend(offset - mean_yaw_offset for offset in yaw_offsets)

    return {
        "position_noise_std_m": max(_std(x_residuals), _std(y_residuals)),
        "yaw_noise_std_rad": _std(yaw_residuals),
        "stationary_segment_count": float(len(stationary_segments)),
    }


def command_summary(run: RunData) -> dict[str, Any]:
    segments = extract_command_segments(run.command_timeline)
    counts: dict[str, int] = {}
    durations: dict[str, float] = {}
    for segment in segments:
        counts[segment.label] = counts.get(segment.label, 0) + 1
        durations[segment.label] = durations.get(segment.label, 0.0) + (
            segment.end_time - segment.start_time
        )
    return {
        "segment_counts": counts,
        "segment_durations_s": durations,
    }


def summarize_run(run: RunData) -> dict[str, Any]:
    return {
        "label": run.label,
        "tracker": tracker_summary(run),
        "visibility": visibility_summary(run),
        "noise": stationary_pose_noise(run),
        "commands": command_summary(run),
    }
