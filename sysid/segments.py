from __future__ import annotations

import math

from sysid.types import AlignedTrajectory, PoseSample, TimeSegment


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _append_segment(
    segments: list[TimeSegment],
    start_time: float,
    end_time: float,
    label: str,
    sample_count: int,
) -> None:
    if end_time <= start_time or sample_count <= 0:
        return
    segments.append(
        TimeSegment(
            start_time=start_time,
            end_time=end_time,
            label=label,
            sample_count=sample_count,
        )
    )


def _split_active_segment(
    segment: TimeSegment, transition_duration_s: float
) -> list[TimeSegment]:
    if segment.label == "idle":
        return [segment]
    duration_s = segment.end_time - segment.start_time
    if duration_s < 2.0 * transition_duration_s:
        return [segment]

    transition_ratio = transition_duration_s / duration_s
    transition_samples = max(1, int(round(segment.sample_count * transition_ratio)))
    if transition_samples * 2 >= segment.sample_count:
        return [segment]

    steady_samples = segment.sample_count - 2 * transition_samples
    start_end = segment.start_time + transition_duration_s
    stop_start = segment.end_time - transition_duration_s
    return [
        TimeSegment(
            segment.start_time, start_end, f"start_{segment.label}", transition_samples
        ),
        TimeSegment(start_end, stop_start, segment.label, steady_samples),
        TimeSegment(
            stop_start, segment.end_time, f"stop_{segment.label}", transition_samples
        ),
    ]


def _classify_motion(
    linear_speed_m_s: float,
    angular_speed_rad_s: float,
    command_linear: float,
    command_angular: float,
) -> str:
    command_linear_abs = abs(command_linear)
    command_angular_abs = abs(command_angular)
    command_turn_ratio = command_angular_abs / max(command_linear_abs, 0.05)

    if (
        linear_speed_m_s < 0.04
        and angular_speed_rad_s < 0.35
        and command_linear_abs < 0.08
        and command_angular_abs < 0.08
    ):
        return "idle"
    if (linear_speed_m_s < 0.14 and angular_speed_rad_s > 0.9) or (
        linear_speed_m_s < 0.08
        and command_turn_ratio > 1.5
        and angular_speed_rad_s > 0.2
    ):
        return "spin"
    if linear_speed_m_s > 0.04 or command_linear_abs > 0.1:
        turn_ratio = angular_speed_rad_s / max(linear_speed_m_s, 1e-6)
        if turn_ratio < 1.3 and angular_speed_rad_s < 0.6 and command_turn_ratio < 0.8:
            return "straight"
    if linear_speed_m_s < 0.06 and abs(command_angular) > abs(command_linear):
        return "spin"
    return "arc"


def extract_motion_segments(
    trajectory: AlignedTrajectory,
    min_duration_s: float = 0.35,
    transition_duration_s: float = 0.25,
    smoothing_window: int = 5,
) -> list[TimeSegment]:
    step_count = len(trajectory.times_s) - 1
    if step_count <= 0:
        return []

    linear_speed = [0.0] * step_count
    angular_speed = [0.0] * step_count
    command_linear = [
        0.5 * (trajectory.left[index] + trajectory.right[index])
        for index in range(step_count)
    ]
    command_angular = [
        0.5 * (trajectory.right[index] - trajectory.left[index])
        for index in range(step_count)
    ]

    half_window = max(0, smoothing_window // 2)
    for index in range(step_count):
        start = max(0, index - half_window)
        end = min(step_count - 1, index + half_window)
        dx = trajectory.x_m[end + 1] - trajectory.x_m[start]
        dy = trajectory.y_m[end + 1] - trajectory.y_m[start]
        dyaw = _wrap_angle(trajectory.yaw_rad[end + 1] - trajectory.yaw_rad[start])
        dt = max(trajectory.times_s[end + 1] - trajectory.times_s[start], 1e-6)
        linear_speed[index] = math.hypot(dx, dy) / dt
        angular_speed[index] = abs(dyaw) / dt

    coarse_segments: list[TimeSegment] = []
    start_time = trajectory.times_s[0]
    current_label = _classify_motion(
        linear_speed[0], angular_speed[0], command_linear[0], command_angular[0]
    )
    current_count = 1

    for index in range(1, step_count):
        label = _classify_motion(
            linear_speed[index],
            angular_speed[index],
            command_linear[index],
            command_angular[index],
        )
        if label == current_label:
            current_count += 1
            continue
        end_time = trajectory.times_s[index]
        if end_time - start_time >= min_duration_s:
            _append_segment(
                coarse_segments, start_time, end_time, current_label, current_count
            )
        start_time = trajectory.times_s[index]
        current_label = label
        current_count = 1

    end_time = trajectory.times_s[-1]
    if end_time - start_time >= min_duration_s:
        _append_segment(
            coarse_segments, start_time, end_time, current_label, current_count
        )

    segments: list[TimeSegment] = []
    for segment in coarse_segments:
        segments.extend(_split_active_segment(segment, transition_duration_s))
    return segments


def filter_pose_samples(
    poses: list[PoseSample], start_time: float, end_time: float
) -> list[PoseSample]:
    return [pose for pose in poses if start_time <= pose.monotonic_time <= end_time]
