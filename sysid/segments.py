from __future__ import annotations

from sysid.types import PoseSample, TimeSegment


def _classify_command(left: float, right: float) -> str:
    if abs(left) < 0.05 and abs(right) < 0.05:
        return "idle"
    if abs(left - right) < 0.08:
        return "straight"
    if abs(left + right) < 0.08:
        return "spin"
    return "arc"


def extract_command_segments(
    command_timeline: list, min_duration_s: float = 0.35
) -> list[TimeSegment]:
    if not command_timeline:
        return []

    segments: list[TimeSegment] = []
    start = command_timeline[0].monotonic_time
    label = _classify_command(command_timeline[0].left, command_timeline[0].right)
    count = 1

    for command in command_timeline[1:]:
        command_label = _classify_command(command.left, command.right)
        if command_label == label:
            count += 1
            continue
        end = command.monotonic_time
        if end - start >= min_duration_s:
            segments.append(
                TimeSegment(
                    start_time=start, end_time=end, label=label, sample_count=count
                )
            )
        start = command.monotonic_time
        label = command_label
        count = 1

    end = command_timeline[-1].monotonic_time
    if end - start >= min_duration_s:
        segments.append(
            TimeSegment(start_time=start, end_time=end, label=label, sample_count=count)
        )
    return segments


def filter_pose_samples(
    poses: list[PoseSample], start_time: float, end_time: float
) -> list[PoseSample]:
    return [pose for pose in poses if start_time <= pose.monotonic_time <= end_time]
