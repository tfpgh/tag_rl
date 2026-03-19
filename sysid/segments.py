from __future__ import annotations

from sysid.types import PoseSample, TimeSegment


def _command_magnitude(left: float, right: float) -> float:
    return max(abs(left), abs(right))


def _classify_command(left: float, right: float) -> str:
    magnitude = _command_magnitude(left, right)
    if magnitude < 0.05:
        return "idle"
    straight_threshold = 0.04 + 0.18 * magnitude
    spin_threshold = 0.04 + 0.18 * magnitude
    if abs(left - right) <= straight_threshold:
        return "straight"
    if left * right <= 0.0 and abs(left + right) <= spin_threshold:
        return "spin"
    return "arc"


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


def extract_command_segments(
    command_timeline: list,
    min_duration_s: float = 0.35,
    transition_duration_s: float = 0.25,
) -> list[TimeSegment]:
    if not command_timeline:
        return []

    coarse_segments: list[TimeSegment] = []
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
            _append_segment(coarse_segments, start, end, label, count)
        start = command.monotonic_time
        label = command_label
        count = 1

    end = command_timeline[-1].monotonic_time
    if end - start >= min_duration_s:
        _append_segment(coarse_segments, start, end, label, count)

    segments: list[TimeSegment] = []
    for segment in coarse_segments:
        segments.extend(_split_active_segment(segment, transition_duration_s))
    return segments


def filter_pose_samples(
    poses: list[PoseSample], start_time: float, end_time: float
) -> list[PoseSample]:
    return [pose for pose in poses if start_time <= pose.monotonic_time <= end_time]
