from __future__ import annotations

import math

from sysid.types import AlignedTrajectory, CommandSample, PoseSample, RunData


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _unwrap_yaws(poses: list[PoseSample]) -> list[float]:
    if not poses:
        return []
    unwrapped = [poses[0].yaw_rad]
    for pose in poses[1:]:
        delta = _wrap_angle(pose.yaw_rad - unwrapped[-1])
        unwrapped.append(unwrapped[-1] + delta)
    return unwrapped


def _sample_command(
    commands: list[CommandSample], time_s: float, start_index: int
) -> tuple[CommandSample, int]:
    index = start_index
    while index + 1 < len(commands) and commands[index + 1].monotonic_time <= time_s:
        index += 1
    return commands[index], index


def align_run(
    run: RunData,
    target_hz: float | None = None,
    min_duration_s: float = 1.0,
) -> AlignedTrajectory:
    poses = [pose for pose in run.target_pose if pose.visible]
    commands = run.command_timeline
    if not poses or not commands:
        return AlignedTrajectory([], [], [], [], [], [])

    send_hz = target_hz
    if send_hz is None:
        send_hz = float(run.metadata.get("teleop", {}).get("send_hz", 20.0))
    dt = 1.0 / max(send_hz, 1e-6)

    start_time = max(poses[0].monotonic_time, commands[0].monotonic_time)
    end_time = min(poses[-1].monotonic_time, commands[-1].monotonic_time)
    if end_time - start_time < min_duration_s:
        return AlignedTrajectory([], [], [], [], [], [])

    pose_times = [pose.monotonic_time for pose in poses]
    pose_x = [pose.x_m for pose in poses]
    pose_y = [pose.y_m for pose in poses]
    pose_yaw = _unwrap_yaws(poses)

    times_s: list[float] = []
    left: list[float] = []
    right: list[float] = []
    x_m: list[float] = []
    y_m: list[float] = []
    yaw_rad: list[float] = []
    command_index = 0
    time_s = start_time
    while time_s <= end_time + 1e-9:
        command, command_index = _sample_command(commands, time_s, command_index)
        times_s.append(time_s)
        left.append(command.left)
        right.append(command.right)
        x_m.append(float(_interp(time_s, pose_times, pose_x)))
        y_m.append(float(_interp(time_s, pose_times, pose_y)))
        yaw_rad.append(float(_interp(time_s, pose_times, pose_yaw)))
        time_s += dt

    return AlignedTrajectory(times_s, left, right, x_m, y_m, yaw_rad)


def _interp(x: float, xp: list[float], fp: list[float]) -> float:
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    lo = 0
    hi = len(xp) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0, x1 = xp[lo], xp[hi]
    y0, y1 = fp[lo], fp[hi]
    if x1 <= x0:
        return y0
    alpha = (x - x0) / (x1 - x0)
    return y0 + alpha * (y1 - y0)
