from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from sysid.types import CommandSample, PoseSample, RunData, TrackerSample


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _pose_from_record(
    monotonic_time: float, payload: dict[str, Any] | None
) -> PoseSample | None:
    if payload is None:
        return None
    if "x_m" not in payload or "y_m" not in payload:
        return None
    x_m = float(payload["x_m"])
    y_m = float(payload["y_m"])
    return PoseSample(
        monotonic_time=monotonic_time,
        x_m=x_m,
        y_m=y_m,
        yaw_rad=float(payload["yaw_rad"]),
        visible=bool(payload.get("visible", True)),
    )


def _command_from_sample(sample: dict[str, Any]) -> CommandSample:
    command = sample["command"]
    return CommandSample(
        monotonic_time=float(sample["monotonic_time"]),
        left=float(command["left"]),
        right=float(command["right"]),
        left_int16=int(command["left_int16"]),
        right_int16=int(command["right_int16"]),
    )


def _command_from_event(event: dict[str, Any]) -> CommandSample:
    scale = 32767.0
    return CommandSample(
        monotonic_time=float(event["monotonic_time"]),
        left=float(event["left_int16"]) / scale,
        right=float(event["right_int16"]) / scale,
        left_int16=int(event["left_int16"]),
        right_int16=int(event["right_int16"]),
    )


def load_run(run_dir: str | Path) -> RunData:
    run_path = Path(run_dir)
    samples = _load_jsonl(run_path / "samples.jsonl")
    commands = _load_jsonl(run_path / "commands.jsonl")
    metadata = json.loads((run_path / "metadata.json").read_text(encoding="utf-8"))

    target_pose: list[PoseSample] = []
    tracker_timeline: list[TrackerSample] = []
    for sample in samples:
        monotonic_time = float(sample["monotonic_time"])
        pose = _pose_from_record(monotonic_time, sample.get("target_robot"))
        if pose is not None:
            target_pose.append(pose)
        tracker = sample["tracker"]
        tracker_timeline.append(
            TrackerSample(
                monotonic_time=monotonic_time,
                capture_ms=float(tracker["capture_ms"]),
                detector_ms=float(tracker["detector_ms"]),
                tracking_ms=float(tracker["tracking_ms"]),
                loop_ms=float(tracker["loop_ms"]),
                visible_tags=int(tracker["visible_tags"]),
                calibration_valid=bool(sample["calibration_valid"]),
            )
        )

    command_timeline = [_command_from_event(event) for event in commands]
    if not command_timeline:
        command_timeline = [_command_from_sample(sample) for sample in samples]

    return RunData(
        run_dir=run_path,
        label=str(metadata["teleop"]["run_label"]),
        target_robot_tag_id=metadata["teleop"].get("robot_tag_id"),
        metadata=metadata,
        samples=samples,
        commands=commands,
        target_pose=target_pose,
        command_timeline=command_timeline,
        tracker_timeline=tracker_timeline,
    )


def discover_run_dirs(paths: Sequence[str | Path]) -> list[Path]:
    run_dirs: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if (path / "metadata.json").exists():
            run_dirs.append(path)
            continue
        if path.is_dir():
            children = sorted(
                child
                for child in path.iterdir()
                if child.is_dir() and (child / "metadata.json").exists()
            )
            run_dirs.extend(children)
            continue
        raise FileNotFoundError(f"No run metadata found at {path}")
    if not run_dirs:
        raise FileNotFoundError("No run directories found")
    return run_dirs
