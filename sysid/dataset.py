from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sysid.types import DatasetSplits, WindowBatch, WindowMetadata

WINDOW_SECONDS = 1.5
PREROLL_SECONDS = 0.5
STRIDE_SECONDS = 0.5
SUBSTEP_DT_SECONDS = 0.005
MAX_ABS_INT16 = 32767.0
GAME_TAIL_TRIM_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class WindowConfig:
    window_seconds: float = WINDOW_SECONDS
    preroll_seconds: float = PREROLL_SECONDS
    stride_seconds: float = STRIDE_SECONDS
    substep_dt_seconds: float = SUBSTEP_DT_SECONDS
    min_scored_samples: int = 10
    min_command_abs: float = 0.05
    min_motion_m: float = 0.003
    min_yaw_change_rad: float = 0.008

    @property
    def total_seconds(self) -> float:
        return self.window_seconds + self.preroll_seconds

    @property
    def num_substeps(self) -> int:
        return int(round(self.total_seconds / self.substep_dt_seconds))


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _load_jsonl_if_exists(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return _load_jsonl(path)


def _game_tail_trim_seconds(run_dir: Path) -> float:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return 0.0
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    collection = metadata.get("collection")
    if not isinstance(collection, dict):
        return 0.0
    if collection.get("mode") != "game":
        return 0.0
    return GAME_TAIL_TRIM_SECONDS


def _command_array(record: dict) -> np.ndarray:
    return (
        np.array([record["left_int16"], record["right_int16"]], dtype=np.float32)
        / MAX_ABS_INT16
    )


def _classify_window(commands: np.ndarray) -> str:
    max_abs = np.max(np.abs(commands), axis=1)
    active = max_abs > 0.05
    if not np.any(active):
        return "idle"

    active_commands = commands[active]
    diff = np.abs(active_commands[:, 0] - active_commands[:, 1])
    prod = active_commands[:, 0] * active_commands[:, 1]
    straight_like = np.mean((prod > 0.0) & (diff < 0.08))
    opposite = np.mean(prod < 0.0)
    left_bias = np.mean(active_commands[:, 0] > active_commands[:, 1])
    right_bias = np.mean(active_commands[:, 1] > active_commands[:, 0])

    if straight_like >= 0.7:
        return "straight"
    if opposite >= 0.2:
        return "pivot"
    if left_bias > right_bias:
        return "left_arc"
    return "right_arc"


def _target_pose_array(sample: dict) -> np.ndarray:
    target = sample["target_robot"]
    return np.array([target["x_m"], target["y_m"], target["yaw_rad"]], dtype=np.float32)


def _estimate_initial_velocity(samples: list[dict], start_index: int) -> np.ndarray:
    if start_index + 1 >= len(samples):
        return np.zeros((3,), dtype=np.float32)
    current = samples[start_index]
    nxt = samples[start_index + 1]
    dt = nxt["monotonic_time"] - current["monotonic_time"]
    if dt <= 0.0:
        return np.zeros((3,), dtype=np.float32)
    current_pose = current["target_robot"]
    next_pose = nxt["target_robot"]
    dx = next_pose["x_m"] - current_pose["x_m"]
    dy = next_pose["y_m"] - current_pose["y_m"]
    dyaw = _wrap_angle(next_pose["yaw_rad"] - current_pose["yaw_rad"])
    return np.array([dx / dt, dy / dt, dyaw / dt], dtype=np.float32)


def _build_command_substeps(
    commands: list[dict], window_start: float, config: WindowConfig
) -> tuple[np.ndarray, np.ndarray]:
    num_substeps = config.num_substeps
    substep_times = (
        window_start
        + np.arange(num_substeps, dtype=np.float64) * config.substep_dt_seconds
    )
    command_times = np.array(
        [command["monotonic_time"] for command in commands], dtype=np.float64
    )
    command_values = np.stack([_command_array(command) for command in commands], axis=0)
    initial_index = int(np.searchsorted(command_times, window_start, side="right") - 1)
    initial_index = max(initial_index, 0)
    initial_command = command_values[initial_index]
    indices = np.searchsorted(command_times, substep_times, side="right") - 1
    indices = np.clip(indices, 0, len(command_values) - 1)
    return command_values[indices], initial_command.astype(np.float32)


def _iter_valid_samples(samples: list[dict]) -> list[dict]:
    return [
        sample
        for sample in samples
        if sample.get("calibration_valid")
        and sample.get("target_robot") is not None
        and sample["target_robot"].get("visible")
    ]


def _record_from_samples(
    split: str,
    run_dir: Path,
    config: WindowConfig,
    samples: list[dict],
    commands: list[dict],
    start_index: int,
    score_start: float,
    window_end: float,
) -> tuple[dict[str, np.ndarray] | None, WindowMetadata | None]:
    if start_index >= len(samples):
        return None, None
    window_start = float(samples[start_index]["monotonic_time"])
    window_samples = [
        item
        for item in samples[start_index:]
        if item["monotonic_time"] <= window_end + 1e-9
    ]
    if len(window_samples) < 2:
        return None, None

    relative_times = np.array(
        [item["monotonic_time"] - window_start for item in window_samples],
        dtype=np.float32,
    )
    target_pose = np.stack(
        [_target_pose_array(item) for item in window_samples], axis=0
    )
    score_mask = relative_times >= (score_start - window_start) - 1e-6
    sample_count = int(np.count_nonzero(score_mask))
    if sample_count < config.min_scored_samples:
        return None, None

    command_substeps, initial_command = _build_command_substeps(
        commands, window_start, config
    )
    scored_commands = command_substeps[
        int(round(config.preroll_seconds / config.substep_dt_seconds)) :
    ]
    score_start_index = int(np.argmax(score_mask))
    translation = np.linalg.norm(
        target_pose[-1, :2] - target_pose[score_start_index, :2]
    )
    yaw_change = abs(
        _wrap_angle(float(target_pose[-1, 2] - target_pose[score_start_index, 2]))
    )
    command_mean_abs = float(np.mean(np.abs(scored_commands)))
    command_max_abs = float(np.max(np.abs(scored_commands)))
    if (
        command_max_abs < config.min_command_abs
        and translation < config.min_motion_m
        and yaw_change < config.min_yaw_change_rad
    ):
        return None, None

    maneuver = _classify_window(scored_commands)
    initial_velocity = _estimate_initial_velocity(samples, start_index)
    record = {
        "relative_sample_times_s": relative_times,
        "target_xy": target_pose[:, :2],
        "target_yaw": target_pose[:, 2],
        "score_mask": score_mask.astype(np.bool_),
        "command_substeps": command_substeps.astype(np.float32),
        "initial_pose": target_pose[0].astype(np.float32),
        "initial_velocity": initial_velocity.astype(np.float32),
        "initial_command": initial_command.astype(np.float32),
    }
    metadata = WindowMetadata(
        split=split,
        run_name=run_dir.name,
        start_time=window_start,
        score_start_time=float(score_start),
        end_time=float(window_end),
        sample_count=sample_count,
        command_mean_abs=command_mean_abs,
        command_max_abs=command_max_abs,
        translation_distance_m=float(translation),
        yaw_change_rad=float(yaw_change),
        maneuver=maneuver,
    )
    return record, metadata


def _segment_records(
    split: str, run_dir: Path, config: WindowConfig
) -> tuple[list[dict[str, np.ndarray]], list[WindowMetadata]]:
    samples = _iter_valid_samples(_load_jsonl(run_dir / "samples.jsonl"))
    commands = _load_jsonl(run_dir / "commands.jsonl")
    segments = _load_jsonl_if_exists(run_dir / "segments.jsonl")
    tail_trim_seconds = _game_tail_trim_seconds(run_dir)
    records: list[dict[str, np.ndarray]] = []
    metadata: list[WindowMetadata] = []
    if not samples or not commands or not segments:
        return records, metadata

    sample_times = np.array(
        [sample["monotonic_time"] for sample in samples], dtype=np.float64
    )
    for segment in segments:
        if segment.get("status") != "completed":
            continue
        started_at = segment.get("started_at")
        ended_at = segment.get("ended_at")
        if started_at is None or ended_at is None:
            continue
        pre_roll_s = float(segment.get("pre_roll_s", config.preroll_seconds))
        window_start = float(started_at)
        score_start = window_start + pre_roll_s
        window_end = float(ended_at) - tail_trim_seconds
        if window_end <= score_start:
            continue
        start_index = int(
            np.searchsorted(sample_times, window_start - 1e-9, side="left")
        )
        record, item = _record_from_samples(
            split,
            run_dir,
            config,
            samples,
            commands,
            start_index,
            score_start,
            window_end,
        )
        if record is None or item is None:
            continue
        records.append(record)
        metadata.append(item)
    return records, metadata


def _discover_run_dirs(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        return []

    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for candidate in [
        split_dir,
        *sorted(path for path in split_dir.rglob("*") if path.is_dir()),
    ]:
        if candidate in seen:
            continue
        if (candidate / "samples.jsonl").exists() and (
            candidate / "commands.jsonl"
        ).exists():
            run_dirs.append(candidate)
            seen.add(candidate)
    return run_dirs


def _window_records(
    split: str, run_dir: Path, config: WindowConfig
) -> tuple[list[dict[str, np.ndarray]], list[WindowMetadata]]:
    samples = _iter_valid_samples(_load_jsonl(run_dir / "samples.jsonl"))
    commands = _load_jsonl(run_dir / "commands.jsonl")
    tail_trim_seconds = _game_tail_trim_seconds(run_dir)
    records: list[dict[str, np.ndarray]] = []
    metadata: list[WindowMetadata] = []
    if not samples or not commands:
        return records, metadata

    max_time = samples[-1]["monotonic_time"] - tail_trim_seconds
    next_window_start = samples[0]["monotonic_time"]
    if max_time <= next_window_start:
        return records, metadata
    for start_index, sample in enumerate(samples):
        window_start = sample["monotonic_time"]
        if window_start + config.total_seconds > max_time:
            break
        if window_start + 1e-9 < next_window_start:
            continue
        score_start = window_start + config.preroll_seconds
        window_end = window_start + config.total_seconds
        record, item = _record_from_samples(
            split,
            run_dir,
            config,
            samples,
            commands,
            start_index,
            score_start,
            window_end,
        )
        if record is None or item is None:
            continue
        records.append(record)
        metadata.append(item)
        next_window_start = window_start + config.stride_seconds
    return records, metadata


def _pack_windows(
    split: str, records: list[dict[str, np.ndarray]], metadata: list[WindowMetadata]
) -> WindowBatch:
    if not records:
        raise ValueError(f"No usable windows found for split '{split}'")
    max_samples = max(len(record["relative_sample_times_s"]) for record in records)
    num_windows = len(records)
    relative_sample_times_s = np.zeros((num_windows, max_samples), dtype=np.float32)
    target_xy = np.zeros((num_windows, max_samples, 2), dtype=np.float32)
    target_yaw = np.zeros((num_windows, max_samples), dtype=np.float32)
    sample_mask = np.zeros((num_windows, max_samples), dtype=np.bool_)
    score_mask = np.zeros((num_windows, max_samples), dtype=np.bool_)

    for index, record in enumerate(records):
        length = len(record["relative_sample_times_s"])
        relative_sample_times_s[index, :length] = record["relative_sample_times_s"]
        target_xy[index, :length] = record["target_xy"]
        target_yaw[index, :length] = record["target_yaw"]
        sample_mask[index, :length] = True
        score_mask[index, :length] = record["score_mask"]

    command_substeps = np.stack(
        [record["command_substeps"] for record in records], axis=0
    )
    initial_pose = np.stack([record["initial_pose"] for record in records], axis=0)
    initial_velocity = np.stack(
        [record["initial_velocity"] for record in records], axis=0
    )
    initial_command = np.stack(
        [record["initial_command"] for record in records], axis=0
    )

    return WindowBatch(
        split=split,
        relative_sample_times_s=relative_sample_times_s,
        target_xy=target_xy,
        target_yaw=target_yaw,
        sample_mask=sample_mask,
        score_mask=score_mask,
        command_substeps=command_substeps,
        initial_pose=initial_pose,
        initial_velocity=initial_velocity,
        initial_command=initial_command,
        metadata=metadata,
    )


def load_dataset_splits(
    root: Path, config: WindowConfig | None = None
) -> DatasetSplits:
    cfg = config or WindowConfig()
    train_records: list[dict[str, np.ndarray]] = []
    train_metadata: list[WindowMetadata] = []
    eval_records: list[dict[str, np.ndarray]] = []
    eval_metadata: list[WindowMetadata] = []
    for run_dir in _discover_run_dirs(root / "train"):
        records, metadata = _segment_records("train", run_dir, cfg)
        if not records:
            records, metadata = _window_records("train", run_dir, cfg)
        train_records.extend(records)
        train_metadata.extend(metadata)
    for run_dir in _discover_run_dirs(root / "eval"):
        records, metadata = _segment_records("eval", run_dir, cfg)
        if not records:
            records, metadata = _window_records("eval", run_dir, cfg)
        eval_records.extend(records)
        eval_metadata.extend(metadata)
    return DatasetSplits(
        train=_pack_windows("train", train_records, train_metadata),
        eval=_pack_windows("eval", eval_records, eval_metadata),
    )
