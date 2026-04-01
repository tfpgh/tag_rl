from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sysid.types import DatasetSplits, WindowBatch, WindowMetadata

WINDOW_SECONDS = 2.0
PREROLL_SECONDS = 0.5
STRIDE_SECONDS = 0.5
SUBSTEP_DT_SECONDS = 0.005
MAX_ABS_INT16 = 32767.0


@dataclass(frozen=True, slots=True)
class WindowConfig:
    window_seconds: float = WINDOW_SECONDS
    preroll_seconds: float = PREROLL_SECONDS
    stride_seconds: float = STRIDE_SECONDS
    substep_dt_seconds: float = SUBSTEP_DT_SECONDS
    min_scored_samples: int = 24
    min_command_abs: float = 0.05
    min_motion_m: float = 0.03
    min_yaw_change_rad: float = 0.08

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


def _window_records(
    split: str, run_dir: Path, config: WindowConfig
) -> tuple[list[dict[str, np.ndarray]], list[WindowMetadata]]:
    samples = _iter_valid_samples(_load_jsonl(run_dir / "samples.jsonl"))
    commands = _load_jsonl(run_dir / "commands.jsonl")
    records: list[dict[str, np.ndarray]] = []
    metadata: list[WindowMetadata] = []
    if not samples or not commands:
        return records, metadata

    max_time = samples[-1]["monotonic_time"]
    next_window_start = samples[0]["monotonic_time"]
    for start_index, sample in enumerate(samples):
        window_start = sample["monotonic_time"]
        if window_start + config.total_seconds > max_time:
            break
        if window_start + 1e-9 < next_window_start:
            continue
        score_start = window_start + config.preroll_seconds
        window_end = window_start + config.total_seconds
        window_samples = [
            item
            for item in samples[start_index:]
            if item["monotonic_time"] <= window_end + 1e-9
        ]
        if len(window_samples) < 2:
            continue
        relative_times = np.array(
            [item["monotonic_time"] - window_start for item in window_samples],
            dtype=np.float32,
        )
        target_pose = np.stack(
            [_target_pose_array(item) for item in window_samples], axis=0
        )
        score_mask = relative_times >= config.preroll_seconds - 1e-6
        sample_count = int(np.count_nonzero(score_mask))
        if sample_count < config.min_scored_samples:
            continue

        command_substeps, initial_command = _build_command_substeps(
            commands, window_start, config
        )
        scored_commands = command_substeps[
            int(round(config.preroll_seconds / config.substep_dt_seconds)) :
        ]
        translation = np.linalg.norm(
            target_pose[-1, :2] - target_pose[int(np.argmax(score_mask)), :2]
        )
        yaw_change = abs(
            _wrap_angle(
                float(target_pose[-1, 2] - target_pose[int(np.argmax(score_mask)), 2])
            )
        )
        command_mean_abs = float(np.mean(np.abs(scored_commands)))
        command_max_abs = float(np.max(np.abs(scored_commands)))
        if (
            command_max_abs < config.min_command_abs
            and translation < config.min_motion_m
            and yaw_change < config.min_yaw_change_rad
        ):
            continue

        maneuver = _classify_window(scored_commands)
        initial_velocity = _estimate_initial_velocity(samples, start_index)
        records.append(
            {
                "relative_sample_times_s": relative_times,
                "target_xy": target_pose[:, :2],
                "target_yaw": target_pose[:, 2],
                "score_mask": score_mask.astype(np.bool_),
                "command_substeps": command_substeps.astype(np.float32),
                "initial_pose": target_pose[0].astype(np.float32),
                "initial_velocity": initial_velocity.astype(np.float32),
                "initial_command": initial_command.astype(np.float32),
            }
        )
        metadata.append(
            WindowMetadata(
                split=split,
                run_name=run_dir.name,
                start_time=window_start,
                score_start_time=score_start,
                end_time=window_end,
                sample_count=sample_count,
                command_mean_abs=command_mean_abs,
                command_max_abs=command_max_abs,
                translation_distance_m=float(translation),
                yaw_change_rad=float(yaw_change),
                maneuver=maneuver,
            )
        )
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
    train_records, train_metadata = _window_records("train", root / "train", cfg)
    eval_records, eval_metadata = _window_records("eval", root / "eval", cfg)
    return DatasetSplits(
        train=_pack_windows("train", train_records, train_metadata),
        eval=_pack_windows("eval", eval_records, eval_metadata),
    )
