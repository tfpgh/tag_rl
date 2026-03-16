#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pickle
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from tensorboardX.proto import event_pb2


EVENT_GLOB = "events.out.tfevents.*"
DEFAULT_METRICS = (
    "optimizer/learning_rate",
    "env/tag_rate",
    "env/time_up_rate",
    "env/mean_episode_length",
    "env/mean_distance",
    "chaser/mean_reward",
    "evader/mean_reward",
    "chaser/value_loss",
    "evader/value_loss",
    "chaser/entropy",
    "evader/entropy",
)
RANDOMIZATION_METRICS = (
    "randomization/obstacle_count",
    "randomization/obstacle_count_std",
    "randomization/action_delay_steps",
    "randomization/action_delay_steps_std",
    "randomization/observation_delay_steps",
    "randomization/observation_delay_steps_std",
    "randomization/action_drop_probability",
    "randomization/action_drop_probability_std",
    "randomization/frame_drop_probability",
    "randomization/frame_drop_probability_std",
    "randomization/stale_observation_probability",
    "randomization/stale_observation_probability_std",
    "randomization/position_noise_std",
    "randomization/position_noise_std_std",
    "randomization/yaw_noise_std",
    "randomization/yaw_noise_std_std",
    "randomization/floor_friction_scale",
    "randomization/floor_friction_scale_std",
    "randomization/floor_friction_scale_abs_deviation",
    "randomization/chaser_mass_scale",
    "randomization/chaser_mass_scale_std",
    "randomization/chaser_mass_scale_abs_deviation",
    "randomization/evader_mass_scale",
    "randomization/evader_mass_scale_std",
    "randomization/evader_mass_scale_abs_deviation",
    "randomization/chaser_motor_balance",
    "randomization/chaser_motor_balance_std",
    "randomization/chaser_motor_balance_abs",
    "randomization/evader_motor_balance",
    "randomization/evader_motor_balance_std",
    "randomization/evader_motor_balance_abs",
    "randomization/chaser_motor_strength_scale",
    "randomization/chaser_motor_strength_scale_std",
    "randomization/chaser_motor_strength_scale_abs_deviation",
    "randomization/evader_motor_strength_scale",
    "randomization/evader_motor_strength_scale_std",
    "randomization/evader_motor_strength_scale_abs_deviation",
)


@dataclass(frozen=True)
class Series:
    steps: np.ndarray
    values: np.ndarray
    wall_times: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a TensorBoard run for curriculum and LR timing."
    )
    parser.add_argument(
        "run_name", help="Run directory name, e.g. Mar15_15-44-03_node002"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Directory under which to search for the run (default: current directory)",
    )
    parser.add_argument(
        "--window-updates",
        type=int,
        default=50,
        help="Window size in updates for milestone comparisons (default: 50)",
    )
    return parser.parse_args()


def find_run_dir(root: Path, run_name: str) -> Path:
    direct = root / run_name
    if direct.is_dir():
        return direct

    matches = [path for path in root.rglob(run_name) if path.is_dir()]
    if not matches:
        raise FileNotFoundError(
            f"Could not find run directory named {run_name!r} under {root}"
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Found multiple run directories for {run_name!r}: {', '.join(str(m) for m in matches)}"
        )
    return matches[0]


def find_event_files(run_dir: Path) -> list[Path]:
    files = sorted(run_dir.glob(EVENT_GLOB))
    if files:
        return files
    recursive = sorted(run_dir.rglob(EVENT_GLOB))
    if not recursive:
        raise FileNotFoundError(f"No TensorBoard event files found under {run_dir}")
    return recursive


def iter_event_records(path: Path):
    with path.open("rb") as handle:
        while True:
            header = handle.read(8)
            if not header:
                break
            if len(header) != 8:
                raise ValueError(f"Corrupt event file header in {path}")
            length = struct.unpack("Q", header)[0]
            handle.read(4)
            payload = handle.read(length)
            handle.read(4)
            if len(payload) != length:
                raise ValueError(f"Corrupt event payload in {path}")
            event = cast(Any, event_pb2).Event()
            event.ParseFromString(payload)
            yield event


def load_scalars(event_files: list[Path]) -> dict[str, Series]:
    raw: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    for event_file in event_files:
        for event in iter_event_records(event_file):
            if not event.HasField("summary"):
                continue
            step = int(event.step)
            wall_time = float(event.wall_time)
            for value in event.summary.value:
                if not value.HasField("simple_value"):
                    continue
                raw[value.tag][step] = (wall_time, float(value.simple_value))

    series_map: dict[str, Series] = {}
    for tag, step_map in raw.items():
        ordered = sorted(step_map.items())
        steps = np.asarray([step for step, _ in ordered], dtype=np.int64)
        wall_times = np.asarray(
            [payload[0] for _, payload in ordered], dtype=np.float64
        )
        values = np.asarray([payload[1] for _, payload in ordered], dtype=np.float64)
        series_map[tag] = Series(steps=steps, values=values, wall_times=wall_times)
    return series_map


def load_run_config(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path | None]:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_files = sorted(checkpoint_dir.glob("step_*.pkl"))
    if checkpoint_files:
        latest = checkpoint_files[-1]
        with latest.open("rb") as handle:
            checkpoint = pickle.load(handle)
        return (
            dict(checkpoint.get("rl_config", {})),
            dict(checkpoint.get("env_config", {})),
            latest,
        )

    rl_config = {
        "num_envs": 16384,
        "num_steps": 128,
        "total_timesteps": int(5e9),
        "lr": 3e-4,
        "anneal_lr": True,
        "lr_decay_start_fraction": 0.0,
        "final_lr_scale": 0.0,
        "update_epochs": 4,
        "num_minibatches": 64,
    }
    env_config = {
        "curriculum": {
            "enabled": True,
            "layout_start": 0.1,
            "pipeline_start": 0.2,
            "dynamics_start": 0.3,
            "full_progress": 0.85,
            "exponent": 1.5,
        }
    }
    return rl_config, env_config, None


def get_nested(payload: dict[str, Any], *keys: str, default: Any) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    window = max(1, min(window, values.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def fmt_value(value: float, precision: int = 4) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{precision}f}"


def scheduled_lr_for_update(
    base_lr: float,
    anneal_lr: bool,
    lr_decay_start_fraction: float,
    final_lr_scale: float,
    total_updates: int,
    update_idx: int,
) -> float:
    if not anneal_lr:
        return base_lr
    total_updates = max(total_updates, 1)
    start_update = lr_decay_start_fraction * total_updates
    if lr_decay_start_fraction >= 1.0:
        return base_lr
    decay_progress = min(
        max((update_idx - start_update) / max(total_updates - start_update, 1.0), 0.0),
        1.0,
    )
    scale = 1.0 + (final_lr_scale - 1.0) * decay_progress
    return base_lr * scale


def phase_slices(progress_steps: list[tuple[str, float]], all_steps: np.ndarray):
    total_step = int(all_steps[-1])
    phase_defs = [
        ("pre_layout", 0.0, progress_steps[0][1]),
        ("layout_only", progress_steps[0][1], progress_steps[1][1]),
        ("layout_pipeline", progress_steps[1][1], progress_steps[2][1]),
        ("full_ramp", progress_steps[2][1], progress_steps[3][1]),
        ("full_randomization", progress_steps[3][1], 1.0),
    ]
    out: list[tuple[str, int, int]] = []
    for name, start_frac, end_frac in phase_defs:
        start_step = int(round(total_step * start_frac))
        end_step = int(round(total_step * end_frac))
        out.append((name, start_step, end_step))
    return out


def report_metric_snapshot(
    series_map: dict[str, Series], total_updates: int
) -> list[str]:
    lines: list[str] = []
    smooth_window = max(5, total_updates // 40)
    for tag in DEFAULT_METRICS:
        if tag not in series_map:
            continue
        series = series_map[tag]
        smooth = moving_average(series.values, smooth_window)
        minimize = any(part in tag for part in ("loss", "time_up_rate", "distance"))
        best_idx = int(np.argmin(smooth)) if minimize else int(np.argmax(smooth))
        lines.append(
            f"- {tag}: final={fmt_value(series.values[-1])}, best_smooth={fmt_value(smooth[best_idx])} at step={int(series.steps[best_idx])}"
        )
    return lines


def report_phase_means(
    series_map: dict[str, Series], phases: list[tuple[str, int, int]]
) -> list[str]:
    focus_metrics = (
        "optimizer/learning_rate",
        "env/tag_rate",
        "env/time_up_rate",
        "env/mean_episode_length",
        "chaser/mean_reward",
        "evader/mean_reward",
    )
    lines: list[str] = []
    for tag in focus_metrics:
        if tag not in series_map:
            continue
        series = series_map[tag]
        parts: list[str] = []
        for phase_name, start_step, end_step in phases:
            mask = (series.steps >= start_step) & (series.steps < end_step)
            parts.append(f"{phase_name}={fmt_value(safe_mean(series.values[mask]))}")
        lines.append(f"- {tag}: " + ", ".join(parts))
    return lines


def report_milestone_deltas(
    series_map: dict[str, Series],
    milestone_steps: list[tuple[str, int]],
    window_updates: int,
    timesteps_per_update: int,
) -> list[str]:
    focus_metrics = (
        "optimizer/learning_rate",
        "env/tag_rate",
        "env/time_up_rate",
        "env/mean_episode_length",
        "chaser/mean_reward",
        "evader/mean_reward",
        "chaser/value_loss",
        "evader/value_loss",
    )
    lines: list[str] = []
    window_steps = max(1, window_updates * timesteps_per_update)
    for milestone_name, milestone_step in milestone_steps:
        parts: list[str] = []
        for tag in focus_metrics:
            if tag not in series_map:
                continue
            series = series_map[tag]
            pre_mask = (series.steps >= milestone_step - window_steps) & (
                series.steps < milestone_step
            )
            post_mask = (series.steps >= milestone_step) & (
                series.steps < milestone_step + window_steps
            )
            pre = safe_mean(series.values[pre_mask])
            post = safe_mean(series.values[post_mask])
            if math.isnan(pre) or math.isnan(post):
                continue
            parts.append(f"{tag}={post - pre:+.4f}")
        if parts:
            lines.append(f"- {milestone_name}: " + "; ".join(parts))
    return lines


def report_randomization(series_map: dict[str, Series]) -> list[str]:
    lines: list[str] = []
    for tag in RANDOMIZATION_METRICS:
        if tag not in series_map:
            continue
        series = series_map[tag]
        lines.append(
            f"- {tag}: start={fmt_value(series.values[0])}, mid={fmt_value(series.values[len(series.values) // 2])}, final={fmt_value(series.values[-1])}"
        )
    return lines


def main() -> None:
    args = parse_args()
    run_dir = find_run_dir(args.root.resolve(), args.run_name)
    event_files = find_event_files(run_dir)
    series_map = load_scalars(event_files)
    if not series_map:
        raise ValueError(f"No scalar summaries found in {run_dir}")

    rl_config, env_config, checkpoint_path = load_run_config(run_dir)
    curriculum = get_nested(env_config, "curriculum", default={})

    num_envs = int(rl_config.get("num_envs", 16384))
    num_steps = int(rl_config.get("num_steps", 128))
    total_timesteps = int(rl_config.get("total_timesteps", int(5e9)))
    lr = float(rl_config.get("lr", 3e-4))
    anneal_lr = bool(rl_config.get("anneal_lr", True))
    lr_decay_start_fraction = float(rl_config.get("lr_decay_start_fraction", 0.0))
    final_lr_scale = float(rl_config.get("final_lr_scale", 0.0))
    timesteps_per_update = num_envs * num_steps
    total_updates = max(1, total_timesteps // timesteps_per_update)

    all_steps = max(series_map.values(), key=lambda series: series.steps[-1]).steps
    max_step = int(all_steps[-1])
    observed_updates = max(1, max_step // timesteps_per_update)

    milestones = [
        ("layout_start", float(curriculum.get("layout_start", 0.1))),
        ("pipeline_start", float(curriculum.get("pipeline_start", 0.2))),
        ("dynamics_start", float(curriculum.get("dynamics_start", 0.3))),
        ("full_progress", float(curriculum.get("full_progress", 0.85))),
    ]
    milestone_steps: list[tuple[str, int]] = []

    print(f"Run: {run_dir}")
    print(f"Event files: {len(event_files)}")
    if checkpoint_path is not None:
        print(f"Checkpoint config source: {checkpoint_path}")
    else:
        print("Checkpoint config source: defaults from repo (no checkpoint found)")
    print(f"Scalar tags: {len(series_map)}")
    print(
        "Training config: "
        f"num_envs={num_envs}, num_steps={num_steps}, timesteps_per_update={timesteps_per_update}, "
        f"total_updates={total_updates}, observed_steps={max_step}, observed_updates~={observed_updates}, "
        f"base_lr={lr}, anneal_lr={anneal_lr}, lr_decay_start_fraction={lr_decay_start_fraction}, "
        f"final_lr_scale={final_lr_scale}"
    )
    print()

    print("Milestones")
    for name, progress in milestones:
        update_idx = int(math.ceil(progress * max(total_updates - 1, 1)))
        step = update_idx * timesteps_per_update
        milestone_steps.append((name, step))
        scheduled_lr = scheduled_lr_for_update(
            lr,
            anneal_lr,
            lr_decay_start_fraction,
            final_lr_scale,
            total_updates,
            update_idx,
        )
        lr_scale = scheduled_lr / lr if lr > 0 else float("nan")
        print(
            f"- {name}: progress={progress:.3f}, update={update_idx}, step={step}, "
            f"scheduled_lr~={scheduled_lr:.8f} ({lr_scale * 100:.1f}% of base)"
        )
    print()

    print("Key Metrics")
    for line in report_metric_snapshot(series_map, total_updates):
        print(line)
    print()

    print("Phase Means")
    phases = phase_slices(milestones, all_steps)
    for line in report_phase_means(series_map, phases):
        print(line)
    print()

    print(f"Milestone Deltas (window={args.window_updates} updates)")
    for line in report_milestone_deltas(
        series_map,
        milestone_steps,
        args.window_updates,
        timesteps_per_update,
    ):
        print(line)
    print()

    print("Randomization Metrics")
    for line in report_randomization(series_map):
        print(line)
    print()

    print("Notes")
    print(
        "- Metrics are logged after each PPO update, so milestone effects usually appear on or just after the listed step."
    )
    print(
        "- Means near 1.0 for friction, mass, or motor strength can still hide active randomization; the new `_std` and `_abs_deviation` series show the spread."
    )
    print(
        "- A concerning pattern is sharp degradation after `pipeline_start` or `dynamics_start` with weak recovery before `full_progress`, especially if LR has already decayed materially."
    )


if __name__ == "__main__":
    main()
