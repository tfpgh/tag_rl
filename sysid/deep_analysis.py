from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from environment.actuation import shape_agent_actions
from environment.config import EnvironmentConfig
from environment.randomization import randomize_model
from runtime import jax_setup as _jax_setup  # noqa: F401
from sysid.dataset import WindowConfig, load_dataset_splits
from sysid.evaluator import (
    RAD_TO_DEG,
    SummaryTerms,
    SysIdEvaluator,
    WindowArrays,
    WindowMetrics,
    _safe_masked_mean,
    _wrap_angle,
)
from sysid.fit import _bound_diagnostics
from sysid.params import (
    DEFAULT_PHYSICAL_PARAMS,
    FROZEN_PARAMS,
    PARAM_NAMES,
    build_domain_and_pipeline_from_physical,
    resolve_physical_params,
)
from sysid.types import EvaluationSummary, ScoreBreakdown, WindowBatch

COMMAND_BUCKETS = (
    (0.0, 0.15, "very_low"),
    (0.15, 0.30, "low"),
    (0.30, 0.50, "medium"),
    (0.50, 2.0, "high"),
)
TIME_BUCKETS = (
    (0.0, 0.35, "early"),
    (0.35, 0.90, "mid"),
    (0.90, 10.0, "late"),
)
ACTIVE_GROUPS = {
    "geometry_only": {"track_width_scale", "wheel_radius_scale"},
    "traction_only": {"traction_scale", "turn_scrub_scale"},
    "motor_only": {
        "motor_strength_scale",
        "back_emf_scale",
        "motor_balance",
        "motor_time_constant_seconds",
    },
    "delays_only": {"command_delay_substeps", "observation_delay_substeps"},
}
FROZEN_FAMILY_OVERRIDES = {
    "caster_family": {
        "caster_radius_scale": [0.85, 1.0, 1.15],
        "caster_slide_friction_scale": [0.70, 1.0, 1.30],
        "caster_torsional_friction_scale": [0.70, 1.0, 1.30],
    },
    "contact_family": {
        "body_contact_friction_scale": [0.70, 1.0, 1.30],
        "wheel_slide_friction_scale": [0.70, 1.0, 1.30],
        "wheel_torsional_friction_scale": [0.70, 1.0, 1.30],
        "wheel_rolling_friction_scale": [0.70, 1.0, 1.30],
    },
    "wheel_dynamics_family": {
        "wheel_joint_damping_scale": [0.60, 1.0, 1.40],
        "wheel_joint_frictionloss_scale": [0.60, 1.0, 1.60],
        "wheel_armature_scale": [0.60, 1.0, 1.60],
    },
    "slip_model_family": {
        "motor_curve_sharpness": [0.0, 1.0, 2.5],
        "traction_slip_threshold": [0.70, 1.0, 1.40],
        "traction_loss_strength": [0.0, 0.5, 1.5],
        "traction_min_scale": [0.35, 0.55, 0.80],
    },
    "com_family": {
        "com_offset_x": [-0.010, 0.0, 0.010],
        "com_offset_z": [-0.010, 0.0, 0.010],
    },
}


class PreparedSplit(NamedTuple):
    batch: WindowBatch
    prepared: WindowArrays
    sampled_pose: np.ndarray
    summary: EvaluationSummary
    window_rows: list[dict[str, Any]]
    sample_payload: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run broad sysid diagnostics against recorded data and a chosen parameter set."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("runs/sysid_4015155/history.jsonl"),
    )
    parser.add_argument(
        "--history-source",
        choices=("global_best", "generation_best", "best_params"),
        default="global_best",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/sysid_deep_analysis")
    )
    parser.add_argument("--window-seconds", type=float, default=1.5)
    parser.add_argument("--preroll-seconds", type=float, default=0.5)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--sensitivity-points", type=int, default=7)
    parser.add_argument(
        "--local-fraction",
        type=float,
        default=0.30,
        help="Fraction of each parameter span to sweep around the chosen params.",
    )
    parser.add_argument("--top-k-windows", type=int, default=30)
    parser.add_argument("--top-k-runs", type=int, default=15)
    parser.add_argument(
        "--save-sample-details",
        action="store_true",
        help="Persist large per-sample arrays to NPZ for downstream plotting.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _discover_runs(root: Path) -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []
    for split in ("train", "eval"):
        split_dir = root / split
        if not split_dir.exists():
            continue
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
                discovered.append((split, candidate))
                seen.add(candidate)
    return discovered


def _safe_percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _timing_stats(times: list[float]) -> dict[str, float]:
    if len(times) < 2:
        return {
            "count": float(len(times)),
            "mean_dt": 0.0,
            "median_dt": 0.0,
            "p95_dt": 0.0,
            "max_dt": 0.0,
            "min_dt": 0.0,
        }
    diffs = np.diff(np.asarray(times, dtype=np.float64))
    return {
        "count": float(len(times)),
        "mean_dt": float(np.mean(diffs)),
        "median_dt": float(np.median(diffs)),
        "p95_dt": float(np.percentile(diffs, 95)),
        "max_dt": float(np.max(diffs)),
        "min_dt": float(np.min(diffs)),
    }


def _load_selected_params(path: Path, history_source: str) -> dict[str, float]:
    if not path.exists():
        nearby = (
            sorted(str(item) for item in path.parent.glob("*.json*"))
            if path.parent.exists()
            else []
        )
        raise FileNotFoundError(
            f"Parameter file not found: {path}. Nearby files: {nearby[:10]}"
        )
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"No JSON rows found in {path}")
        payload = rows[-1]
    else:
        payload = json.loads(text)
    if history_source == "best_params" and "best_params" in payload:
        payload = payload["best_params"]
    elif history_source in payload and isinstance(payload[history_source], dict):
        nested = payload[history_source]
        if "params" in nested:
            payload = nested["params"]
        else:
            payload = nested
    elif "best_params" in payload:
        payload = payload["best_params"]
    elif "global_best" in payload and "params" in payload["global_best"]:
        payload = payload["global_best"]["params"]
    elif "generation_best" in payload and "params" in payload["generation_best"]:
        payload = payload["generation_best"]["params"]
    resolved = resolve_physical_params(payload)
    return {key: float(np.asarray(value)) for key, value in resolved.items()}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _audit_runs(data_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate = {
        "runs": 0,
        "samples": 0,
        "commands": 0,
        "modes": Counter(),
        "splits": Counter(),
        "visibility_fraction": [],
        "calibration_valid_fraction": [],
        "sample_dt": [],
        "command_dt": [],
    }
    for split, run_dir in _discover_runs(data_root):
        samples = _load_jsonl(run_dir / "samples.jsonl")
        commands = _load_jsonl(run_dir / "commands.jsonl")
        metadata_path = run_dir / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        collection = (
            metadata.get("collection", {}) if isinstance(metadata, dict) else {}
        )
        mode = (
            collection.get("mode", "unknown")
            if isinstance(collection, dict)
            else "unknown"
        )
        visible_count = sum(
            1
            for sample in samples
            if sample.get("target_robot") is not None
            and sample["target_robot"].get("visible")
        )
        calib_valid_count = sum(
            1 for sample in samples if sample.get("calibration_valid")
        )
        sample_times = [float(sample["monotonic_time"]) for sample in samples]
        command_times = [float(command["monotonic_time"]) for command in commands]
        duration_s = (
            sample_times[-1] - sample_times[0] if len(sample_times) >= 2 else 0.0
        )
        row = {
            "split": split,
            "run_name": run_dir.name,
            "path": str(run_dir),
            "mode": mode,
            "samples": len(samples),
            "commands": len(commands),
            "duration_s": duration_s,
            "visibility_fraction": visible_count / len(samples) if samples else 0.0,
            "calibration_valid_fraction": calib_valid_count / len(samples)
            if samples
            else 0.0,
        }
        row.update(
            {
                f"sample_{key}": value
                for key, value in _timing_stats(sample_times).items()
            }
        )
        row.update(
            {
                f"command_{key}": value
                for key, value in _timing_stats(command_times).items()
            }
        )
        rows.append(row)
        aggregate["runs"] += 1
        aggregate["samples"] += len(samples)
        aggregate["commands"] += len(commands)
        aggregate["modes"][mode] += 1
        aggregate["splits"][split] += 1
        aggregate["visibility_fraction"].append(row["visibility_fraction"])
        aggregate["calibration_valid_fraction"].append(
            row["calibration_valid_fraction"]
        )
        if len(sample_times) >= 2:
            aggregate["sample_dt"].extend(
                np.diff(np.asarray(sample_times, dtype=np.float64)).tolist()
            )
        if len(command_times) >= 2:
            aggregate["command_dt"].extend(
                np.diff(np.asarray(command_times, dtype=np.float64)).tolist()
            )
    summary = {
        "runs": aggregate["runs"],
        "samples": aggregate["samples"],
        "commands": aggregate["commands"],
        "modes": dict(aggregate["modes"]),
        "splits": dict(aggregate["splits"]),
        "mean_visibility_fraction": float(np.mean(aggregate["visibility_fraction"]))
        if aggregate["visibility_fraction"]
        else 0.0,
        "mean_calibration_valid_fraction": float(
            np.mean(aggregate["calibration_valid_fraction"])
        )
        if aggregate["calibration_valid_fraction"]
        else 0.0,
        "sample_dt_p95": _safe_percentile(aggregate["sample_dt"], 95),
        "sample_dt_max": max(aggregate["sample_dt"]) if aggregate["sample_dt"] else 0.0,
        "command_dt_p95": _safe_percentile(aggregate["command_dt"], 95),
        "command_dt_max": max(aggregate["command_dt"])
        if aggregate["command_dt"]
        else 0.0,
    }
    return rows, summary


def _bucket_label(value: float, buckets: Iterable[tuple[float, float, str]]) -> str:
    for low, high, label in buckets:
        if low <= value < high:
            return label
    return "overflow"


def _yaw_deg(angle_rad: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.abs(np.arctan2(np.sin(angle_rad), np.cos(angle_rad))) * 180.0 / np.pi
    )


def _local_xy(xy: np.ndarray, origin: np.ndarray, yaw0: float) -> np.ndarray:
    delta = xy - origin[None, :]
    c = math.cos(yaw0)
    s = math.sin(yaw0)
    return np.stack(
        [c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]], axis=-1
    )


def _compute_pose_rates(
    relative_times: np.ndarray, xy: np.ndarray, yaw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(relative_times) < 2:
        zeros = np.zeros_like(relative_times)
        return zeros, zeros, zeros
    dt = np.gradient(relative_times, edge_order=1)
    vx = np.gradient(xy[:, 0], relative_times, edge_order=1)
    vy = np.gradient(xy[:, 1], relative_times, edge_order=1)
    dyaw = np.gradient(np.unwrap(yaw), relative_times, edge_order=1)
    speed = np.sqrt(vx * vx + vy * vy)
    return speed, dyaw, dt


def _response_metrics(
    relative_times: np.ndarray,
    command_mag: np.ndarray,
    target_speed: np.ndarray,
    sim_speed: np.ndarray,
) -> dict[str, float]:
    command_delta = np.abs(np.diff(command_mag, prepend=command_mag[:1]))
    transition_indices = np.where(command_delta > 0.08)[0]
    if len(transition_indices) == 0:
        return {
            "transition_count": 0.0,
            "transition_speed_mae": 0.0,
            "transition_sim_lag_s": 0.0,
            "transition_target_rise_s": 0.0,
            "transition_sim_rise_s": 0.0,
        }
    speed_maes: list[float] = []
    lag_values: list[float] = []
    target_rises: list[float] = []
    sim_rises: list[float] = []
    for index in transition_indices:
        start_time = relative_times[index]
        mask = (relative_times >= start_time) & (relative_times <= start_time + 0.5)
        if np.count_nonzero(mask) < 3:
            continue
        window_times = relative_times[mask] - start_time
        target_window = target_speed[mask]
        sim_window = sim_speed[mask]
        speed_maes.append(float(np.mean(np.abs(sim_window - target_window))))
        target_peak = float(np.max(np.abs(target_window)))
        sim_peak = float(np.max(np.abs(sim_window)))
        target_thresh = 0.5 * target_peak
        sim_thresh = 0.5 * sim_peak
        target_hit = np.where(np.abs(target_window) >= target_thresh)[0]
        sim_hit = np.where(np.abs(sim_window) >= sim_thresh)[0]
        if len(target_hit) > 0:
            target_rises.append(float(window_times[target_hit[0]]))
        if len(sim_hit) > 0:
            sim_rises.append(float(window_times[sim_hit[0]]))
        if len(target_hit) > 0 and len(sim_hit) > 0:
            lag_values.append(
                float(window_times[sim_hit[0]] - window_times[target_hit[0]])
            )
    return {
        "transition_count": float(len(transition_indices)),
        "transition_speed_mae": float(np.mean(speed_maes)) if speed_maes else 0.0,
        "transition_sim_lag_s": float(np.mean(lag_values)) if lag_values else 0.0,
        "transition_target_rise_s": float(np.mean(target_rises))
        if target_rises
        else 0.0,
        "transition_sim_rise_s": float(np.mean(sim_rises)) if sim_rises else 0.0,
    }


def _window_rows_from_pose(
    split: str,
    batch: WindowBatch,
    sampled_pose: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    sample_arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    for index, metadata in enumerate(batch.metadata):
        valid_mask = batch.sample_mask[index] & batch.score_mask[index]
        if not np.any(valid_mask):
            continue
        rel_times = batch.relative_sample_times_s[index][valid_mask]
        sim = sampled_pose[index][valid_mask]
        target_xy = batch.target_xy[index][valid_mask]
        target_yaw = batch.target_yaw[index][valid_mask]
        pos_err = np.linalg.norm(sim[:, :2] - target_xy, axis=-1)
        yaw_err_rad = np.arctan2(
            np.sin(sim[:, 2] - target_yaw), np.cos(sim[:, 2] - target_yaw)
        )
        yaw_err_deg = _yaw_deg(yaw_err_rad)
        target_local = _local_xy(target_xy, target_xy[0], float(target_yaw[0]))
        sim_local = _local_xy(sim[:, :2], target_xy[0], float(target_yaw[0]))
        local_delta = sim_local - target_local
        forward_err = np.abs(local_delta[:, 0])
        lateral_err = np.abs(local_delta[:, 1])
        target_speed, target_yaw_rate, _ = _compute_pose_rates(
            rel_times, target_xy, target_yaw
        )
        sim_speed, sim_yaw_rate, _ = _compute_pose_rates(
            rel_times, sim[:, :2], sim[:, 2]
        )
        command_rel = np.linspace(
            0.0,
            batch.command_substeps.shape[1] * 0.005,
            batch.command_substeps.shape[1],
            endpoint=False,
            dtype=np.float32,
        )
        cmd = batch.command_substeps[index]
        command_mag = np.max(np.abs(cmd), axis=-1)
        sample_command_mag = np.interp(rel_times, command_rel, command_mag)
        transition_metrics = _response_metrics(
            rel_times, sample_command_mag, target_speed, sim_speed
        )
        early_mask = rel_times <= min(0.5, float(rel_times[-1]))
        late_mask = rel_times >= max(float(rel_times[-1]) - 0.5, 0.0)
        row = {
            "split": split,
            "run_name": metadata.run_name,
            "maneuver": metadata.maneuver,
            "window_index": index,
            "start_time": metadata.start_time,
            "end_time": metadata.end_time,
            "sample_count": metadata.sample_count,
            "command_mean_abs": metadata.command_mean_abs,
            "command_max_abs": metadata.command_max_abs,
            "command_bucket": _bucket_label(metadata.command_mean_abs, COMMAND_BUCKETS),
            "translation_distance_m": metadata.translation_distance_m,
            "yaw_change_rad": metadata.yaw_change_rad,
            "position_mae_m": float(np.mean(pos_err)),
            "position_rmse_m": float(np.sqrt(np.mean(pos_err * pos_err))),
            "yaw_mae_deg": float(np.mean(yaw_err_deg)),
            "endpoint_position_m": float(pos_err[-1]),
            "endpoint_yaw_deg": float(yaw_err_deg[-1]),
            "forward_mae_m": float(np.mean(forward_err)),
            "lateral_mae_m": float(np.mean(lateral_err)),
            "endpoint_forward_err_m": float(forward_err[-1]),
            "endpoint_lateral_err_m": float(lateral_err[-1]),
            "speed_mae_mps": float(np.mean(np.abs(sim_speed - target_speed))),
            "yaw_rate_mae_radps": float(
                np.mean(np.abs(sim_yaw_rate - target_yaw_rate))
            ),
            "early_position_mae_m": float(np.mean(pos_err[early_mask]))
            if np.any(early_mask)
            else 0.0,
            "late_position_mae_m": float(np.mean(pos_err[late_mask]))
            if np.any(late_mask)
            else 0.0,
            "early_yaw_mae_deg": float(np.mean(yaw_err_deg[early_mask]))
            if np.any(early_mask)
            else 0.0,
            "late_yaw_mae_deg": float(np.mean(yaw_err_deg[late_mask]))
            if np.any(late_mask)
            else 0.0,
            "max_position_err_m": float(np.max(pos_err)),
            "max_yaw_err_deg": float(np.max(yaw_err_deg)),
            "score_proxy": float(np.mean(pos_err) / 0.05 + np.mean(yaw_err_deg) / 15.0),
            **transition_metrics,
        }
        rows.append(row)
        sample_arrays["split"].append(np.full(rel_times.shape, index, dtype=np.int32))
        sample_arrays["relative_time_s"].append(rel_times.astype(np.float32))
        sample_arrays["position_err_m"].append(pos_err.astype(np.float32))
        sample_arrays["yaw_err_deg"].append(yaw_err_deg.astype(np.float32))
        sample_arrays["forward_err_m"].append(forward_err.astype(np.float32))
        sample_arrays["lateral_err_m"].append(lateral_err.astype(np.float32))
        sample_arrays["target_speed_mps"].append(target_speed.astype(np.float32))
        sample_arrays["sim_speed_mps"].append(sim_speed.astype(np.float32))
        sample_arrays["target_yaw_rate_radps"].append(
            target_yaw_rate.astype(np.float32)
        )
        sample_arrays["sim_yaw_rate_radps"].append(sim_yaw_rate.astype(np.float32))
        sample_arrays["command_mag"].append(sample_command_mag.astype(np.float32))
    packed = {
        key: np.concatenate(value, axis=0)
        if value
        else np.zeros((0,), dtype=np.float32)
        for key, value in sample_arrays.items()
    }
    return rows, packed


def _group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    summary: list[dict[str, Any]] = []
    for group, items in sorted(grouped.items()):
        summary.append(
            {
                key: group,
                "windows": len(items),
                "position_rmse_m": float(
                    np.mean([item["position_rmse_m"] for item in items])
                ),
                "yaw_mae_deg": float(np.mean([item["yaw_mae_deg"] for item in items])),
                "forward_mae_m": float(
                    np.mean([item["forward_mae_m"] for item in items])
                ),
                "lateral_mae_m": float(
                    np.mean([item["lateral_mae_m"] for item in items])
                ),
                "transition_speed_mae": float(
                    np.mean([item["transition_speed_mae"] for item in items])
                ),
                "score_proxy": float(np.mean([item["score_proxy"] for item in items])),
            }
        )
    return summary


def _physical_param_pytree(params: dict[str, float]) -> dict[str, jax.Array]:
    resolved = resolve_physical_params(params)
    return {
        key: jnp.asarray(value, dtype=jnp.float32) for key, value in resolved.items()
    }


def _build_physical_pose_evaluator(evaluator: SysIdEvaluator):
    def simulate_window(
        physical_params: dict[str, jax.Array], window: WindowArrays
    ) -> jax.Array:
        domain_params, pipeline_params = build_domain_and_pipeline_from_physical(
            physical_params
        )
        model = randomize_model(
            evaluator.env.mjx_model, domain_params, evaluator.env.model_indices
        )
        initial_data = evaluator._initial_data(
            model, window.initial_pose, window.initial_velocity
        )
        initial_buffer = jnp.repeat(
            window.initial_command[None, :], evaluator.action_buffer_len, axis=0
        )

        def substep(carry: tuple[mjx.Data, jax.Array], proposed_action: jax.Array):
            data, action_buffer = carry
            action_buffer = (
                jnp.roll(action_buffer, shift=-1, axis=0).at[-1].set(proposed_action)
            )
            delayed_action = action_buffer[-1 - pipeline_params.action_delay_substeps]
            delayed_action = shape_agent_actions(
                delayed_action,
                data.qpos,
                data.qvel,
                domain_params.chaser,
                evaluator.env.model_indices.chaser,
                evaluator.qpos_slices.chaser_root,
                evaluator.dof_slices.chaser_root,
            )
            data = data.replace(
                ctrl=jnp.concatenate(
                    [delayed_action, jnp.zeros((2,), dtype=jnp.float32)]
                )
            )
            data = mjx.step(model, data)
            return (data, action_buffer), jnp.array(
                [
                    data.qpos[evaluator.qpos_slices.chaser_root][0],
                    data.qpos[evaluator.qpos_slices.chaser_root][1],
                    jnp.arctan2(
                        2.0
                        * (
                            data.qpos[evaluator.qpos_slices.chaser_root][3]
                            * data.qpos[evaluator.qpos_slices.chaser_root][6]
                            + data.qpos[evaluator.qpos_slices.chaser_root][4]
                            * data.qpos[evaluator.qpos_slices.chaser_root][5]
                        ),
                        1.0
                        - 2.0
                        * (
                            data.qpos[evaluator.qpos_slices.chaser_root][5]
                            * data.qpos[evaluator.qpos_slices.chaser_root][5]
                            + data.qpos[evaluator.qpos_slices.chaser_root][6]
                            * data.qpos[evaluator.qpos_slices.chaser_root][6]
                        ),
                    ),
                    data.qvel[evaluator.dof_slices.chaser_root][0],
                    data.qvel[evaluator.dof_slices.chaser_root][1],
                    data.qvel[evaluator.dof_slices.chaser_root][5],
                ],
                dtype=jnp.float32,
            )

        _, pose_history = jax.lax.scan(
            substep,
            (initial_data, initial_buffer),
            window.command_substeps,
            length=evaluator.num_substeps,
        )
        sample_indices = jnp.clip(
            jnp.rint(
                window.relative_sample_times_s / evaluator.substep_dt_seconds
            ).astype(jnp.int32)
            - 1
            - pipeline_params.observation_delay_substeps,
            0,
            evaluator.num_substeps - 1,
        )
        return pose_history[sample_indices]

    return jax.jit(jax.vmap(simulate_window, in_axes=(None, 0)))


def _build_physical_summary_evaluator(evaluator: SysIdEvaluator):
    pose_eval = _build_physical_pose_evaluator(evaluator)

    def summarize_all(
        physical_params: dict[str, jax.Array], windows: WindowArrays
    ) -> SummaryTerms:
        sampled_pose = pose_eval(physical_params, windows)

        def summarize_window(window: WindowArrays, sampled: jax.Array) -> SummaryTerms:
            score = evaluator._score_from_sampled_pose(sampled, window)
            valid_mask = window.sample_mask & window.score_mask
            pos_err = jnp.linalg.norm(sampled[:, :2] - window.target_xy, axis=-1)
            yaw_err = jnp.abs(_wrap_angle(sampled[:, 2] - window.target_yaw))
            last_index = jnp.maximum(jnp.sum(valid_mask.astype(jnp.int32)) - 1, 0)
            metrics = WindowMetrics(
                position_mae_m=_safe_masked_mean(pos_err, valid_mask),
                position_rmse_m=jnp.sqrt(
                    _safe_masked_mean(pos_err * pos_err, valid_mask)
                ),
                yaw_mae_rad=_safe_masked_mean(yaw_err, valid_mask),
                yaw_mae_deg=_safe_masked_mean(yaw_err, valid_mask) * RAD_TO_DEG,
                endpoint_position_m=pos_err[last_index],
                endpoint_yaw_rad=yaw_err[last_index],
                endpoint_yaw_deg=yaw_err[last_index] * RAD_TO_DEG,
            )
            return SummaryTerms(score=score, metrics=metrics)

        return jax.vmap(summarize_window, in_axes=(0, 0))(windows, sampled_pose)

    return jax.jit(summarize_all)


def _summary_from_terms(
    per_window_terms: SummaryTerms, batch: WindowBatch
) -> EvaluationSummary:
    total = np.asarray(jax.device_get(per_window_terms.score.total), dtype=np.float64)
    local = np.asarray(jax.device_get(per_window_terms.score.local), dtype=np.float64)
    increment = np.asarray(
        jax.device_get(per_window_terms.score.increment), dtype=np.float64
    )
    transition = np.asarray(
        jax.device_get(per_window_terms.score.transition), dtype=np.float64
    )
    anchor = np.asarray(jax.device_get(per_window_terms.score.anchor), dtype=np.float64)
    pivot_stability = np.asarray(
        jax.device_get(per_window_terms.score.pivot_stability), dtype=np.float64
    )
    position_mae = np.asarray(
        jax.device_get(per_window_terms.metrics.position_mae_m), dtype=np.float64
    )
    position_rmse = np.asarray(
        jax.device_get(per_window_terms.metrics.position_rmse_m), dtype=np.float64
    )
    yaw_mae_rad = np.asarray(
        jax.device_get(per_window_terms.metrics.yaw_mae_rad), dtype=np.float64
    )
    yaw_mae_deg = np.asarray(
        jax.device_get(per_window_terms.metrics.yaw_mae_deg), dtype=np.float64
    )
    endpoint_position = np.asarray(
        jax.device_get(per_window_terms.metrics.endpoint_position_m), dtype=np.float64
    )
    endpoint_yaw_rad = np.asarray(
        jax.device_get(per_window_terms.metrics.endpoint_yaw_rad), dtype=np.float64
    )
    endpoint_yaw_deg = np.asarray(
        jax.device_get(per_window_terms.metrics.endpoint_yaw_deg), dtype=np.float64
    )
    maneuver_scores: dict[str, float] = {}
    for maneuver in sorted({item.maneuver for item in batch.metadata}):
        indices = [
            idx for idx, item in enumerate(batch.metadata) if item.maneuver == maneuver
        ]
        maneuver_scores[maneuver] = float(total[indices].mean())
    return EvaluationSummary(
        score=ScoreBreakdown(
            total=float(total.mean()),
            local=float(local.mean()),
            increment=float(increment.mean()),
            transition=float(transition.mean()),
            anchor=float(anchor.mean()),
            pivot_stability=float(pivot_stability.mean()),
        ),
        metrics={
            "position_mae_m": float(position_mae.mean()),
            "position_rmse_m": float(position_rmse.mean()),
            "yaw_mae_rad": float(yaw_mae_rad.mean()),
            "yaw_mae_deg": float(yaw_mae_deg.mean()),
            "endpoint_position_m": float(endpoint_position.mean()),
            "endpoint_yaw_rad": float(endpoint_yaw_rad.mean()),
            "endpoint_yaw_deg": float(endpoint_yaw_deg.mean()),
        },
        maneuver_scores=maneuver_scores,
        windows=len(batch.metadata),
    )


def _evaluate_physical_summary(
    summary_eval,
    physical_params: dict[str, float],
    batch: WindowBatch,
    prepared: WindowArrays,
) -> EvaluationSummary:
    per_window_terms = summary_eval(_physical_param_pytree(physical_params), prepared)
    return _summary_from_terms(per_window_terms, batch)


def _sweep_values(
    base: float, low: float, high: float, points: int, local_fraction: float
) -> list[float]:
    span = high - low
    if span <= 0:
        return [base]
    local_radius = 0.5 * span * local_fraction
    sweep_low = max(low, base - local_radius)
    sweep_high = min(high, base + local_radius)
    if sweep_low == sweep_high:
        return [base]
    values = np.linspace(sweep_low, sweep_high, points, dtype=np.float64)
    return [float(value) for value in values]


def _build_active_sensitivity_candidates(
    base_params: dict[str, float],
    points: int,
    local_fraction: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for name in PARAM_NAMES:
        bounds = (
            _bound_diagnostics({name: base_params[name]})["distance_to_bounds"]
            if False
            else None
        )
        del bounds
        spec_low = None
        spec_high = None
        # Read bounds via current resolved params module globals without importing extra structures.
        from sysid.params import PARAM_BOUNDS

        spec_low = PARAM_BOUNDS[name].low
        spec_high = PARAM_BOUNDS[name].high
        for value in _sweep_values(
            base_params[name], spec_low, spec_high, points, local_fraction
        ):
            candidate = dict(base_params)
            candidate[name] = value
            candidates.append(
                {
                    "candidate_type": "active_sensitivity",
                    "label": f"active::{name}::{value:.6g}",
                    "family": "active",
                    "parameter": name,
                    "value": value,
                    "params": candidate,
                }
            )
    return candidates


def _build_counterfactual_candidates(
    base_params: dict[str, float],
) -> list[dict[str, Any]]:
    defaults = dict(DEFAULT_PHYSICAL_PARAMS)
    defaults.update(FROZEN_PARAMS)
    candidates = [
        {
            "candidate_type": "counterfactual",
            "label": "counterfactual::base",
            "family": "baseline",
            "params": dict(base_params),
        },
        {
            "candidate_type": "counterfactual",
            "label": "counterfactual::natural_defaults",
            "family": "baseline",
            "params": dict(defaults),
        },
    ]
    for label, active_names in ACTIVE_GROUPS.items():
        params = dict(defaults)
        for name in active_names:
            params[name] = base_params[name]
        candidates.append(
            {
                "candidate_type": "counterfactual",
                "label": f"counterfactual::{label}",
                "family": label,
                "params": params,
            }
        )
    return candidates


def _build_frozen_family_candidates(
    base_params: dict[str, float],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for family, overrides in FROZEN_FAMILY_OVERRIDES.items():
        keys = list(overrides.keys())
        value_lists = [overrides[key] for key in keys]
        for values in zip(*value_lists, strict=True):
            params = dict(base_params)
            label_bits = []
            for key, raw_value in zip(keys, values, strict=True):
                value = raw_value
                if key in {"motor_curve_sharpness", "com_offset_x", "com_offset_z"}:
                    params[key] = float(value)
                else:
                    params[key] = float(
                        base_params.get(key, DEFAULT_PHYSICAL_PARAMS.get(key, 1.0))
                        * value
                    )
                label_bits.append(f"{key}={params[key]:.5g}")
            candidates.append(
                {
                    "candidate_type": "frozen_family",
                    "label": f"frozen::{family}::{'|'.join(label_bits)}",
                    "family": family,
                    "params": params,
                }
            )
    return candidates


def _evaluate_candidate_set(
    candidates: list[dict[str, Any]],
    train_prepared: PreparedSplit,
    eval_prepared: PreparedSplit,
    physical_summary_eval,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in candidates:
        params = item["params"]
        train_summary = _evaluate_physical_summary(
            physical_summary_eval, params, train_prepared.batch, train_prepared.prepared
        )
        eval_summary = _evaluate_physical_summary(
            physical_summary_eval, params, eval_prepared.batch, eval_prepared.prepared
        )
        rows.append(
            {
                "candidate_type": item["candidate_type"],
                "label": item["label"],
                "family": item.get("family", ""),
                "parameter": item.get("parameter", ""),
                "value": item.get("value", ""),
                "train_total": train_summary.score.total,
                "train_position_rmse_m": train_summary.metrics["position_rmse_m"],
                "train_yaw_mae_deg": train_summary.metrics["yaw_mae_deg"],
                "eval_total": eval_summary.score.total,
                "eval_local": eval_summary.score.local,
                "eval_increment": eval_summary.score.increment,
                "eval_transition": eval_summary.score.transition,
                "eval_anchor": eval_summary.score.anchor,
                "eval_pivot_stability": eval_summary.score.pivot_stability,
                "eval_position_rmse_m": eval_summary.metrics["position_rmse_m"],
                "eval_position_mae_m": eval_summary.metrics["position_mae_m"],
                "eval_yaw_mae_deg": eval_summary.metrics["yaw_mae_deg"],
                "eval_endpoint_position_m": eval_summary.metrics["endpoint_position_m"],
                "eval_endpoint_yaw_deg": eval_summary.metrics["endpoint_yaw_deg"],
            }
        )
    return rows


def _objective_audit(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidate_rows:
        return {}
    objectives = {
        "eval_total": True,
        "eval_position_rmse_m": True,
        "eval_yaw_mae_deg": True,
        "eval_endpoint_position_m": True,
        "eval_endpoint_yaw_deg": True,
    }
    out: dict[str, Any] = {}
    for key in objectives:
        best = sorted(candidate_rows, key=lambda row: float(row[key]))[:10]
        out[key] = [
            {
                "label": row["label"],
                "family": row["family"],
                key: row[key],
                "eval_total": row["eval_total"],
                "eval_position_rmse_m": row["eval_position_rmse_m"],
                "eval_yaw_mae_deg": row["eval_yaw_mae_deg"],
            }
            for row in best
        ]
    return out


def _top_windows(
    rows: list[dict[str, Any]], key: str, top_k: int
) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row[key]), reverse=True)[:top_k]


def _best_windows(
    rows: list[dict[str, Any]], key: str, top_k: int
) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row[key]))[:top_k]


def _prepare_split(
    split: str,
    batch: WindowBatch,
    evaluator: SysIdEvaluator,
    physical_pose_eval,
    physical_summary_eval,
    base_params: dict[str, float],
) -> PreparedSplit:
    prepared = evaluator.prepare_batch(batch)
    physical_pytree = _physical_param_pytree(base_params)
    sampled_pose = np.asarray(
        jax.device_get(physical_pose_eval(physical_pytree, prepared))
    )
    summary = _evaluate_physical_summary(
        physical_summary_eval, base_params, batch, prepared
    )
    window_rows, sample_payload = _window_rows_from_pose(split, batch, sampled_pose)
    return PreparedSplit(
        batch=batch,
        prepared=prepared,
        sampled_pose=sampled_pose,
        summary=summary,
        window_rows=window_rows,
        sample_payload=sample_payload,
    )


def _merge_split_rows(
    train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [*train_rows, *eval_rows]


def _write_markdown_summary(
    path: Path,
    params_path: Path,
    params: dict[str, float],
    train_split: PreparedSplit,
    eval_split: PreparedSplit,
    dataset_summary: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    frozen_candidates: list[dict[str, Any]],
) -> None:
    eval_rows = eval_split.window_rows
    maneuver_summary = _group_summary(eval_rows, "maneuver")
    command_summary = _group_summary(eval_rows, "command_bucket")
    run_summary = _group_summary(eval_rows, "run_name")
    worst_maneuver = (
        max(maneuver_summary, key=lambda row: row["score_proxy"])
        if maneuver_summary
        else None
    )
    worst_command = (
        max(command_summary, key=lambda row: row["score_proxy"])
        if command_summary
        else None
    )
    best_frozen = None
    frozen_rows = [
        row for row in candidate_rows if row["candidate_type"] == "frozen_family"
    ]
    if frozen_rows:
        best_frozen = min(frozen_rows, key=lambda row: float(row["eval_total"]))
    lines = [
        "# SysID Deep Analysis",
        "",
        f"- Params source: `{params_path}`",
        f"- Eval total: `{eval_split.summary.score.total:.4f}`",
        f"- Eval position RMSE: `{eval_split.summary.metrics['position_rmse_m']:.4f} m`",
        f"- Eval yaw MAE: `{eval_split.summary.metrics['yaw_mae_deg']:.2f} deg`",
        f"- Train total: `{train_split.summary.score.total:.4f}`",
        f"- Dataset runs: `{dataset_summary['runs']}`",
        "",
        "## High-Level Reads",
        "",
    ]
    if worst_maneuver is not None:
        lines.append(
            f"- Worst maneuver bucket on eval: `{worst_maneuver['maneuver']}` with score proxy `{worst_maneuver['score_proxy']:.3f}`, RMSE `{worst_maneuver['position_rmse_m']:.3f} m`, yaw `{worst_maneuver['yaw_mae_deg']:.2f} deg`."
        )
    if worst_command is not None:
        lines.append(
            f"- Worst command bucket on eval: `{worst_command['command_bucket']}` with score proxy `{worst_command['score_proxy']:.3f}`, forward error `{worst_command['forward_mae_m']:.3f} m`, lateral error `{worst_command['lateral_mae_m']:.3f} m`."
        )
    lines.extend(
        [
            f"- Mean eval early position MAE is `{float(np.mean([row['early_position_mae_m'] for row in eval_rows])):.3f} m`; late MAE is `{float(np.mean([row['late_position_mae_m'] for row in eval_rows])):.3f} m`. This helps separate transient mismatch from drift.",
            f"- Mean eval transition speed MAE is `{float(np.mean([row['transition_speed_mae'] for row in eval_rows])):.3f} m/s` and mean simulated lag is `{float(np.mean([row['transition_sim_lag_s'] for row in eval_rows])):.3f} s`.",
        ]
    )
    if best_frozen is not None:
        improvement = eval_split.summary.score.total - float(best_frozen["eval_total"])
        lines.append(
            f"- Best frozen-family counterfactual is `{best_frozen['label']}` with eval total `{best_frozen['eval_total']:.4f}` (improvement `{improvement:.4f}`). Large improvement here suggests the current compact sysid fit is bottlenecked by frozen structure rather than active params alone."
        )
    lines.extend(
        [
            "",
            "## Assumptions To Question",
            "",
            "- Delay is modeled only as integer command/observation substeps. If transition lag dominates and delay-only candidates help a lot, that assumption is likely too crude.",
            "- The evaluator uses a single active robot with the evader parked and no obstacles. If worst runs are mostly game data with richer contacts, the model family may be missing important interaction structure.",
            "- Slip/contact terms are mostly frozen. If pivot/high-command buckets remain much worse than straight buckets, or frozen-family contact/slip tests help materially, contact structure is a likely bottleneck.",
            "- Motor nonlinearity is mostly frozen. If early-window and launch mismatches dominate, actuator compression or transient structure may be under-modeled.",
            "- If local one-at-a-time sweeps are flat around the chosen params and no parameter is near bounds, the current physics family may really be saturated for this data/objective.",
            "",
            "## Top Problem Runs",
            "",
        ]
    )
    for item in sorted(run_summary, key=lambda row: row["score_proxy"], reverse=True)[
        :10
    ]:
        lines.append(
            f"- `{item['run_name']}`: score proxy `{item['score_proxy']:.3f}`, RMSE `{item['position_rmse_m']:.3f} m`, yaw `{item['yaw_mae_deg']:.2f} deg`."
        )
    lines.extend(["", "## Parameter Snapshot", ""])
    for key in sorted(params):
        lines.append(f"- `{key}` = `{params[key]:.6g}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}

    start = time.perf_counter()
    params = _load_selected_params(args.params, args.history_source)
    timing["load_params_s"] = time.perf_counter() - start

    start = time.perf_counter()
    run_rows, dataset_summary = _audit_runs(args.data_root)
    timing["audit_runs_s"] = time.perf_counter() - start

    start = time.perf_counter()
    window_config = WindowConfig(
        window_seconds=args.window_seconds,
        preroll_seconds=args.preroll_seconds,
        stride_seconds=args.stride_seconds,
    )
    dataset = load_dataset_splits(args.data_root, window_config)
    evaluator = SysIdEvaluator.create(EnvironmentConfig(), window_config.num_substeps)
    physical_pose_eval = _build_physical_pose_evaluator(evaluator)
    physical_summary_eval = _build_physical_summary_evaluator(evaluator)
    timing["load_dataset_and_compile_s"] = time.perf_counter() - start

    start = time.perf_counter()
    train_split = _prepare_split(
        "train",
        dataset.train,
        evaluator,
        physical_pose_eval,
        physical_summary_eval,
        params,
    )
    eval_split = _prepare_split(
        "eval",
        dataset.eval,
        evaluator,
        physical_pose_eval,
        physical_summary_eval,
        params,
    )
    timing["baseline_eval_s"] = time.perf_counter() - start

    start = time.perf_counter()
    active_candidates = _build_active_sensitivity_candidates(
        params, args.sensitivity_points, args.local_fraction
    )
    counterfactual_candidates = _build_counterfactual_candidates(params)
    frozen_candidates = _build_frozen_family_candidates(params)
    candidate_rows = _evaluate_candidate_set(
        [*counterfactual_candidates, *active_candidates, *frozen_candidates],
        train_split,
        eval_split,
        physical_summary_eval,
    )
    timing["candidate_evals_s"] = time.perf_counter() - start

    all_window_rows = _merge_split_rows(train_split.window_rows, eval_split.window_rows)
    output_run_rows = sorted(run_rows, key=lambda row: row["duration_s"], reverse=True)
    output_window_rows = sorted(
        all_window_rows, key=lambda row: (row["split"], row["window_index"])
    )
    group_tables = {
        "by_maneuver": {
            "train": _group_summary(train_split.window_rows, "maneuver"),
            "eval": _group_summary(eval_split.window_rows, "maneuver"),
        },
        "by_command_bucket": {
            "train": _group_summary(train_split.window_rows, "command_bucket"),
            "eval": _group_summary(eval_split.window_rows, "command_bucket"),
        },
        "by_run": {
            "train": _group_summary(train_split.window_rows, "run_name"),
            "eval": _group_summary(eval_split.window_rows, "run_name"),
        },
    }
    objective_audit = _objective_audit(candidate_rows)
    worst_windows = {
        "eval_total_proxy": _top_windows(
            eval_split.window_rows, "score_proxy", args.top_k_windows
        ),
        "eval_position_rmse": _top_windows(
            eval_split.window_rows, "position_rmse_m", args.top_k_windows
        ),
        "eval_yaw_mae": _top_windows(
            eval_split.window_rows, "yaw_mae_deg", args.top_k_windows
        ),
        "eval_transition": _top_windows(
            eval_split.window_rows, "transition_speed_mae", args.top_k_windows
        ),
    }
    best_windows = {
        "eval_total_proxy": _best_windows(
            eval_split.window_rows, "score_proxy", args.top_k_windows
        ),
        "eval_position_rmse": _best_windows(
            eval_split.window_rows, "position_rmse_m", args.top_k_windows
        ),
    }
    top_runs = sorted(
        group_tables["by_run"]["eval"], key=lambda row: row["score_proxy"], reverse=True
    )[: args.top_k_runs]

    report = {
        "params_path": str(args.params),
        "history_source": args.history_source,
        "window_config": asdict(window_config),
        "timing": timing,
        "dataset_summary": dataset_summary,
        "bound_diagnostics": _bound_diagnostics(
            {key: params[key] for key in PARAM_NAMES}
        ),
        "train_summary": {
            "score": asdict(train_split.summary.score),
            "metrics": train_split.summary.metrics,
            "maneuver_scores": train_split.summary.maneuver_scores,
            "windows": train_split.summary.windows,
        },
        "eval_summary": {
            "score": asdict(eval_split.summary.score),
            "metrics": eval_split.summary.metrics,
            "maneuver_scores": eval_split.summary.maneuver_scores,
            "windows": eval_split.summary.windows,
        },
        "group_tables": group_tables,
        "objective_audit": objective_audit,
        "top_problem_runs": top_runs,
    }

    _write_json(args.output_dir / "report.json", report)
    _write_json(
        args.output_dir / "params.json",
        {"params": params, "frozen_defaults": FROZEN_PARAMS},
    )
    _write_json(args.output_dir / "dataset_summary.json", dataset_summary)
    _write_json(args.output_dir / "group_tables.json", group_tables)
    _write_json(args.output_dir / "objective_audit.json", objective_audit)
    _write_json(args.output_dir / "worst_windows.json", worst_windows)
    _write_json(args.output_dir / "best_windows.json", best_windows)
    _write_csv(args.output_dir / "run_summary.csv", output_run_rows)
    _write_csv(args.output_dir / "window_metrics.csv", output_window_rows)
    _write_csv(args.output_dir / "candidate_summary.csv", candidate_rows)
    _write_markdown_summary(
        args.output_dir / "summary.md",
        args.params,
        params,
        train_split,
        eval_split,
        dataset_summary,
        candidate_rows,
        frozen_candidates,
    )
    if args.save_sample_details:
        np.savez_compressed(
            args.output_dir / "train_sample_details.npz", **train_split.sample_payload
        )
        np.savez_compressed(
            args.output_dir / "eval_sample_details.npz", **eval_split.sample_payload
        )

    print(
        "deep_analysis complete: "
        f"eval_total={eval_split.summary.score.total:.4f} "
        f"eval_rmse={eval_split.summary.metrics['position_rmse_m']:.4f}m "
        f"eval_yaw={eval_split.summary.metrics['yaw_mae_deg']:.2f}deg "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
