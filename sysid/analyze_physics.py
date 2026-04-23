from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

from runtime import jax_setup as _jax_setup  # noqa: F401

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from environment.actuation import shape_agent_actions_with_diagnostics
from environment.config import EnvironmentConfig
from environment.randomization import randomize_model
from sysid.dataset import WindowConfig, load_dataset_splits
from sysid.evaluator import SysIdEvaluator
from sysid.params import (
    DEFAULT_PHYSICAL_PARAMS,
    PARAM_NAMES,
    build_domain_and_pipeline,
    physical_to_latent,
    resolve_physical_params,
)
from sysid.types import WindowBatch

PLOT_W = 1280
PLOT_H = 720
PLOT_BG = (250, 250, 250)
PLOT_FG = (40, 40, 40)
PLOT_GRID = (220, 220, 220)
PLOT_BLUE = (219, 132, 52)
PLOT_RED = (72, 85, 221)
PLOT_GREEN = (70, 150, 70)
PLOT_ORANGE = (50, 155, 235)


class DiagnosticTerms(NamedTuple):
    traction_active_fraction: jax.Array
    traction_mean_scale: jax.Array
    traction_min_scale: jax.Array
    launch_traction_active_fraction: jax.Array
    launch_traction_mean_scale: jax.Array
    launch_traction_min_scale: jax.Array
    slip_mean: jax.Array
    slip_max: jax.Array
    motor_curve_mean_compression: jax.Array
    motor_curve_max_compression: jax.Array
    traction_mean_attenuation: jax.Array
    traction_max_attenuation: jax.Array
    wheel_speed_mismatch_mean: jax.Array
    wheel_speed_mismatch_max: jax.Array


def _load_params(path: Path, history_source: str) -> dict[str, float]:
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.suffix != ".jsonl"
        else None
    )
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"No rows found in {path}")
        if history_source == "last_generation":
            payload = rows[-1]["generation_best"]["params"]
        elif history_source == "best_train":
            payload = min(
                rows,
                key=lambda row: row["generation_best"]["train_summary"]["score"][
                    "total"
                ],
            )["generation_best"]["params"]
        elif history_source == "best_eval_generation":
            payload = min(
                rows,
                key=lambda row: row["generation_best"]["eval_summary"]["score"][
                    "total"
                ],
            )["generation_best"]["params"]
        elif history_source == "global_best":
            payload = rows[-1]["global_best"]["params"]
        else:
            raise ValueError(f"Unsupported history source: {history_source}")
    else:
        if "best_params" in payload:
            payload = payload["best_params"]
        elif "global_best" in payload and "params" in payload["global_best"]:
            payload = payload["global_best"]["params"]
        elif "generation_best" in payload and "params" in payload["generation_best"]:
            payload = payload["generation_best"]["params"]
    return resolve_physical_params(payload)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _median_dt(times: np.ndarray) -> float:
    if times.size < 2:
        return 0.0
    diffs = np.diff(times)
    diffs = diffs[diffs > 1e-9]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _best_lag_seconds(
    score_relative_times_s: np.ndarray,
    pos_err: np.ndarray,
    yaw_err_deg: np.ndarray,
    target_xy: np.ndarray,
    target_yaw: np.ndarray,
    sim_xy: np.ndarray,
    sim_yaw: np.ndarray,
) -> float:
    valid_len = len(score_relative_times_s)
    if valid_len < 5:
        return 0.0
    best_shift = 0
    best_score = float(np.mean(pos_err + 0.01 * yaw_err_deg))
    for shift in range(-8, 9):
        if shift == 0:
            continue
        if shift > 0:
            lhs = slice(shift, valid_len)
            rhs = slice(0, valid_len - shift)
        else:
            lhs = slice(0, valid_len + shift)
            rhs = slice(-shift, valid_len)
        if lhs.stop - lhs.start < 5:
            continue
        shifted_pos = np.linalg.norm(sim_xy[lhs] - target_xy[rhs], axis=-1)
        shifted_yaw = np.abs(_wrap_angle(sim_yaw[lhs] - target_yaw[rhs])) * (
            180.0 / np.pi
        )
        score = float(np.mean(shifted_pos + 0.01 * shifted_yaw))
        if score < best_score:
            best_score = score
            best_shift = shift
    dt = _median_dt(score_relative_times_s)
    return float(best_shift * dt)


def _forward_velocity_from_pose(
    xy: np.ndarray, yaw: np.ndarray, times: np.ndarray
) -> np.ndarray:
    if len(times) < 2:
        return np.zeros_like(times)
    vx = np.gradient(xy[:, 0], times, edge_order=1)
    vy = np.gradient(xy[:, 1], times, edge_order=1)
    return vx * np.cos(yaw) + vy * np.sin(yaw)


def _masked_time_bin_means(
    time_values: list[np.ndarray],
    metric_values: list[np.ndarray],
    bins: np.ndarray,
) -> list[float]:
    out: list[float] = []
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        bucket: list[np.ndarray] = []
        for times, values in zip(time_values, metric_values, strict=True):
            mask = (times >= low) & (times < high)
            if np.any(mask):
                bucket.append(values[mask])
        if not bucket:
            out.append(0.0)
            continue
        concatenated = np.concatenate(bucket)
        out.append(float(np.mean(concatenated)))
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _draw_axes(
    image: np.ndarray, title: str, x_label: str, y_label: str
) -> tuple[int, int, int, int]:
    left = 110
    top = 70
    right = PLOT_W - 40
    bottom = PLOT_H - 90
    cv2.putText(
        image, title, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, PLOT_FG, 2, cv2.LINE_AA
    )
    cv2.line(image, (left, top), (left, bottom), PLOT_FG, 2)
    cv2.line(image, (left, bottom), (right, bottom), PLOT_FG, 2)
    cv2.putText(
        image,
        x_label,
        (right - 220, PLOT_H - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        PLOT_FG,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        y_label,
        (30, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        PLOT_FG,
        2,
        cv2.LINE_AA,
    )
    for idx in range(1, 6):
        y = int(round(top + (bottom - top) * idx / 5))
        cv2.line(image, (left, y), (right, y), PLOT_GRID, 1)
    return left, top, right, bottom


def _save_bar_chart(
    path: Path, title: str, items: list[tuple[str, float]], color: tuple[int, int, int]
) -> None:
    image = np.full((PLOT_H, PLOT_W, 3), PLOT_BG, dtype=np.uint8)
    left, top, right, bottom = _draw_axes(image, title, "Category", "Value")
    if not items:
        cv2.imwrite(str(path), image)
        return
    max_value = max(value for _, value in items)
    max_value = max(max_value, 1e-6)
    bar_width = max((right - left - 20) // len(items) - 20, 20)
    for idx, (label, value) in enumerate(items):
        x0 = left + 20 + idx * (bar_width + 20)
        x1 = x0 + bar_width
        y1 = bottom - 1
        y0 = int(round(bottom - (value / max_value) * (bottom - top - 20)))
        cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)
        cv2.putText(
            image,
            f"{value:.3f}",
            (x0, max(y0 - 8, top + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            PLOT_FG,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label[:14],
            (x0 - 6, bottom + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            PLOT_FG,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), image)


def _save_scatter_plot(
    path: Path,
    title: str,
    xs: np.ndarray,
    ys: np.ndarray,
    colors: np.ndarray,
    x_label: str,
    y_label: str,
) -> None:
    image = np.full((PLOT_H, PLOT_W, 3), PLOT_BG, dtype=np.uint8)
    left, top, right, bottom = _draw_axes(image, title, x_label, y_label)
    if xs.size == 0:
        cv2.imwrite(str(path), image)
        return
    x_min = float(xs.min())
    x_max = float(xs.max())
    y_min = float(ys.min())
    y_max = float(ys.max())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    for x, y, is_launch in zip(xs, ys, colors, strict=True):
        px = int(round(left + ((float(x) - x_min) / x_span) * (right - left)))
        py = int(round(bottom - ((float(y) - y_min) / y_span) * (bottom - top)))
        cv2.circle(
            image, (px, py), 4, PLOT_RED if is_launch else PLOT_BLUE, -1, cv2.LINE_AA
        )
    cv2.putText(
        image,
        "launch-heavy",
        (right - 220, top + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        PLOT_RED,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "other",
        (right - 220, top + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        PLOT_BLUE,
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), image)


def _save_line_plot(
    path: Path,
    title: str,
    x_values: np.ndarray,
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    x_label: str,
    y_label: str,
) -> None:
    image = np.full((PLOT_H, PLOT_W, 3), PLOT_BG, dtype=np.uint8)
    left, top, right, bottom = _draw_axes(image, title, x_label, y_label)
    if x_values.size == 0 or not series:
        cv2.imwrite(str(path), image)
        return
    y_max = max(max(values) for _, values, _ in series)
    y_max = max(y_max, 1e-6)
    x_min = float(x_values[0])
    x_max = float(x_values[-1])
    x_span = max(x_max - x_min, 1e-6)
    for idx, (label, values, color) in enumerate(series):
        pts: list[tuple[int, int]] = []
        for x, y in zip(x_values, values, strict=True):
            px = int(round(left + ((float(x) - x_min) / x_span) * (right - left)))
            py = int(round(bottom - (float(y) / y_max) * (bottom - top)))
            pts.append((px, py))
        if len(pts) >= 2:
            cv2.polylines(
                image,
                [np.asarray(pts, dtype=np.int32)],
                isClosed=False,
                color=color,
                thickness=2,
            )
        cv2.putText(
            image,
            label,
            (right - 250, top + 24 + idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), image)


def _launch_mask(
    batch: WindowBatch, index: int, score_start_time: float, launch_duration_s: float
) -> np.ndarray:
    substep_dt = 0.005
    substep_times = (
        np.arange(batch.command_substeps.shape[1], dtype=np.float32) * substep_dt
    )
    return (substep_times >= score_start_time) & (
        substep_times < score_start_time + launch_duration_s
    )


def _build_diagnostic_evaluator(evaluator: SysIdEvaluator, launch_duration_s: float):
    substep_times = (
        jnp.arange(evaluator.num_substeps, dtype=jnp.float32)
        * evaluator.substep_dt_seconds
    )

    def diagnose_window(latent: jax.Array, window) -> DiagnosticTerms:
        domain_params, pipeline_params = build_domain_and_pipeline(latent)
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
            (
                final_action,
                motor_action,
                traction_scale,
                slip_ratio,
                wheel_ground_speed,
                wheel_surface_speed,
            ) = shape_agent_actions_with_diagnostics(
                delayed_action,
                data.qpos,
                data.qvel,
                domain_params.chaser,
                evaluator.env.model_indices.chaser,
                evaluator.qpos_slices.chaser_root,
                evaluator.dof_slices.chaser_root,
            )
            data = data.replace(
                ctrl=jnp.concatenate([final_action, jnp.zeros((2,), dtype=jnp.float32)])
            )
            data = mjx.step(model, data)
            slip_active = jnp.any(traction_scale < 0.999, axis=-1).astype(jnp.float32)
            motor_compression = jnp.mean(
                jnp.maximum(jnp.abs(delayed_action) - jnp.abs(motor_action), 0.0)
            )
            traction_attenuation = jnp.mean(
                jnp.maximum(jnp.abs(motor_action) - jnp.abs(final_action), 0.0)
            )
            wheel_mismatch = jnp.max(jnp.abs(wheel_surface_speed - wheel_ground_speed))
            stats = jnp.array(
                [
                    slip_active,
                    jnp.mean(traction_scale),
                    jnp.min(traction_scale),
                    jnp.mean(slip_ratio),
                    jnp.max(slip_ratio),
                    motor_compression,
                    jnp.max(
                        jnp.maximum(
                            jnp.abs(delayed_action) - jnp.abs(motor_action), 0.0
                        )
                    ),
                    traction_attenuation,
                    jnp.max(
                        jnp.maximum(jnp.abs(motor_action) - jnp.abs(final_action), 0.0)
                    ),
                    wheel_mismatch,
                ],
                dtype=jnp.float32,
            )
            return (data, action_buffer), stats

        _, stats = jax.lax.scan(
            substep,
            (initial_data, initial_buffer),
            window.command_substeps,
            length=evaluator.num_substeps,
        )
        score_start_time = jnp.min(
            jnp.where(window.score_mask, window.relative_sample_times_s, jnp.inf)
        )
        launch_mask = (substep_times >= score_start_time) & (
            substep_times < score_start_time + launch_duration_s
        )

        def masked_mean(column: jax.Array, mask: jax.Array) -> jax.Array:
            weights = mask.astype(jnp.float32)
            return jnp.sum(column * weights) / jnp.maximum(jnp.sum(weights), 1.0)

        def masked_min(column: jax.Array, mask: jax.Array) -> jax.Array:
            return jnp.min(jnp.where(mask, column, jnp.inf))

        return DiagnosticTerms(
            traction_active_fraction=jnp.mean(stats[:, 0]),
            traction_mean_scale=jnp.mean(stats[:, 1]),
            traction_min_scale=jnp.min(stats[:, 2]),
            launch_traction_active_fraction=masked_mean(stats[:, 0], launch_mask),
            launch_traction_mean_scale=masked_mean(stats[:, 1], launch_mask),
            launch_traction_min_scale=masked_min(stats[:, 2], launch_mask),
            slip_mean=jnp.mean(stats[:, 3]),
            slip_max=jnp.max(stats[:, 4]),
            motor_curve_mean_compression=jnp.mean(stats[:, 5]),
            motor_curve_max_compression=jnp.max(stats[:, 6]),
            traction_mean_attenuation=jnp.mean(stats[:, 7]),
            traction_max_attenuation=jnp.max(stats[:, 8]),
            wheel_speed_mismatch_mean=jnp.mean(stats[:, 9]),
            wheel_speed_mismatch_max=jnp.max(stats[:, 9]),
        )

    return jax.jit(jax.vmap(diagnose_window, in_axes=(None, 0)))


def _to_float_dict(named_tuple: Any) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for field in named_tuple._fields:
        out[field] = (
            np.asarray(jax.device_get(getattr(named_tuple, field)))
            .astype(np.float64)
            .tolist()
        )
    return out


def _analyze_split(
    split_name: str,
    batch: WindowBatch,
    evaluator: SysIdEvaluator,
    latent: jax.Array,
    output_dir: Path,
    top_k: int,
    launch_duration_s: float,
    launch_step_threshold: float,
) -> dict[str, Any]:
    prepared = evaluator.prepare_batch(batch)
    summary_eval = evaluator.build_summary_evaluator(prepared)
    pose_eval = jax.jit(jax.vmap(evaluator._simulate_pose_history, in_axes=(None, 0)))
    diagnostic_eval = _build_diagnostic_evaluator(evaluator, launch_duration_s)

    per_window_terms = summary_eval(latent, prepared)
    sampled_pose = np.asarray(jax.device_get(pose_eval(latent, prepared)))
    diagnostics = diagnostic_eval(latent, prepared)

    score_arrays = _to_float_dict(per_window_terms.score)
    metric_arrays = _to_float_dict(per_window_terms.metrics)
    diagnostic_arrays = _to_float_dict(diagnostics)

    rows: list[dict[str, Any]] = []
    per_window_times_all: list[np.ndarray] = []
    per_window_pos_all: list[np.ndarray] = []
    per_window_yaw_all: list[np.ndarray] = []
    per_window_times_launch: list[np.ndarray] = []
    per_window_pos_launch: list[np.ndarray] = []
    per_window_yaw_launch: list[np.ndarray] = []

    for index, metadata in enumerate(batch.metadata):
        valid_mask = batch.sample_mask[index] & batch.score_mask[index]
        if not np.any(valid_mask):
            continue
        rel_times = batch.relative_sample_times_s[index][valid_mask]
        score_start_time = float(rel_times[0])
        score_rel_times = rel_times - score_start_time
        sim = sampled_pose[index][valid_mask]
        target_xy = batch.target_xy[index][valid_mask]
        target_yaw = batch.target_yaw[index][valid_mask]
        pos_err = np.linalg.norm(sim[:, :2] - target_xy, axis=-1)
        yaw_err_deg = np.abs(_wrap_angle(sim[:, 2] - target_yaw)) * (180.0 / np.pi)

        early_mask = score_rel_times <= min(1.0, float(score_rel_times[-1]))
        late_threshold = max(float(score_rel_times[-1]) - 1.0, 0.0)
        late_mask = score_rel_times >= late_threshold
        target_forward_velocity = _forward_velocity_from_pose(
            target_xy, target_yaw, rel_times
        )
        sim_forward_velocity = sim[:, 3] * np.cos(sim[:, 2]) + sim[:, 4] * np.sin(
            sim[:, 2]
        )
        target_forward_accel = (
            np.gradient(target_forward_velocity, rel_times, edge_order=1)
            if len(rel_times) >= 2
            else np.zeros_like(rel_times)
        )
        sim_forward_accel = (
            np.gradient(sim_forward_velocity, rel_times, edge_order=1)
            if len(rel_times) >= 2
            else np.zeros_like(rel_times)
        )
        accel_window_mask = score_rel_times <= min(0.75, float(score_rel_times[-1]))

        commands = batch.command_substeps[index]
        command_mag = np.max(np.abs(commands), axis=-1)
        command_steps = (
            np.linalg.norm(commands[1:] - commands[:-1], axis=-1)
            if len(commands) > 1
            else np.zeros((0,), dtype=np.float32)
        )
        launch_substep_mask = _launch_mask(
            batch, index, score_start_time, launch_duration_s
        )
        launch_step_max = float(
            np.max(command_steps[launch_substep_mask[1:]])
            if np.any(launch_substep_mask[1:]) and command_steps.size
            else 0.0
        )
        launch_window = bool(launch_step_max >= launch_step_threshold)
        near_sat_fraction = float(np.mean(command_mag >= 0.9))

        per_window_times_all.append(score_rel_times)
        per_window_pos_all.append(pos_err)
        per_window_yaw_all.append(yaw_err_deg)
        if launch_window:
            per_window_times_launch.append(score_rel_times)
            per_window_pos_launch.append(pos_err)
            per_window_yaw_launch.append(yaw_err_deg)

        row = {
            "split": split_name,
            "run_name": metadata.run_name,
            "window_index": index,
            "start_time": metadata.start_time,
            "score_start_time": metadata.score_start_time,
            "end_time": metadata.end_time,
            "maneuver": metadata.maneuver,
            "sample_count": metadata.sample_count,
            "command_mean_abs": metadata.command_mean_abs,
            "command_max_abs": metadata.command_max_abs,
            "command_step_mean": float(np.mean(command_steps))
            if command_steps.size
            else 0.0,
            "command_step_max": float(np.max(command_steps))
            if command_steps.size
            else 0.0,
            "launch_step_max": launch_step_max,
            "near_saturation_fraction": near_sat_fraction,
            "translation_distance_m": metadata.translation_distance_m,
            "yaw_change_rad": metadata.yaw_change_rad,
            "launch_window": launch_window,
            "score_total": score_arrays["total"][index],
            "score_local": score_arrays["local"][index],
            "score_increment": score_arrays["increment"][index],
            "score_transition": score_arrays["transition"][index],
            "score_anchor": score_arrays["anchor"][index],
            "score_pivot_stability": score_arrays["pivot_stability"][index],
            "position_mae_m": metric_arrays["position_mae_m"][index],
            "position_rmse_m": metric_arrays["position_rmse_m"][index],
            "yaw_mae_deg": metric_arrays["yaw_mae_deg"][index],
            "endpoint_position_m": metric_arrays["endpoint_position_m"][index],
            "endpoint_yaw_deg": metric_arrays["endpoint_yaw_deg"][index],
            "early_position_mae_m": _mean(pos_err[early_mask]),
            "late_position_mae_m": _mean(pos_err[late_mask]),
            "early_yaw_mae_deg": _mean(yaw_err_deg[early_mask]),
            "late_yaw_mae_deg": _mean(yaw_err_deg[late_mask]),
            "forward_velocity_mae_mps": _mean(
                np.abs(sim_forward_velocity - target_forward_velocity)
            ),
            "launch_forward_velocity_mae_mps": _mean(
                np.abs(
                    sim_forward_velocity[accel_window_mask]
                    - target_forward_velocity[accel_window_mask]
                )
            ),
            "launch_forward_accel_mae_mps2": _mean(
                np.abs(
                    sim_forward_accel[accel_window_mask]
                    - target_forward_accel[accel_window_mask]
                )
            ),
            "launch_peak_forward_accel_diff_mps2": float(
                abs(
                    np.max(sim_forward_accel[accel_window_mask], initial=0.0)
                    - np.max(target_forward_accel[accel_window_mask], initial=0.0)
                )
            ),
            "best_lag_seconds": _best_lag_seconds(
                score_rel_times,
                pos_err,
                yaw_err_deg,
                target_xy,
                target_yaw,
                sim[:, :2],
                sim[:, 2],
            ),
            "traction_active_fraction": diagnostic_arrays["traction_active_fraction"][
                index
            ],
            "traction_mean_scale": diagnostic_arrays["traction_mean_scale"][index],
            "traction_min_scale": diagnostic_arrays["traction_min_scale"][index],
            "launch_traction_active_fraction": diagnostic_arrays[
                "launch_traction_active_fraction"
            ][index],
            "launch_traction_mean_scale": diagnostic_arrays[
                "launch_traction_mean_scale"
            ][index],
            "launch_traction_min_scale": diagnostic_arrays["launch_traction_min_scale"][
                index
            ],
            "slip_mean": diagnostic_arrays["slip_mean"][index],
            "slip_max": diagnostic_arrays["slip_max"][index],
            "motor_curve_mean_compression": diagnostic_arrays[
                "motor_curve_mean_compression"
            ][index],
            "motor_curve_max_compression": diagnostic_arrays[
                "motor_curve_max_compression"
            ][index],
            "traction_mean_attenuation": diagnostic_arrays["traction_mean_attenuation"][
                index
            ],
            "traction_max_attenuation": diagnostic_arrays["traction_max_attenuation"][
                index
            ],
            "wheel_speed_mismatch_mean": diagnostic_arrays["wheel_speed_mismatch_mean"][
                index
            ],
            "wheel_speed_mismatch_max": diagnostic_arrays["wheel_speed_mismatch_max"][
                index
            ],
        }
        rows.append(row)

    rows.sort(key=lambda row: row["score_total"], reverse=True)
    _write_csv(output_dir / f"{split_name}_window_metrics.csv", rows)
    _write_csv(output_dir / f"{split_name}_worst_windows.csv", rows[: max(top_k, 1)])
    launch_rows = [row for row in rows if row["launch_window"]]
    _write_csv(
        output_dir / f"{split_name}_worst_launch_windows.csv",
        launch_rows[: max(top_k, 1)],
    )

    by_maneuver: dict[str, dict[str, float]] = {}
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["maneuver"]].append(row)
    for maneuver, maneuver_rows in sorted(grouped_rows.items()):
        by_maneuver[maneuver] = {
            "windows": len(maneuver_rows),
            "score_total": _mean(
                np.asarray([row["score_total"] for row in maneuver_rows])
            ),
            "position_rmse_m": _mean(
                np.asarray([row["position_rmse_m"] for row in maneuver_rows])
            ),
            "yaw_mae_deg": _mean(
                np.asarray([row["yaw_mae_deg"] for row in maneuver_rows])
            ),
            "launch_forward_accel_mae_mps2": _mean(
                np.asarray(
                    [row["launch_forward_accel_mae_mps2"] for row in maneuver_rows]
                )
            ),
        }

    by_run: dict[str, dict[str, float]] = {}
    grouped_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_by_run[row["run_name"]].append(row)
    for run_name, run_rows in sorted(grouped_by_run.items()):
        by_run[run_name] = {
            "windows": len(run_rows),
            "score_total": _mean(np.asarray([row["score_total"] for row in run_rows])),
            "position_rmse_m": _mean(
                np.asarray([row["position_rmse_m"] for row in run_rows])
            ),
            "yaw_mae_deg": _mean(np.asarray([row["yaw_mae_deg"] for row in run_rows])),
        }

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _save_bar_chart(
        plots_dir / f"{split_name}_maneuver_position_rmse.png",
        f"{split_name} mean position RMSE by maneuver",
        [(name, values["position_rmse_m"]) for name, values in by_maneuver.items()],
        PLOT_BLUE,
    )
    _save_bar_chart(
        plots_dir / f"{split_name}_maneuver_yaw_mae.png",
        f"{split_name} mean yaw MAE by maneuver",
        [(name, values["yaw_mae_deg"]) for name, values in by_maneuver.items()],
        PLOT_ORANGE,
    )
    _save_scatter_plot(
        plots_dir / f"{split_name}_command_vs_rmse.png",
        f"{split_name} command magnitude vs position RMSE",
        np.asarray([row["command_max_abs"] for row in rows], dtype=np.float32),
        np.asarray([row["position_rmse_m"] for row in rows], dtype=np.float32),
        np.asarray([row["launch_window"] for row in rows], dtype=np.bool_),
        "max command abs",
        "position RMSE (m)",
    )
    _save_scatter_plot(
        plots_dir / f"{split_name}_launch_step_vs_yaw.png",
        f"{split_name} launch command step vs yaw MAE",
        np.asarray([row["launch_step_max"] for row in rows], dtype=np.float32),
        np.asarray([row["yaw_mae_deg"] for row in rows], dtype=np.float32),
        np.asarray([row["launch_window"] for row in rows], dtype=np.bool_),
        "launch step max",
        "yaw MAE (deg)",
    )

    bins = np.arange(0.0, 5.25, 0.25, dtype=np.float32)
    centers = 0.5 * (bins[:-1] + bins[1:])
    _save_line_plot(
        plots_dir / f"{split_name}_position_error_over_time.png",
        f"{split_name} position error over score-relative time",
        centers,
        [
            (
                "all windows",
                _masked_time_bin_means(per_window_times_all, per_window_pos_all, bins),
                PLOT_BLUE,
            ),
            (
                "launch windows",
                _masked_time_bin_means(
                    per_window_times_launch, per_window_pos_launch, bins
                ),
                PLOT_RED,
            ),
        ],
        "score-relative time (s)",
        "position error (m)",
    )
    _save_line_plot(
        plots_dir / f"{split_name}_yaw_error_over_time.png",
        f"{split_name} yaw error over score-relative time",
        centers,
        [
            (
                "all windows",
                _masked_time_bin_means(per_window_times_all, per_window_yaw_all, bins),
                PLOT_GREEN,
            ),
            (
                "launch windows",
                _masked_time_bin_means(
                    per_window_times_launch, per_window_yaw_launch, bins
                ),
                PLOT_RED,
            ),
        ],
        "score-relative time (s)",
        "yaw error (deg)",
    )

    summary = {
        "windows": len(rows),
        "launch_windows": len(launch_rows),
        "score_total": _mean(
            np.asarray([row["score_total"] for row in rows], dtype=np.float32)
        ),
        "position_rmse_m": _mean(
            np.asarray([row["position_rmse_m"] for row in rows], dtype=np.float32)
        ),
        "yaw_mae_deg": _mean(
            np.asarray([row["yaw_mae_deg"] for row in rows], dtype=np.float32)
        ),
        "endpoint_position_m": _mean(
            np.asarray([row["endpoint_position_m"] for row in rows], dtype=np.float32)
        ),
        "endpoint_yaw_deg": _mean(
            np.asarray([row["endpoint_yaw_deg"] for row in rows], dtype=np.float32)
        ),
        "launch_forward_accel_mae_mps2": _mean(
            np.asarray(
                [row["launch_forward_accel_mae_mps2"] for row in rows], dtype=np.float32
            )
        ),
        "traction_active_fraction": _mean(
            np.asarray(
                [row["traction_active_fraction"] for row in rows], dtype=np.float32
            )
        ),
        "launch_traction_active_fraction": _mean(
            np.asarray(
                [row["launch_traction_active_fraction"] for row in rows],
                dtype=np.float32,
            )
        ),
        "motor_curve_mean_compression": _mean(
            np.asarray(
                [row["motor_curve_mean_compression"] for row in rows], dtype=np.float32
            )
        ),
        "traction_mean_attenuation": _mean(
            np.asarray(
                [row["traction_mean_attenuation"] for row in rows], dtype=np.float32
            )
        ),
        "best_lag_seconds": _mean(
            np.asarray([row["best_lag_seconds"] for row in rows], dtype=np.float32)
        ),
    }

    worst_windows = rows[: max(top_k, 1)]
    worst_launch_windows = launch_rows[: max(top_k, 1)]
    return {
        "summary": summary,
        "by_maneuver": by_maneuver,
        "by_run": by_run,
        "worst_windows": worst_windows,
        "worst_launch_windows": worst_launch_windows,
        "paths": {
            "window_metrics_csv": str(output_dir / f"{split_name}_window_metrics.csv"),
            "worst_windows_csv": str(output_dir / f"{split_name}_worst_windows.csv"),
            "worst_launch_windows_csv": str(
                output_dir / f"{split_name}_worst_launch_windows.csv"
            ),
            "plots_dir": str(plots_dir),
        },
    }


def _print_split_report(name: str, report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(
        f"[{name}] windows={summary['windows']} launch_windows={summary['launch_windows']}"
    )
    print(
        "  total={:.3f} pos_rmse={:.3f}m yaw_mae={:.2f}deg endpoint_pos={:.3f}m endpoint_yaw={:.2f}deg".format(
            summary["score_total"],
            summary["position_rmse_m"],
            summary["yaw_mae_deg"],
            summary["endpoint_position_m"],
            summary["endpoint_yaw_deg"],
        )
    )
    print(
        "  launch_accel_mae={:.3f}m/s^2 traction_active={:.3f} launch_traction_active={:.3f} motor_compression={:.4f} traction_attenuation={:.4f} lag={:.3f}s".format(
            summary["launch_forward_accel_mae_mps2"],
            summary["traction_active_fraction"],
            summary["launch_traction_active_fraction"],
            summary["motor_curve_mean_compression"],
            summary["traction_mean_attenuation"],
            summary["best_lag_seconds"],
        )
    )
    print("  worst windows:")
    for row in report["worst_windows"][:5]:
        print(
            "    {} {:.2f}-{:.2f}s {} total={:.3f} pos_rmse={:.3f} yaw={:.2f} launch_step={:.3f}".format(
                row["run_name"],
                row["start_time"],
                row["end_time"],
                row["maneuver"],
                row["score_total"],
                row["position_rmse_m"],
                row["yaw_mae_deg"],
                row["launch_step_max"],
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze sysid physics mismatch against recorded data."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument(
        "--history-source",
        choices=[
            "global_best",
            "best_train",
            "best_eval_generation",
            "last_generation",
        ],
        default="global_best",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sysid_analysis"))
    parser.add_argument("--split", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--preroll-seconds", type=float, default=0.5)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--launch-duration-seconds", type=float, default=1.0)
    parser.add_argument("--launch-step-threshold", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    params = _load_params(args.params, args.history_source)
    config = WindowConfig(
        window_seconds=args.window_seconds,
        preroll_seconds=args.preroll_seconds,
        stride_seconds=args.stride_seconds,
    )
    dataset = load_dataset_splits(args.data_root, config)
    evaluator = SysIdEvaluator.create(
        EnvironmentConfig(), num_substeps=config.num_substeps
    )
    latent = physical_to_latent(params)

    report: dict[str, Any] = {
        "params_path": str(args.params),
        "history_source": args.history_source,
        "window_config": {
            "window_seconds": args.window_seconds,
            "preroll_seconds": args.preroll_seconds,
            "stride_seconds": args.stride_seconds,
        },
        "params": params,
        "splits": {},
    }

    split_batches = []
    if args.split in {"train", "both"}:
        split_batches.append(("train", dataset.train))
    if args.split in {"eval", "both"}:
        split_batches.append(("eval", dataset.eval))

    for split_name, batch in split_batches:
        split_output_dir = args.output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        split_report = _analyze_split(
            split_name,
            batch,
            evaluator,
            latent,
            split_output_dir,
            args.top_k,
            args.launch_duration_seconds,
            args.launch_step_threshold,
        )
        report["splits"][split_name] = split_report
        _print_split_report(split_name, split_report)

    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved analysis report to {report_path}")


if __name__ == "__main__":
    main()
