from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from environment.config import EnvironmentConfig
from environment.environment import TagEnvironment
from environment.model_build import build_mjx_model
from environment.mujoco_data import yaw_to_quaternion
from sysid.dataset import DatasetConfig, build_prepared_dataset
from sysid.load_runs import discover_run_dirs, load_run
from sysid.replay import (
    NominalParameters,
    _segment_family,
    encode_params,
    make_dataset_evaluator,
    params_to_domain_params,
    summarize_metrics,
)
from sysid.types import PreparedDataset


def _wrap_angle_np(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _moving_average(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values
    kernel = np.ones(2 * radius + 1, dtype=np.float64)
    kernel /= kernel.sum()
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), axis=1, arr=values
    )


def _pose_errors(
    predicted: np.ndarray, references: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    position_error = np.linalg.norm(predicted[..., :2] - references[..., :2], axis=-1)
    yaw_error = np.abs(_wrap_angle_np(predicted[..., 2] - references[..., 2]))
    return position_error, yaw_error


def _linear_speed(poses: np.ndarray, dt: float) -> np.ndarray:
    deltas = poses[:, 1:, :2] - poses[:, :-1, :2]
    speed = np.linalg.norm(deltas, axis=-1) / max(dt, 1e-6)
    return np.concatenate([speed[:, :1], speed], axis=1)


def _yaw_rate(poses: np.ndarray, dt: float) -> np.ndarray:
    dyaw = _wrap_angle_np(poses[:, 1:, 2] - poses[:, :-1, 2]) / max(dt, 1e-6)
    return np.concatenate([dyaw[:, :1], dyaw], axis=1)


def _signed_lateral_error(predicted: np.ndarray, references: np.ndarray) -> np.ndarray:
    delta = predicted[..., :2] - references[..., :2]
    yaw = references[..., 2]
    left_normal = np.stack([-np.sin(yaw), np.cos(yaw)], axis=-1)
    return np.sum(delta * left_normal, axis=-1)


def _command_features(controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    linear = 0.5 * (controls[..., 0] + controls[..., 1])
    angular = 0.5 * (controls[..., 1] - controls[..., 0])
    return linear, angular


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    total = float(np.sum(values * mask))
    count = float(np.sum(mask))
    return total / max(count, 1.0)


def _masked_rmse(values: np.ndarray, mask: np.ndarray) -> float:
    total = float(np.sum(np.square(values) * mask))
    count = float(np.sum(mask))
    return float(np.sqrt(total / max(count, 1.0)))


def _segment_indices(dataset: PreparedDataset, label: str) -> np.ndarray:
    return np.asarray(
        [index for index, value in enumerate(dataset.segment_labels) if value == label],
        dtype=np.int32,
    )


def _subset_dataset(dataset: PreparedDataset, indices: np.ndarray) -> PreparedDataset:
    selection = jnp.asarray(indices, dtype=jnp.int32)
    return PreparedDataset(
        initial_poses=dataset.initial_poses[selection],
        controls=dataset.controls[selection],
        references=dataset.references[selection],
        mask=dataset.mask[selection],
        role=dataset.role,
        run_names=tuple(dataset.run_names[index] for index in indices.tolist()),
        segment_labels=tuple(
            dataset.segment_labels[index] for index in indices.tolist()
        ),
    )


def _load_params(args: argparse.Namespace) -> NominalParameters:
    if args.params_json is not None:
        payload = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
        if "best_params" in payload:
            payload = payload["best_params"]
        payload.setdefault("track_width_scale", 1.0)
        payload.setdefault("wheel_scrub_scale", 1.0)
        return NominalParameters(**payload)
    return NominalParameters(
        action_delay_substeps=args.action_delay_substeps,
        observation_delay_substeps=args.observation_delay_substeps,
        mass_scale=args.mass_scale,
        com_offset_x=args.com_offset_x,
        com_offset_y=args.com_offset_y,
        track_width_scale=args.track_width_scale,
        wheel_friction_scale=args.wheel_friction_scale,
        wheel_scrub_scale=args.wheel_scrub_scale,
        caster_friction_scale=args.caster_friction_scale,
        wheel_frictionloss_scale=args.wheel_frictionloss_scale,
        motor_strength_scale=args.motor_strength_scale,
        back_emf_scale=args.back_emf_scale,
        motor_balance=args.motor_balance,
    )


def replay_dataset(
    dataset: PreparedDataset,
    params: NominalParameters,
    env_config: EnvironmentConfig | None = None,
) -> np.ndarray:
    env = TagEnvironment(env_config or EnvironmentConfig())
    is_chaser = dataset.role == "chaser"
    num_segments = int(dataset.initial_poses.shape[0])
    zero_qvel = jnp.zeros((num_segments, env.mj_model.nv), dtype=jnp.float32)
    zero_ctrl = jnp.zeros((num_segments, env.mj_model.nu), dtype=jnp.float32)
    parked_xy = jnp.asarray(
        [
            0.5 * env.config.arena_width - 2.0 * env.config.agent_radius,
            -(0.5 * env.config.arena_height - 2.0 * env.config.agent_radius),
        ],
        dtype=jnp.float32,
    )
    parked_quat = yaw_to_quaternion(jnp.asarray(0.0, dtype=jnp.float32))
    identity_quaternion = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    model = build_mjx_model(
        env.mj_model,
        params_to_domain_params(params, dataset.role),
        env.model_indices,
    )

    def make_initial_qpos(initial_pose: jax.Array) -> jax.Array:
        qpos = jnp.zeros(env.mj_model.nq, dtype=jnp.float32)
        target_xy = initial_pose[:2]
        target_quat = yaw_to_quaternion(initial_pose[2])
        if is_chaser:
            chaser_xy, chaser_quat = target_xy, target_quat
            evader_xy, evader_quat = parked_xy, parked_quat
        else:
            chaser_xy, chaser_quat = parked_xy, parked_quat
            evader_xy, evader_quat = target_xy, target_quat
        qpos = qpos.at[
            env.joint_qpos_slices.chaser_root.start : env.joint_qpos_slices.chaser_root.start
            + 3
        ].set(
            jnp.asarray(
                [chaser_xy[0], chaser_xy[1], env.config.agent_z], dtype=jnp.float32
            )
        )
        qpos = qpos.at[
            env.joint_qpos_slices.chaser_root.start
            + 3 : env.joint_qpos_slices.chaser_root.start + 7
        ].set(chaser_quat)
        qpos = qpos.at[env.joint_qpos_slices.chaser_caster_ball_joint].set(
            identity_quaternion
        )
        qpos = qpos.at[
            env.joint_qpos_slices.evader_root.start : env.joint_qpos_slices.evader_root.start
            + 3
        ].set(
            jnp.asarray(
                [evader_xy[0], evader_xy[1], env.config.agent_z], dtype=jnp.float32
            )
        )
        qpos = qpos.at[
            env.joint_qpos_slices.evader_root.start
            + 3 : env.joint_qpos_slices.evader_root.start + 7
        ].set(evader_quat)
        qpos = qpos.at[env.joint_qpos_slices.evader_caster_ball_joint].set(
            identity_quaternion
        )
        return qpos

    initial_qpos = jax.vmap(make_initial_qpos)(dataset.initial_poses)

    def init_data(qpos: jax.Array, qvel: jax.Array, ctrl: jax.Array) -> mjx.Data:
        data = env._template_mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        return env.forward(model, data)

    batched_data = jax.vmap(init_data)(initial_qpos, zero_qvel, zero_ctrl)
    action_buffer = jnp.zeros((num_segments, 11, env.mj_model.nu), dtype=jnp.float32)
    observation_buffer = jnp.tile(dataset.initial_poses[:, None, :], (1, 11, 1))
    controls = dataset.controls
    controls_left = controls[..., 0]
    controls_right = controls[..., 1]
    zeros = jnp.zeros_like(controls_left.T)
    proposed_ctrl = (
        jnp.stack([controls_left.T, controls_right.T, zeros, zeros], axis=-1)
        if is_chaser
        else jnp.stack([zeros, zeros, controls_left.T, controls_right.T], axis=-1)
    )

    def extract_pose(data: mjx.Data) -> jax.Array:
        chaser, evader = env._agent_pair_kinematics(data)
        agent_kin = chaser if is_chaser else evader
        return jnp.asarray(
            [agent_kin.position_xy[0], agent_kin.position_xy[1], agent_kin.yaw],
            dtype=jnp.float32,
        )

    action_delay = jnp.asarray(params.action_delay_substeps, dtype=jnp.int32)
    observation_delay = jnp.asarray(params.observation_delay_substeps, dtype=jnp.int32)

    def rollout_step(
        carry: tuple[mjx.Data, jax.Array, jax.Array], proposed_t: jax.Array
    ) -> tuple[tuple[mjx.Data, jax.Array, jax.Array], jax.Array]:
        data, buffer, pose_buffer = carry

        def physics_substep(
            inner_carry: tuple[mjx.Data, jax.Array, jax.Array], _: None
        ) -> tuple[tuple[mjx.Data, jax.Array, jax.Array], jax.Array]:
            inner_data, inner_buffer, inner_pose_buffer = inner_carry
            updated_buffer = jnp.concatenate(
                [proposed_t[:, None, :], inner_buffer[:, :-1, :]], axis=1
            )
            applied = jnp.take(updated_buffer, action_delay, axis=1)
            inner_data = inner_data.replace(ctrl=applied)
            stepped = jax.vmap(lambda single: mjx.step(model, single))(inner_data)
            poses = jax.vmap(extract_pose)(stepped)
            updated_pose_buffer = jnp.concatenate(
                [poses[:, None, :], inner_pose_buffer[:, :-1, :]], axis=1
            )
            observed = jnp.take(updated_pose_buffer, observation_delay, axis=1)
            return (stepped, updated_buffer, updated_pose_buffer), observed

        (data, buffer, pose_buffer), observed_substeps = jax.lax.scan(
            physics_substep,
            (data, buffer, pose_buffer),
            None,
            length=env.substeps_per_action,
        )
        return (data, buffer, pose_buffer), observed_substeps[-1]

    (_, _, _), predicted = jax.lax.scan(
        rollout_step,
        (batched_data, action_buffer, observation_buffer),
        proposed_ctrl,
    )
    return np.asarray(jax.device_get(predicted.transpose(1, 0, 2)))


def _metrics_for_indices(
    dataset: PreparedDataset,
    predicted: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    if indices.size == 0:
        return {}
    references = np.asarray(dataset.references)[indices]
    controls = np.asarray(dataset.controls)[indices]
    mask = np.asarray(dataset.mask)[indices]
    predicted_sel = predicted[indices]
    dt = 1.0 / 20.0

    position_error, yaw_error = _pose_errors(predicted_sel, references)
    pred_speed = _linear_speed(predicted_sel, dt)
    ref_speed = _linear_speed(references, dt)
    pred_yaw_rate = _yaw_rate(predicted_sel, dt)
    ref_yaw_rate = _yaw_rate(references, dt)
    lateral_error = _signed_lateral_error(predicted_sel, references)
    linear_cmd, angular_cmd = _command_features(controls)
    curvature_mask = (ref_speed > 0.08).astype(np.float64) * mask
    pred_curvature = pred_yaw_rate / np.maximum(pred_speed, 0.08)
    ref_curvature = ref_yaw_rate / np.maximum(ref_speed, 0.08)

    segment_scores: list[tuple[float, int]] = []
    for local_index in range(predicted_sel.shape[0]):
        valid = mask[local_index] > 0.0
        if not np.any(valid):
            continue
        score = float(
            0.5 * np.sqrt(np.mean(np.square(position_error[local_index, valid])))
            + 0.3 * np.sqrt(np.mean(np.square(yaw_error[local_index, valid])))
            + 0.2
            * np.sqrt(
                np.mean(
                    np.square(
                        pred_yaw_rate[local_index, valid]
                        - ref_yaw_rate[local_index, valid]
                    )
                )
            )
            / 20.0
        )
        segment_scores.append((score, int(indices[local_index])))

    command_active = ((np.abs(linear_cmd) + np.abs(angular_cmd)) > 0.15).astype(
        np.float64
    ) * mask
    left_turn = (angular_cmd > 0.15).astype(np.float64) * mask
    right_turn = (angular_cmd < -0.15).astype(np.float64) * mask
    forward = (linear_cmd > 0.15).astype(np.float64) * mask
    reverse = (linear_cmd < -0.15).astype(np.float64) * mask

    start_window = int(min(5, mask.shape[1]))
    start_mask = np.zeros_like(mask)
    start_mask[:, :start_window] = mask[:, :start_window]
    end_mask = np.zeros_like(mask)
    for row_index in range(mask.shape[0]):
        valid_count = int(np.sum(mask[row_index] > 0.0))
        if valid_count <= 0:
            continue
        window_start = max(0, valid_count - start_window)
        end_mask[row_index, window_start:valid_count] = mask[
            row_index, window_start:valid_count
        ]

    return {
        "segment_count": int(indices.size),
        "sample_count": int(mask.sum()),
        "position_rmse_m": _masked_rmse(position_error, mask),
        "yaw_rmse_rad": _masked_rmse(yaw_error, mask),
        "yaw_rate_rmse_rad_s": _masked_rmse(pred_yaw_rate - ref_yaw_rate, mask),
        "speed_rmse_m_s": _masked_rmse(pred_speed - ref_speed, mask),
        "curvature_rmse": _masked_rmse(pred_curvature - ref_curvature, curvature_mask),
        "mean_position_bias_m": _masked_mean(position_error, mask),
        "mean_lateral_bias_m": _masked_mean(lateral_error, mask),
        "mean_yaw_bias_rad": _masked_mean(
            _wrap_angle_np(predicted_sel[..., 2] - references[..., 2]), mask
        ),
        "mean_speed_bias_m_s": _masked_mean(pred_speed - ref_speed, mask),
        "mean_yaw_rate_bias_rad_s": _masked_mean(pred_yaw_rate - ref_yaw_rate, mask),
        "mean_command_linear": _masked_mean(linear_cmd, mask),
        "mean_command_angular": _masked_mean(angular_cmd, mask),
        "forward_speed_bias_m_s": _masked_mean(pred_speed - ref_speed, forward),
        "reverse_speed_bias_m_s": _masked_mean(pred_speed - ref_speed, reverse),
        "left_turn_yaw_rate_bias_rad_s": _masked_mean(
            pred_yaw_rate - ref_yaw_rate, left_turn
        ),
        "right_turn_yaw_rate_bias_rad_s": _masked_mean(
            pred_yaw_rate - ref_yaw_rate, right_turn
        ),
        "active_command_position_rmse_m": _masked_rmse(position_error, command_active),
        "start_window_position_rmse_m": _masked_rmse(position_error, start_mask),
        "start_window_yaw_rate_rmse_rad_s": _masked_rmse(
            pred_yaw_rate - ref_yaw_rate, start_mask
        ),
        "end_window_position_rmse_m": _masked_rmse(position_error, end_mask),
        "end_window_yaw_rate_rmse_rad_s": _masked_rmse(
            pred_yaw_rate - ref_yaw_rate, end_mask
        ),
        "worst_segment_indices": [
            index for _, index in sorted(segment_scores, reverse=True)[:5]
        ],
    }


def _worst_segments(
    dataset: PreparedDataset, predicted: np.ndarray, top_k: int
) -> list[dict[str, Any]]:
    references = np.asarray(dataset.references)
    mask = np.asarray(dataset.mask)
    dt = 1.0 / 20.0
    position_error, yaw_error = _pose_errors(predicted, references)
    pred_yaw_rate = _yaw_rate(predicted, dt)
    ref_yaw_rate = _yaw_rate(references, dt)
    rows: list[dict[str, Any]] = []
    for index in range(predicted.shape[0]):
        valid = mask[index] > 0.0
        if not np.any(valid):
            continue
        score = float(
            0.5 * np.sqrt(np.mean(np.square(position_error[index, valid])))
            + 0.3 * np.sqrt(np.mean(np.square(yaw_error[index, valid])))
            + 0.2
            * np.sqrt(
                np.mean(
                    np.square(pred_yaw_rate[index, valid] - ref_yaw_rate[index, valid])
                )
            )
            / 20.0
        )
        rows.append(
            {
                "index": index,
                "run_name": dataset.run_names[index],
                "segment_label": dataset.segment_labels[index],
                "score": score,
                "position_rmse_m": float(
                    np.sqrt(np.mean(np.square(position_error[index, valid])))
                ),
                "yaw_rmse_rad": float(
                    np.sqrt(np.mean(np.square(yaw_error[index, valid])))
                ),
                "yaw_rate_rmse_rad_s": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                pred_yaw_rate[index, valid] - ref_yaw_rate[index, valid]
                            )
                        )
                    )
                ),
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[:top_k]


def _trace_for_segment(
    dataset: PreparedDataset, predicted: np.ndarray, index: int, dt: float
) -> dict[str, Any]:
    references = np.asarray(dataset.references)[index]
    controls = np.asarray(dataset.controls)[index]
    mask = np.asarray(dataset.mask)[index] > 0.0
    valid_count = int(np.sum(mask))
    predicted_seg = predicted[index, :valid_count]
    references_seg = references[:valid_count]
    controls_seg = controls[:valid_count]

    pred_speed = _linear_speed(predicted_seg[None, ...], dt)[0]
    ref_speed = _linear_speed(references_seg[None, ...], dt)[0]
    pred_yaw_rate = _yaw_rate(predicted_seg[None, ...], dt)[0]
    ref_yaw_rate = _yaw_rate(references_seg[None, ...], dt)[0]
    position_error, yaw_error = _pose_errors(
        predicted_seg[None, ...], references_seg[None, ...]
    )
    linear_cmd, angular_cmd = _command_features(controls_seg[None, ...])

    return {
        "index": index,
        "run_name": dataset.run_names[index],
        "segment_label": dataset.segment_labels[index],
        "time_s": [round(step * dt, 4) for step in range(valid_count)],
        "command_linear": linear_cmd[0].round(6).tolist(),
        "command_angular": angular_cmd[0].round(6).tolist(),
        "real_speed_m_s": ref_speed.round(6).tolist(),
        "sim_speed_m_s": pred_speed.round(6).tolist(),
        "real_yaw_rate_rad_s": ref_yaw_rate.round(6).tolist(),
        "sim_yaw_rate_rad_s": pred_yaw_rate.round(6).tolist(),
        "position_error_m": position_error[0].round(6).tolist(),
        "yaw_error_rad": yaw_error[0].round(6).tolist(),
        "real_pose": references_seg.round(6).tolist(),
        "sim_pose": predicted_seg.round(6).tolist(),
    }


def _command_response_bins(
    dataset: PreparedDataset,
    predicted: np.ndarray,
    dt: float,
    num_bins: int = 5,
) -> dict[str, Any]:
    references = np.asarray(dataset.references)
    controls = np.asarray(dataset.controls)
    mask = np.asarray(dataset.mask)
    pred_speed = _linear_speed(predicted, dt)
    ref_speed = _linear_speed(references, dt)
    pred_yaw_rate = _yaw_rate(predicted, dt)
    ref_yaw_rate = _yaw_rate(references, dt)
    linear_cmd, angular_cmd = _command_features(controls)

    def summarize_bins(
        values: np.ndarray, response_real: np.ndarray, response_sim: np.ndarray
    ) -> list[dict[str, Any]]:
        valid = mask > 0.0
        values_flat = values[valid]
        real_flat = response_real[valid]
        sim_flat = response_sim[valid]
        if values_flat.size == 0:
            return []
        edges = np.linspace(-1.0, 1.0, num_bins + 1)
        buckets: list[dict[str, Any]] = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=False):
            if hi >= 1.0:
                selection = (values_flat >= lo) & (values_flat <= hi)
            else:
                selection = (values_flat >= lo) & (values_flat < hi)
            count = int(np.sum(selection))
            if count == 0:
                continue
            buckets.append(
                {
                    "command_min": float(lo),
                    "command_max": float(hi),
                    "count": count,
                    "real_mean": float(np.mean(real_flat[selection])),
                    "sim_mean": float(np.mean(sim_flat[selection])),
                    "bias": float(np.mean(sim_flat[selection] - real_flat[selection])),
                }
            )
        return buckets

    direction_masks = {
        "all": mask > 0.0,
        "left_turn": (angular_cmd > 0.15) & (mask > 0.0),
        "right_turn": (angular_cmd < -0.15) & (mask > 0.0),
        "forward": (linear_cmd > 0.15) & (mask > 0.0),
        "reverse": (linear_cmd < -0.15) & (mask > 0.0),
    }

    response: dict[str, Any] = {
        "linear_command_vs_speed": summarize_bins(linear_cmd, ref_speed, pred_speed),
        "angular_command_vs_yaw_rate": summarize_bins(
            angular_cmd, ref_yaw_rate, pred_yaw_rate
        ),
        "by_direction": {},
    }
    for name, direction_mask in direction_masks.items():
        mask_float = direction_mask.astype(np.float64)
        response["by_direction"][name] = {
            "speed_bias_mean": _masked_mean(pred_speed - ref_speed, mask_float),
            "yaw_rate_bias_mean": _masked_mean(
                pred_yaw_rate - ref_yaw_rate, mask_float
            ),
            "speed_rmse": _masked_rmse(pred_speed - ref_speed, mask_float),
            "yaw_rate_rmse": _masked_rmse(pred_yaw_rate - ref_yaw_rate, mask_float),
        }
    return response


def _bound_proximity_report(
    params: NominalParameters, env: TagEnvironment
) -> dict[str, Any]:
    bounds = {
        "action_delay_substeps": (
            0.0,
            float(
                min(
                    max(
                        env.config.pipeline_randomization.max_action_delay_substeps, 40
                    ),
                    10,
                )
            ),
        ),
        "observation_delay_substeps": (
            0.0,
            float(
                min(
                    max(
                        env.config.pipeline_randomization.max_observation_delay_substeps,
                        40,
                    ),
                    10,
                )
            ),
        ),
        "mass_scale": (0.97, 1.03),
        "com_offset_x": (-0.005, 0.005),
        "com_offset_y": (-0.005, 0.005),
        "track_width_scale": (0.94, 1.06),
        "wheel_friction_scale": (0.85, 1.15),
        "wheel_scrub_scale": (0.6, 1.4),
        "caster_friction_scale": (0.3, 1.0),
        "wheel_frictionloss_scale": (0.8, 1.4),
        "motor_strength_scale": (0.8, 1.25),
        "back_emf_scale": (0.85, 1.15),
        "motor_balance": (-0.1, 0.1),
    }
    report: dict[str, Any] = {}
    for name, value in asdict(params).items():
        low, high = bounds[name]
        normalized = 0.5 if high <= low else (float(value) - low) / (high - low)
        report[name] = {
            "value": float(value),
            "min": low,
            "max": high,
            "normalized": normalized,
            "near_lower": normalized <= 0.05,
            "near_upper": normalized >= 0.95,
        }
    return report


def _conclusions(payload: dict[str, Any]) -> list[str]:
    overall = payload["overall"]
    by_family = payload["by_family"]
    conclusions: list[str] = []

    straight = by_family.get("straight")
    if straight is not None:
        speed_bias = float(straight["mean_speed_bias_m_s"])
        if speed_bias < -0.08:
            conclusions.append(
                f"Sim is too slow on straight segments (mean speed bias {speed_bias:.3f} m/s)."
            )
        elif speed_bias > 0.08:
            conclusions.append(
                f"Sim is too fast on straight segments (mean speed bias {speed_bias:.3f} m/s)."
            )

    arc = by_family.get("arc")
    if arc is not None:
        curvature_rmse = float(arc["curvature_rmse"])
        right_turn_bias = float(arc["right_turn_yaw_rate_bias_rad_s"])
        left_turn_bias = float(arc["left_turn_yaw_rate_bias_rad_s"])
        if curvature_rmse > 1.5:
            conclusions.append(
                f"Arc turn coupling is wrong (curvature RMSE {curvature_rmse:.3f}), so turn-rate vs forward-speed behavior does not match real motion."
            )
        if abs(right_turn_bias - left_turn_bias) > 2.0:
            conclusions.append(
                f"Turn-direction asymmetry is large (left yaw-rate bias {left_turn_bias:.3f} rad/s, right {right_turn_bias:.3f} rad/s)."
            )

    spin = by_family.get("spin")
    if spin is not None:
        spin_yaw_rate_rmse = float(spin["yaw_rate_rmse_rad_s"])
        spin_position_rmse = float(spin["position_rmse_m"])
        if spin_yaw_rate_rmse > 4.0 and spin_position_rmse < 0.05:
            conclusions.append(
                f"Spin mismatch is primarily rotational, not translational (yaw-rate RMSE {spin_yaw_rate_rmse:.3f} rad/s, position RMSE {spin_position_rmse:.3f} m)."
            )

    start_yaw_rate = float(overall["start_window_yaw_rate_rmse_rad_s"])
    end_yaw_rate = float(overall["end_window_yaw_rate_rmse_rad_s"])
    if start_yaw_rate > 3.0 and abs(start_yaw_rate - end_yaw_rate) < 2.0:
        conclusions.append(
            f"Errors stay large from segment start to end (start yaw-rate RMSE {start_yaw_rate:.3f}, end {end_yaw_rate:.3f}), which points to steady-state model mismatch more than pure timing delay."
        )
    elif start_yaw_rate > end_yaw_rate + 2.0:
        conclusions.append(
            f"Early response mismatch dominates (start yaw-rate RMSE {start_yaw_rate:.3f} vs end {end_yaw_rate:.3f}), suggesting delay or actuator response issues."
        )

    worst_segments = payload.get("worst_segments", [])
    if worst_segments:
        conclusions.append(
            f"Top failing segment class is `{worst_segments[0]['segment_label']}`; inspect `worst_segments` for exact indices and labels."
        )

    if not conclusions:
        conclusions.append(
            "No dominant failure mode exceeded the heuristic thresholds."
        )
    return conclusions


def run_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    params = _load_params(args)
    runs = [load_run(path) for path in discover_run_dirs(args.run_dirs)]
    dataset_config = DatasetConfig(
        min_segment_duration_s=args.min_segment_duration_s,
        transition_duration_s=args.transition_duration_s,
        min_segment_steps=args.min_segment_steps,
        max_segment_steps=args.max_segment_steps,
    )
    dataset = build_prepared_dataset(runs, config=dataset_config)
    predicted = replay_dataset(dataset, params)
    env = TagEnvironment(EnvironmentConfig())
    references = np.asarray(dataset.references)
    mask = np.asarray(dataset.mask)
    dt = 1.0 / env.config.action_frequency
    overall_indices = np.arange(predicted.shape[0], dtype=np.int32)
    families: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(dataset.segment_labels):
        family = _segment_family(label)
        if family is not None:
            families[family].append(index)
    label_rollup: dict[str, Any] = {}
    for label in sorted(set(dataset.segment_labels)):
        label_rollup[label] = _metrics_for_indices(
            dataset, predicted, _segment_indices(dataset, label)
        )
    family_rollup = {
        family: _metrics_for_indices(
            dataset, predicted, np.asarray(indices, dtype=np.int32)
        )
        for family, indices in sorted(families.items())
    }
    run_rollup = {
        run_name: _metrics_for_indices(
            dataset,
            predicted,
            np.asarray(
                [
                    index
                    for index, value in enumerate(dataset.run_names)
                    if value == run_name
                ],
                dtype=np.int32,
            ),
        )
        for run_name in sorted(set(dataset.run_names))
    }

    overall_metrics = _metrics_for_indices(dataset, predicted, overall_indices)
    replay_metrics = make_dataset_evaluator(dataset).evaluate_population(
        jnp.asarray([encode_params(params)], dtype=jnp.float32)
    )[0]
    worst_segments = _worst_segments(dataset, predicted, top_k=args.top_k)

    position_error, yaw_error = _pose_errors(predicted, references)
    pred_speed = _linear_speed(predicted, dt)
    ref_speed = _linear_speed(references, dt)
    pred_yaw_rate = _yaw_rate(predicted, dt)
    ref_yaw_rate = _yaw_rate(references, dt)
    smoothed_pos = _moving_average(position_error, radius=2)
    smoothed_yaw = _moving_average(yaw_error, radius=2)
    smoothed_speed_bias = _moving_average(pred_speed - ref_speed, radius=2)

    payload = {
        "params": asdict(params),
        "bound_proximity": _bound_proximity_report(params, env),
        "dataset": {
            "role": dataset.role,
            "segment_count": int(dataset.initial_poses.shape[0]),
            "segment_steps": int(dataset.controls.shape[1]),
            "run_names": sorted(set(dataset.run_names)),
            "segment_labels": {
                label: int(sum(1 for value in dataset.segment_labels if value == label))
                for label in sorted(set(dataset.segment_labels))
            },
        },
        "overall": {
            **summarize_metrics(replay_metrics),
            **overall_metrics,
            "position_error_p50_m": float(
                np.percentile(position_error[mask > 0.0], 50)
            ),
            "position_error_p90_m": float(
                np.percentile(position_error[mask > 0.0], 90)
            ),
            "yaw_error_p50_rad": float(np.percentile(yaw_error[mask > 0.0], 50)),
            "yaw_error_p90_rad": float(np.percentile(yaw_error[mask > 0.0], 90)),
            "speed_bias_p90_abs_m_s": float(
                np.percentile(np.abs((pred_speed - ref_speed)[mask > 0.0]), 90)
            ),
            "yaw_rate_bias_p90_abs_rad_s": float(
                np.percentile(np.abs((pred_yaw_rate - ref_yaw_rate)[mask > 0.0]), 90)
            ),
            "smoothed_position_error_p90_m": float(
                np.percentile(smoothed_pos[mask > 0.0], 90)
            ),
            "smoothed_yaw_error_p90_rad": float(
                np.percentile(smoothed_yaw[mask > 0.0], 90)
            ),
            "smoothed_speed_bias_p90_abs_m_s": float(
                np.percentile(np.abs(smoothed_speed_bias[mask > 0.0]), 90)
            ),
        },
        "by_family": family_rollup,
        "by_label": label_rollup,
        "by_run": run_rollup,
        "command_response": _command_response_bins(dataset, predicted, dt),
        "worst_segments": worst_segments,
        "worst_segment_traces": [
            _trace_for_segment(dataset, predicted, item["index"], dt)
            for item in worst_segments[: args.trace_k]
        ],
        "diagnostic_hypotheses": [
            "High yaw-rate RMSE with modest position RMSE usually points to motor strength, back-EMF, motor balance, or caster friction mismatch.",
            "Large start-window errors suggest action delay, observation delay, or actuator response mismatch rather than steady-state friction.",
            "Large straight lateral bias suggests asymmetric wheel behavior or COM/caster mismatch.",
            "Large arc curvature RMSE suggests the turn-rate-to-speed relationship is wrong even if endpoint errors look acceptable.",
        ],
    }
    payload["conclusions"] = _conclusions(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--params-json", type=str, default=None)
    parser.add_argument("--action-delay-substeps", type=int, default=0)
    parser.add_argument("--observation-delay-substeps", type=int, default=0)
    parser.add_argument("--mass-scale", type=float, default=1.0)
    parser.add_argument("--com-offset-x", type=float, default=0.0)
    parser.add_argument("--com-offset-y", type=float, default=0.0)
    parser.add_argument("--track-width-scale", type=float, default=1.0)
    parser.add_argument("--wheel-friction-scale", type=float, default=1.0)
    parser.add_argument("--wheel-scrub-scale", type=float, default=1.0)
    parser.add_argument("--caster-friction-scale", type=float, default=1.0)
    parser.add_argument("--wheel-frictionloss-scale", type=float, default=1.0)
    parser.add_argument("--motor-strength-scale", type=float, default=1.0)
    parser.add_argument("--back-emf-scale", type=float, default=1.0)
    parser.add_argument("--motor-balance", type=float, default=0.0)
    parser.add_argument("--max-segment-steps", type=int, default=160)
    parser.add_argument("--min-segment-steps", type=int, default=12)
    parser.add_argument("--min-segment-duration-s", type=float, default=0.6)
    parser.add_argument("--transition-duration-s", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--trace-k", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--per-segment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_diagnosis(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
