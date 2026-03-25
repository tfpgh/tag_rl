from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from environment.config import EnvironmentConfig
from environment.environment import TagEnvironment
from environment.model_build import (
    build_mjx_models_process_parallel,
    create_process_model_pool,
    device_put_sharded_mjx_models,
    mjx_model_in_axes,
    nominal_host_agent_dynamics_params,
    pad_stacked_mjx_models,
    reshape_stacked_mjx_models_for_devices,
    stack_mjx_models,
    HostAgentDynamicsParams,
    HostDomainParams,
)
from environment.mujoco_data import yaw_to_quaternion
from sysid.dataset import build_prepared_dataset
from sysid.types import PreparedDataset, ReplayMetrics, RunData

SEGMENT_WEIGHTS = {
    "arc": 0.6,
    "straight": 0.3,
    "spin": 0.1,
}


@dataclass(frozen=True, slots=True)
class NominalParameters:
    action_delay_substeps: int = 0
    observation_delay_substeps: int = 0
    mass_scale: float = 1.0
    com_offset_x: float = 0.0
    com_offset_y: float = 0.0
    track_width_scale: float = 1.0
    wheel_friction_scale: float = 1.0
    wheel_scrub_scale: float = 1.0
    caster_friction_scale: float = 1.0
    wheel_frictionloss_scale: float = 1.0
    motor_strength_scale: float = 1.0
    back_emf_scale: float = 1.0
    motor_balance: float = 0.0


def infer_target_role(run: RunData) -> str:
    if run.target_robot_tag_id == 5:
        return "evader"
    return "chaser"


def summarize_metrics(metrics: ReplayMetrics) -> dict[str, float]:
    return {
        "position_rmse_m": metrics.position_rmse_m,
        "yaw_rmse_rad": metrics.yaw_rmse_rad,
        "yaw_rate_rmse_rad_s": metrics.yaw_rate_rmse_rad_s,
        "endpoint_position_error_m": metrics.endpoint_position_error_m,
        "endpoint_yaw_error_rad": metrics.endpoint_yaw_error_rad,
        "score": metrics.score,
        "sample_count": float(metrics.sample_count),
    }


def evaluate_run(
    run: RunData,
    params: NominalParameters,
    target_hz: float | None = None,
    env_config: EnvironmentConfig | None = None,
    build_workers: int | None = None,
) -> ReplayMetrics:
    dataset = build_prepared_dataset([run], target_hz=target_hz)
    evaluator = make_dataset_evaluator(
        dataset, env_config=env_config, build_workers=build_workers
    )
    metrics = evaluator.evaluate_population(
        jnp.asarray([encode_params(params)], dtype=jnp.float32)
    )
    return metrics[0]


def encode_params(params: NominalParameters) -> jnp.ndarray:
    return jnp.asarray(
        [
            float(params.action_delay_substeps),
            float(params.observation_delay_substeps),
            params.mass_scale,
            params.com_offset_x,
            params.com_offset_y,
            params.track_width_scale,
            params.wheel_friction_scale,
            params.wheel_scrub_scale,
            params.caster_friction_scale,
            params.wheel_frictionloss_scale,
            params.motor_strength_scale,
            params.back_emf_scale,
            params.motor_balance,
        ],
        dtype=jnp.float32,
    )


def _segment_family(label: str) -> str | None:
    if label in SEGMENT_WEIGHTS:
        return label
    for family in SEGMENT_WEIGHTS:
        if label.endswith(family):
            return family
    return None


def decoded_values_to_params(
    values: jax.Array,
    action_delay_substeps: jax.Array,
    observation_delay_substeps: jax.Array,
) -> NominalParameters:
    return NominalParameters(
        action_delay_substeps=int(action_delay_substeps),
        observation_delay_substeps=int(observation_delay_substeps),
        mass_scale=float(values[0]),
        com_offset_x=float(values[1]),
        com_offset_y=float(values[2]),
        track_width_scale=float(values[3]),
        wheel_friction_scale=float(values[4]),
        wheel_scrub_scale=float(values[5]),
        caster_friction_scale=float(values[6]),
        wheel_frictionloss_scale=float(values[7]),
        motor_strength_scale=float(values[8]),
        back_emf_scale=float(values[9]),
        motor_balance=float(values[10]),
    )


def params_to_domain_params(params: NominalParameters, role: str) -> HostDomainParams:
    parked_agent = nominal_host_agent_dynamics_params()
    agent = HostAgentDynamicsParams(
        mass_scale=params.mass_scale,
        com_offset_x=params.com_offset_x,
        com_offset_y=params.com_offset_y,
        track_width_scale=params.track_width_scale,
        wheel_friction_scale=params.wheel_friction_scale,
        wheel_scrub_scale=params.wheel_scrub_scale,
        caster_friction_scale=params.caster_friction_scale,
        wheel_frictionloss_scale=params.wheel_frictionloss_scale,
        motor_strength_scale=params.motor_strength_scale,
        back_emf_scale=params.back_emf_scale,
        motor_balance=params.motor_balance,
    )
    if role == "chaser":
        return HostDomainParams(chaser=agent, evader=parked_agent)
    return HostDomainParams(chaser=parked_agent, evader=agent)


@dataclass(slots=True)
class DatasetEvaluator:
    decode_population: Callable
    evaluate_population: Callable
    evaluate_population_fitness: Callable
    population_size_hint: int
    search_bounds: dict[str, tuple[float, float]]
    device_count: int


def make_dataset_evaluator(
    dataset: PreparedDataset,
    env_config: EnvironmentConfig | None = None,
    build_workers: int | None = None,
) -> DatasetEvaluator:
    env = TagEnvironment(env_config or EnvironmentConfig())
    process_pool: ProcessPoolExecutor | None = None
    if build_workers is not None and build_workers > 1:
        process_pool = create_process_model_pool(env.model_xml, build_workers)
    device_count = max(1, jax.local_device_count())
    max_action_delay = max(
        env.config.pipeline_randomization.max_action_delay_substeps, 40
    )
    max_observation_delay = max(
        env.config.pipeline_randomization.max_observation_delay_substeps, 40
    )
    is_chaser = dataset.role == "chaser"
    num_segments = int(dataset.initial_poses.shape[0])
    zero_qvel = jnp.zeros((num_segments, env.mj_model.nv), dtype=jnp.float32)
    zero_ctrl = jnp.zeros((num_segments, env.mj_model.nu), dtype=jnp.float32)
    mask = dataset.mask
    references = dataset.references
    initial_poses = dataset.initial_poses
    controls_left = dataset.controls[..., 0]
    controls_right = dataset.controls[..., 1]
    segment_weights = jnp.asarray(
        [
            SEGMENT_WEIGHTS.get(_segment_family(label) or "", 0.0)
            for label in dataset.segment_labels
        ],
        dtype=jnp.float32,
    )
    parked_xy = jnp.asarray(
        [
            0.5 * env.config.arena_width - 2.0 * env.config.agent_radius,
            -(0.5 * env.config.arena_height - 2.0 * env.config.agent_radius),
        ],
        dtype=jnp.float32,
    )
    parked_yaw = 0.0
    parked_quat = yaw_to_quaternion(jnp.asarray(parked_yaw, dtype=jnp.float32))
    identity_quaternion = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    lower_bounds = jnp.asarray(
        [
            0.0,
            0.0,
            0.97,
            -0.005,
            -0.005,
            0.94,
            0.85,
            0.6,
            0.3,
            0.8,
            0.8,
            0.85,
            -0.1,
        ],
        dtype=jnp.float32,
    )
    upper_bounds = jnp.asarray(
        [
            float(min(max_action_delay, 10)),
            float(min(max_observation_delay, 10)),
            1.03,
            0.005,
            0.005,
            1.06,
            1.15,
            1.4,
            1.0,
            1.4,
            1.25,
            1.15,
            0.1,
        ],
        dtype=jnp.float32,
    )

    def decode_population(
        population: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        bounded = lower_bounds + 0.5 * (jnp.tanh(population) + 1.0) * (
            upper_bounds - lower_bounds
        )
        action_delay = jnp.clip(
            jnp.rint(bounded[:, 0]), 0.0, float(max_action_delay)
        ).astype(jnp.int32)
        observation_delay = jnp.clip(
            jnp.rint(bounded[:, 1]), 0.0, float(max_observation_delay)
        ).astype(jnp.int32)
        continuous = bounded[:, 2:]
        return continuous, action_delay, observation_delay

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

    def init_data(model: mjx.Model) -> mjx.Data:
        def init_one(qpos: jax.Array, qvel: jax.Array, ctrl: jax.Array) -> mjx.Data:
            data = env._template_mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
            return env.forward(model, data)

        return jax.vmap(init_one)(initial_qpos, zero_qvel, zero_ctrl)

    def proposed_controls() -> jax.Array:
        zeros = jnp.zeros_like(controls_left.T)
        if is_chaser:
            return jnp.stack([controls_left.T, controls_right.T, zeros, zeros], axis=-1)
        return jnp.stack([zeros, zeros, controls_left.T, controls_right.T], axis=-1)

    proposed_ctrl = proposed_controls()

    def extract_pose(data: mjx.Data) -> jax.Array:
        chaser, evader = env._agent_pair_kinematics(data)
        agent = chaser if is_chaser else evader
        return jnp.asarray(
            [agent.position_xy[0], agent.position_xy[1], agent.yaw], dtype=jnp.float32
        )

    def evaluate_single_model(
        model: mjx.Model,
        action_delay_substeps: jax.Array,
        observation_delay_substeps: jax.Array,
    ) -> jax.Array:
        batched_data = init_data(model)
        action_buffer = jnp.zeros(
            (num_segments, max_action_delay + 1, env.mj_model.nu),
            dtype=jnp.float32,
        )
        observation_buffer = jnp.tile(
            dataset.initial_poses[:, None, :], (1, max_observation_delay + 1, 1)
        )

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
                applied = jnp.take(updated_buffer, action_delay_substeps, axis=1)
                inner_data = inner_data.replace(ctrl=applied)
                stepped = jax.vmap(lambda single: mjx.step(model, single))(inner_data)
                poses = jax.vmap(extract_pose)(stepped)
                updated_pose_buffer = jnp.concatenate(
                    [poses[:, None, :], inner_pose_buffer[:, :-1, :]], axis=1
                )
                observed = jnp.take(
                    updated_pose_buffer, observation_delay_substeps, axis=1
                )
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
        predicted = predicted.transpose(1, 0, 2)
        valid_prediction = jnp.all(jnp.isfinite(predicted))
        position_sq = jnp.sum(
            jnp.square(predicted[..., :2] - references[..., :2]), axis=-1
        )
        yaw_delta = jnp.arctan2(
            jnp.sin(predicted[..., 2] - references[..., 2]),
            jnp.cos(predicted[..., 2] - references[..., 2]),
        )
        yaw_sq = jnp.square(yaw_delta)
        predicted_yaw = jnp.concatenate(
            [initial_poses[:, 2:3], predicted[..., 2]], axis=1
        )
        reference_yaw = jnp.concatenate(
            [initial_poses[:, 2:3], references[..., 2]], axis=1
        )
        predicted_yaw_rate = env.config.action_frequency * jnp.arctan2(
            jnp.sin(predicted_yaw[:, 1:] - predicted_yaw[:, :-1]),
            jnp.cos(predicted_yaw[:, 1:] - predicted_yaw[:, :-1]),
        )
        reference_yaw_rate = env.config.action_frequency * jnp.arctan2(
            jnp.sin(reference_yaw[:, 1:] - reference_yaw[:, :-1]),
            jnp.cos(reference_yaw[:, 1:] - reference_yaw[:, :-1]),
        )
        yaw_rate_sq = jnp.square(predicted_yaw_rate - reference_yaw_rate)
        valid_steps = jnp.maximum(jnp.sum(mask), 1.0)
        position_rmse = jnp.sqrt(jnp.sum(position_sq * mask) / valid_steps)
        yaw_rmse = jnp.sqrt(jnp.sum(yaw_sq * mask) / valid_steps)
        yaw_rate_rmse = jnp.sqrt(jnp.sum(yaw_rate_sq * mask) / valid_steps)

        def family_rmse(
            squared_error: jax.Array, family_weight: float
        ) -> tuple[jax.Array, jax.Array]:
            segment_mask = jnp.asarray(
                segment_weights == family_weight, dtype=jnp.float32
            )[:, None]
            family_steps = jnp.sum(mask * segment_mask)
            rmse = jnp.sqrt(
                jnp.sum(squared_error * mask * segment_mask)
                / jnp.maximum(family_steps, 1.0)
            )
            return rmse, family_steps

        weighted_score = jnp.asarray(0.0, dtype=jnp.float32)
        active_weight = jnp.asarray(0.0, dtype=jnp.float32)
        for family_weight in SEGMENT_WEIGHTS.values():
            family_position_rmse, family_steps = family_rmse(position_sq, family_weight)
            family_yaw_rmse, _ = family_rmse(yaw_sq, family_weight)
            family_yaw_rate_rmse, _ = family_rmse(yaw_rate_sq, family_weight)
            family_normalized_yaw_rate_rmse = (
                family_yaw_rate_rmse / env.config.action_frequency
            )
            family_score = (
                0.5 * family_position_rmse
                + 0.3 * family_yaw_rmse
                + 0.2 * family_normalized_yaw_rate_rmse
            )
            is_active = jnp.asarray(family_steps > 0.0, dtype=jnp.float32)
            weighted_score = weighted_score + is_active * family_weight * family_score
            active_weight = active_weight + is_active * family_weight

        segment_counts = jnp.maximum(jnp.sum(mask, axis=1).astype(jnp.int32), 1)
        last_indices = segment_counts - 1
        endpoint_position_sq = jnp.take_along_axis(
            position_sq, last_indices[:, None], axis=1
        )[:, 0]
        endpoint_yaw = jnp.abs(
            jnp.take_along_axis(yaw_delta, last_indices[:, None], axis=1)[:, 0]
        )
        endpoint_position = jnp.mean(jnp.sqrt(endpoint_position_sq))
        endpoint_yaw_mean = jnp.mean(endpoint_yaw)
        score = weighted_score / jnp.maximum(active_weight, 1e-6)
        raw_metrics = jnp.asarray(
            [
                position_rmse,
                yaw_rmse,
                yaw_rate_rmse,
                endpoint_position,
                endpoint_yaw_mean,
                score,
                valid_steps,
            ],
            dtype=jnp.float32,
        )
        safe_metrics = jnp.asarray(
            [10.0, 10.0, 50.0, 10.0, 10.0, 1e6, valid_steps], dtype=jnp.float32
        )
        finite_metrics = jnp.all(jnp.isfinite(raw_metrics))
        return jnp.where(valid_prediction & finite_metrics, raw_metrics, safe_metrics)

    evaluate_population_raw_single = jax.jit(evaluate_single_model)
    evaluate_population_raw_batched = jax.jit(
        jax.vmap(evaluate_single_model, in_axes=(0, 0, 0))
    )

    def _make_sharded_eval(model_tree):
        return jax.pmap(
            lambda models, action_delays, observation_delays: jax.vmap(
                evaluate_single_model, in_axes=(0, 0, 0)
            )(models, action_delays, observation_delays),
            in_axes=(mjx_model_in_axes(model_tree), 0, 0),
        )

    def evaluate_population_raw(population: jax.Array) -> jax.Array:
        population_host = np.asarray(jax.device_get(population), dtype=np.float32)
        values, action_delays, observation_delays = decode_population(
            jnp.asarray(population_host)
        )
        values = np.asarray(jax.device_get(values), dtype=np.float32)
        action_delays = np.asarray(jax.device_get(action_delays), dtype=np.int32)
        observation_delays = np.asarray(
            jax.device_get(observation_delays), dtype=np.int32
        )
        params = [
            decoded_values_to_params(values[i], action_delays[i], observation_delays[i])
            for i in range(population_host.shape[0])
        ]
        domain_params_list: list[HostDomainParams] = [
            params_to_domain_params(param, dataset.role) for param in params
        ]
        models = build_mjx_models_process_parallel(
            env.model_xml,
            domain_params_list,
            max_workers=build_workers,
            executor=process_pool,
        )
        model_batch = stack_mjx_models(models)
        population_size = len(models)
        action_delays_jax = jnp.asarray(action_delays, dtype=jnp.int32)
        observation_delays_jax = jnp.asarray(observation_delays, dtype=jnp.int32)
        if device_count == 1:
            return evaluate_population_raw_batched(
                model_batch,
                action_delays_jax,
                observation_delays_jax,
            )

        remainder = population_size % device_count
        pad = (device_count - remainder) % device_count
        if pad > 0:
            model_batch = pad_stacked_mjx_models(model_batch, pad)
            action_delays_jax = jnp.pad(action_delays_jax, ((0, pad),))
            observation_delays_jax = jnp.pad(observation_delays_jax, ((0, pad),))

        sharded_models = reshape_stacked_mjx_models_for_devices(
            model_batch, device_count
        )
        shard_size = action_delays_jax.shape[0] // device_count
        sharded_action_delays = action_delays_jax.reshape(device_count, shard_size)
        sharded_observation_delays = observation_delays_jax.reshape(
            device_count, shard_size
        )
        devices = jax.local_devices()[:device_count]
        sharded_models = device_put_sharded_mjx_models(sharded_models, devices)
        sharded_action_delays = jax.device_put_sharded(
            [sharded_action_delays[i] for i in range(device_count)], devices
        )
        sharded_observation_delays = jax.device_put_sharded(
            [sharded_observation_delays[i] for i in range(device_count)], devices
        )
        raw = _make_sharded_eval(sharded_models)(
            sharded_models,
            sharded_action_delays,
            sharded_observation_delays,
        )
        raw = raw.reshape(device_count * shard_size, raw.shape[-1])
        return raw[:population_size]

    def evaluate_population(population: jax.Array) -> list[ReplayMetrics]:
        raw = evaluate_population_raw(population)
        return [
            ReplayMetrics(
                position_rmse_m=float(item[0]),
                yaw_rmse_rad=float(item[1]),
                yaw_rate_rmse_rad_s=float(item[2]),
                endpoint_position_error_m=float(item[3]),
                endpoint_yaw_error_rad=float(item[4]),
                score=float(item[5]),
                sample_count=int(item[6]),
            )
            for item in raw
        ]

    def evaluate_population_fitness(population: jax.Array) -> jax.Array:
        return evaluate_population_raw(population)[:, 5]

    search_bounds = {
        "action_delay_substeps": (0.0, float(min(max_action_delay, 10))),
        "observation_delay_substeps": (0.0, float(min(max_observation_delay, 10))),
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

    return DatasetEvaluator(
        decode_population=decode_population,
        evaluate_population=evaluate_population,
        evaluate_population_fitness=evaluate_population_fitness,
        population_size_hint=num_segments,
        search_bounds=search_bounds,
        device_count=device_count,
    )
