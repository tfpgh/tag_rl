from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from mujoco import mjx

from environment.config import EnvironmentConfig
from environment.environment import TagEnvironment
from environment.mujoco_data import yaw_to_quaternion
from environment.randomization import randomize_model
from environment.types import AgentDynamicsParams, DomainParams
from sysid.dataset import build_prepared_dataset
from sysid.types import PreparedDataset, ReplayMetrics, RunData


@dataclass(frozen=True, slots=True)
class NominalParameters:
    action_delay_steps: int = 0
    observation_delay_steps: int = 0
    floor_friction_scale: float = 1.0
    mass_scale: float = 1.0
    com_offset_x: float = 0.0
    com_offset_y: float = 0.0
    inertia_scale: float = 1.0
    wheel_friction_scale: float = 1.0
    caster_friction_scale: float = 1.0
    wheel_damping_scale: float = 1.0
    wheel_frictionloss_scale: float = 1.0
    motor_strength_scale: float = 1.0
    motor_balance: float = 0.0


def infer_target_role(run: RunData) -> str:
    if run.target_robot_tag_id == 5:
        return "evader"
    return "chaser"


def summarize_metrics(metrics: ReplayMetrics) -> dict[str, float]:
    return {
        "position_rmse_m": metrics.position_rmse_m,
        "yaw_rmse_rad": metrics.yaw_rmse_rad,
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
) -> ReplayMetrics:
    dataset = build_prepared_dataset([run], target_hz=target_hz)
    evaluator = make_dataset_evaluator(dataset, env_config=env_config)
    metrics = evaluator.evaluate_population(
        jnp.asarray([encode_params(params)], dtype=jnp.float32)
    )
    return metrics[0]


def encode_params(params: NominalParameters) -> jnp.ndarray:
    return jnp.asarray(
        [
            float(params.action_delay_steps),
            float(params.observation_delay_steps),
            params.floor_friction_scale,
            params.mass_scale,
            params.com_offset_x,
            params.com_offset_y,
            params.inertia_scale,
            params.wheel_friction_scale,
            params.caster_friction_scale,
            params.wheel_damping_scale,
            params.wheel_frictionloss_scale,
            params.motor_strength_scale,
            params.motor_balance,
        ],
        dtype=jnp.float32,
    )


def decoded_values_to_params(
    values: jax.Array,
    action_delay_steps: jax.Array,
    observation_delay_steps: jax.Array,
) -> NominalParameters:
    return NominalParameters(
        action_delay_steps=int(action_delay_steps),
        observation_delay_steps=int(observation_delay_steps),
        floor_friction_scale=float(values[0]),
        mass_scale=float(values[1]),
        com_offset_x=float(values[2]),
        com_offset_y=float(values[3]),
        inertia_scale=float(values[4]),
        wheel_friction_scale=float(values[5]),
        caster_friction_scale=float(values[6]),
        wheel_damping_scale=float(values[7]),
        wheel_frictionloss_scale=float(values[8]),
        motor_strength_scale=float(values[9]),
        motor_balance=float(values[10]),
    )


@dataclass(slots=True)
class DatasetEvaluator:
    decode_population: Callable
    evaluate_population: Callable
    evaluate_population_fitness: Callable
    population_size_hint: int
    search_bounds: dict[str, tuple[float, float]]


def make_dataset_evaluator(
    dataset: PreparedDataset,
    env_config: EnvironmentConfig | None = None,
) -> DatasetEvaluator:
    env = TagEnvironment(env_config or EnvironmentConfig())
    max_action_delay = max(env.config.pipeline_randomization.max_action_delay_steps, 3)
    max_observation_delay = max(
        env.config.pipeline_randomization.max_observation_delay_steps, 3
    )
    is_chaser = dataset.role == "chaser"
    num_segments = int(dataset.initial_poses.shape[0])
    zero_qvel = jnp.zeros((num_segments, env.mj_model.nv), dtype=jnp.float32)
    zero_ctrl = jnp.zeros((num_segments, env.mj_model.nu), dtype=jnp.float32)
    mask = dataset.mask
    references = dataset.references
    controls_left = dataset.controls[..., 0]
    controls_right = dataset.controls[..., 1]
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
            0.7,
            0.7,
            -0.006,
            -0.006,
            0.6,
            0.7,
            0.6,
            0.4,
            0.4,
            0.6,
            -0.2,
        ],
        dtype=jnp.float32,
    )
    upper_bounds = jnp.asarray(
        [
            float(max_action_delay),
            float(max_observation_delay),
            1.4,
            1.4,
            0.006,
            0.006,
            1.6,
            1.4,
            1.6,
            1.8,
            2.0,
            1.6,
            0.2,
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

    def default_agent() -> AgentDynamicsParams:
        one = jnp.asarray(1.0, dtype=jnp.float32)
        return AgentDynamicsParams(
            mass_scale=one,
            com_offset_xy=jnp.zeros(2, dtype=jnp.float32),
            inertia_scale=one,
            wheel_friction_scale=one,
            caster_friction_scale=one,
            wheel_damping_scale=one,
            wheel_frictionloss_scale=one,
            motor_strength_scale=one,
            motor_balance=jnp.asarray(0.0, dtype=jnp.float32),
        )

    parked_agent = default_agent()

    def domain_params(values: jax.Array) -> DomainParams:
        floor_friction_scale = values[0]
        agent = AgentDynamicsParams(
            mass_scale=values[1],
            com_offset_xy=jnp.asarray([values[2], values[3]], dtype=jnp.float32),
            inertia_scale=values[4],
            wheel_friction_scale=values[5],
            caster_friction_scale=values[6],
            wheel_damping_scale=values[7],
            wheel_frictionloss_scale=values[8],
            motor_strength_scale=values[9],
            motor_balance=values[10],
        )
        if is_chaser:
            return DomainParams(
                floor_friction_scale=floor_friction_scale,
                chaser=agent,
                evader=parked_agent,
            )
        return DomainParams(
            floor_friction_scale=floor_friction_scale, chaser=parked_agent, evader=agent
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

    def evaluate_one(
        values: jax.Array,
        action_delay_steps: jax.Array,
        observation_delay_steps: jax.Array,
    ) -> jax.Array:
        model = randomize_model(env.mjx_model, domain_params(values), env.model_indices)
        batched_data = init_data(model)
        action_buffer = jnp.zeros(
            (num_segments, max_action_delay + 1, env.mj_model.nu), dtype=jnp.float32
        )
        observation_buffer = jnp.tile(
            dataset.initial_poses[:, None, :], (1, max_observation_delay + 1, 1)
        )

        def rollout_step(
            carry: tuple[mjx.Data, jax.Array, jax.Array], proposed_t: jax.Array
        ) -> tuple[tuple[mjx.Data, jax.Array, jax.Array], jax.Array]:
            data, buffer, pose_buffer = carry
            updated_buffer = jnp.concatenate(
                [proposed_t[:, None, :], buffer[:, :-1, :]], axis=1
            )
            applied = jnp.take(updated_buffer, action_delay_steps, axis=1)
            data = data.replace(ctrl=applied)

            def physics_substep(current: mjx.Data, _: None) -> tuple[mjx.Data, None]:
                stepped = jax.vmap(lambda single: mjx.step(model, single))(current)
                return stepped, None

            data, _ = jax.lax.scan(
                physics_substep, data, None, length=env.substeps_per_action
            )
            poses = jax.vmap(extract_pose)(data)
            updated_pose_buffer = jnp.concatenate(
                [poses[:, None, :], pose_buffer[:, :-1, :]], axis=1
            )
            observed = jnp.take(updated_pose_buffer, observation_delay_steps, axis=1)
            return (data, updated_buffer, updated_pose_buffer), observed

        (_, _, _), predicted = jax.lax.scan(
            rollout_step,
            (batched_data, action_buffer, observation_buffer),
            proposed_ctrl,
        )
        predicted = predicted.transpose(1, 0, 2)
        position_sq = jnp.sum(
            jnp.square(predicted[..., :2] - references[..., :2]), axis=-1
        )
        yaw_delta = jnp.arctan2(
            jnp.sin(predicted[..., 2] - references[..., 2]),
            jnp.cos(predicted[..., 2] - references[..., 2]),
        )
        valid_steps = jnp.maximum(jnp.sum(mask), 1.0)
        position_rmse = jnp.sqrt(jnp.sum(position_sq * mask) / valid_steps)
        yaw_rmse = jnp.sqrt(jnp.sum(jnp.square(yaw_delta) * mask) / valid_steps)

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
        score = (
            position_rmse
            + 0.25 * yaw_rmse
            + 0.5 * endpoint_position
            + 0.1 * endpoint_yaw_mean
        )
        return jnp.asarray(
            [
                position_rmse,
                yaw_rmse,
                endpoint_position,
                endpoint_yaw_mean,
                score,
                valid_steps,
            ],
            dtype=jnp.float32,
        )

    evaluate_population_raw = jax.jit(jax.vmap(evaluate_one, in_axes=(0, 0, 0)))

    def evaluate_population(population: jax.Array) -> list[ReplayMetrics]:
        values, action_delays, observation_delays = decode_population(population)
        raw = evaluate_population_raw(values, action_delays, observation_delays)
        return [
            ReplayMetrics(
                position_rmse_m=float(item[0]),
                yaw_rmse_rad=float(item[1]),
                endpoint_position_error_m=float(item[2]),
                endpoint_yaw_error_rad=float(item[3]),
                score=float(item[4]),
                sample_count=int(item[5]),
            )
            for item in raw
        ]

    evaluate_population_fitness = jax.jit(
        lambda population: evaluate_population_raw(*decode_population(population))[:, 4]
    )

    search_bounds = {
        "action_delay_steps": (0.0, float(max_action_delay)),
        "observation_delay_steps": (0.0, float(max_observation_delay)),
        "floor_friction_scale": (0.7, 1.4),
        "mass_scale": (0.7, 1.4),
        "com_offset_x": (-0.006, 0.006),
        "com_offset_y": (-0.006, 0.006),
        "inertia_scale": (0.6, 1.6),
        "wheel_friction_scale": (0.7, 1.4),
        "caster_friction_scale": (0.6, 1.6),
        "wheel_damping_scale": (0.4, 1.8),
        "wheel_frictionloss_scale": (0.4, 2.0),
        "motor_strength_scale": (0.6, 1.6),
        "motor_balance": (-0.2, 0.2),
    }

    return DatasetEvaluator(
        decode_population=decode_population,
        evaluate_population=evaluate_population,
        evaluate_population_fitness=evaluate_population_fitness,
        population_size_hint=num_segments,
        search_bounds=search_bounds,
    )
