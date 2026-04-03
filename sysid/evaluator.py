from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from environment.config import EnvironmentConfig
from environment.environment import TagEnvironment
from environment.motor import shape_motor_command
from environment.mujoco_data import JointDofSlices, JointQposSlices, yaw_to_quaternion
from environment.randomization import randomize_model
from sysid.params import (
    MAX_COMMAND_DELAY_SUBSTEPS,
    MAX_OBSERVATION_DELAY_SUBSTEPS,
    build_domain_and_pipeline,
)
from sysid.types import EvaluationSummary, ScoreBreakdown, WindowBatch


POS_SCALE = 0.05
YAW_SCALE = 0.15
VEL_SCALE = 0.25
YAW_RATE_SCALE = 1.0
POS_WEIGHT = 1.0
YAW_WEIGHT = 0.7
VEL_WEIGHT = 0.25
YAW_RATE_WEIGHT = 0.25
ENDPOINT_WEIGHT = 0.5
PSEUDO_HUBER_DELTA = 1.0
EVADER_PARK_POS = jnp.array([5.0, 5.0, 0.0299], dtype=jnp.float32)
RAD_TO_DEG = 180.0 / jnp.pi


class WindowArrays(NamedTuple):
    relative_sample_times_s: jax.Array
    target_xy: jax.Array
    target_yaw: jax.Array
    sample_mask: jax.Array
    score_mask: jax.Array
    command_substeps: jax.Array
    initial_pose: jax.Array
    initial_velocity: jax.Array
    initial_command: jax.Array


class ScoreTerms(NamedTuple):
    total: jax.Array
    position: jax.Array
    yaw: jax.Array
    linear_velocity: jax.Array
    angular_velocity: jax.Array
    endpoint: jax.Array


class WindowMetrics(NamedTuple):
    position_mae_m: jax.Array
    position_rmse_m: jax.Array
    yaw_mae_rad: jax.Array
    yaw_mae_deg: jax.Array
    endpoint_position_m: jax.Array
    endpoint_yaw_rad: jax.Array
    endpoint_yaw_deg: jax.Array


class SummaryTerms(NamedTuple):
    score: ScoreTerms
    metrics: WindowMetrics


def _wrap_angle(angle: jax.Array) -> jax.Array:
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def _pseudo_huber(value: jax.Array, delta: float = PSEUDO_HUBER_DELTA) -> jax.Array:
    scaled = value / delta
    return delta * delta * (jnp.sqrt(1.0 + scaled * scaled) - 1.0)


def _safe_masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    weight = mask.astype(jnp.float32)
    return jnp.sum(values * weight) / jnp.maximum(weight.sum(), 1.0)


def _extract_pose(root_qpos: jax.Array, root_qvel: jax.Array) -> jax.Array:
    return jnp.array(
        [
            root_qpos[0],
            root_qpos[1],
            jnp.arctan2(
                2.0 * (root_qpos[3] * root_qpos[6] + root_qpos[4] * root_qpos[5]),
                1.0 - 2.0 * (root_qpos[5] * root_qpos[5] + root_qpos[6] * root_qpos[6]),
            ),
            root_qvel[0],
            root_qvel[1],
            root_qvel[5],
        ],
        dtype=jnp.float32,
    )


@dataclass(slots=True)
class SysIdEvaluator:
    env: TagEnvironment
    qpos_slices: JointQposSlices
    dof_slices: JointDofSlices
    num_substeps: int
    substep_dt_seconds: float
    action_buffer_len: int
    devices: list[Any]

    @classmethod
    def create(cls, config: EnvironmentConfig, num_substeps: int) -> "SysIdEvaluator":
        env_config = EnvironmentConfig.from_dict(
            {
                "arena": {
                    "chaser_freeze_seconds": 0,
                    "episode_max_length": 30,
                },
                "layout_randomization": {"max_obstacles": 0},
                "pipeline_randomization": {
                    "enabled": False,
                    "max_action_delay_substeps": MAX_COMMAND_DELAY_SUBSTEPS,
                    "max_observation_delay_substeps": MAX_OBSERVATION_DELAY_SUBSTEPS,
                },
                "dynamics_randomization": {"enabled": False},
                "curriculum": {"enabled": False},
            }
        )
        env_config.arena.arena_width = config.arena_width
        env_config.arena.arena_height = config.arena_height
        env_config.arena.agent_radius = config.agent_radius
        env_config.arena.agent_z = config.agent_z
        env = TagEnvironment(env_config)
        return cls(
            env=env,
            qpos_slices=env.joint_qpos_slices,
            dof_slices=env.joint_dof_slices,
            num_substeps=num_substeps,
            substep_dt_seconds=float(env.mj_model.opt.timestep),
            action_buffer_len=MAX_COMMAND_DELAY_SUBSTEPS + 1,
            devices=list(jax.local_devices()),
        )

    def prepare_batch(self, batch: WindowBatch) -> WindowArrays:
        return WindowArrays(
            relative_sample_times_s=jnp.asarray(batch.relative_sample_times_s),
            target_xy=jnp.asarray(batch.target_xy),
            target_yaw=jnp.asarray(batch.target_yaw),
            sample_mask=jnp.asarray(batch.sample_mask),
            score_mask=jnp.asarray(batch.score_mask),
            command_substeps=jnp.asarray(batch.command_substeps),
            initial_pose=jnp.asarray(batch.initial_pose),
            initial_velocity=jnp.asarray(batch.initial_velocity),
            initial_command=jnp.asarray(batch.initial_command),
        )

    def _initial_data(
        self, model: mjx.Model, initial_pose: jax.Array, initial_velocity: jax.Array
    ) -> mjx.Data:
        qpos = jnp.zeros((self.env.mj_model.nq,), dtype=jnp.float32)
        qvel = jnp.zeros((self.env.mj_model.nv,), dtype=jnp.float32)
        chaser_quat = yaw_to_quaternion(initial_pose[2])
        qpos = qpos.at[self.qpos_slices.chaser_root].set(
            jnp.array(
                [
                    initial_pose[0],
                    initial_pose[1],
                    self.env.config.agent_z,
                    *chaser_quat,
                ],
                dtype=jnp.float32,
            )
        )
        evader_quat = yaw_to_quaternion(jnp.asarray(0.0, dtype=jnp.float32))
        qpos = qpos.at[self.qpos_slices.evader_root].set(
            jnp.array(
                [
                    EVADER_PARK_POS[0],
                    EVADER_PARK_POS[1],
                    EVADER_PARK_POS[2],
                    *evader_quat,
                ],
                dtype=jnp.float32,
            )
        )
        qpos = qpos.at[self.qpos_slices.chaser_caster_ball_joint].set(
            jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        )
        qpos = qpos.at[self.qpos_slices.evader_caster_ball_joint].set(
            jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        )
        qvel = qvel.at[self.dof_slices.chaser_root].set(
            jnp.array(
                [
                    initial_velocity[0],
                    initial_velocity[1],
                    0.0,
                    0.0,
                    0.0,
                    initial_velocity[2],
                ],
                dtype=jnp.float32,
            )
        )
        return self.env.forward(
            model, self.env._template_mjx_data.replace(qpos=qpos, qvel=qvel)
        )

    def _simulate_pose_history(
        self, latent: jax.Array, window: WindowArrays
    ) -> jax.Array:
        domain_params, pipeline_params = build_domain_and_pipeline(latent)
        model = randomize_model(
            self.env.mjx_model, domain_params, self.env.model_indices
        )
        initial_data = self._initial_data(
            model, window.initial_pose, window.initial_velocity
        )
        initial_buffer = jnp.repeat(
            window.initial_command[None, :], self.action_buffer_len, axis=0
        )
        initial_motor_command = jnp.zeros((2,), dtype=jnp.float32)
        motor_deadzone = domain_params.chaser.motor_deadzone
        motor_time_constant_seconds = domain_params.chaser.motor_time_constant_seconds

        def substep(
            carry: tuple[mjx.Data, jax.Array, jax.Array], proposed_action: jax.Array
        ) -> tuple[tuple[mjx.Data, jax.Array, jax.Array], jax.Array]:
            data, action_buffer, motor_command = carry
            action_buffer = (
                jnp.roll(action_buffer, shift=-1, axis=0).at[-1].set(proposed_action)
            )
            delayed_action = action_buffer[-1 - pipeline_params.action_delay_substeps]
            motor_command = shape_motor_command(
                motor_command,
                delayed_action,
                motor_deadzone,
                motor_time_constant_seconds,
                self.substep_dt_seconds,
            )
            data = data.replace(
                ctrl=jnp.concatenate(
                    [motor_command, jnp.zeros((2,), dtype=jnp.float32)]
                )
            )
            data = mjx.step(model, data)
            return (data, action_buffer, motor_command), _extract_pose(
                data.qpos[self.qpos_slices.chaser_root],
                data.qvel[self.dof_slices.chaser_root],
            )

        _, pose_history = jax.lax.scan(
            substep,
            (initial_data, initial_buffer, initial_motor_command),
            window.command_substeps,
            length=self.num_substeps,
        )
        sample_indices = jnp.clip(
            jnp.rint(window.relative_sample_times_s / self.substep_dt_seconds).astype(
                jnp.int32
            )
            - 1
            - pipeline_params.observation_delay_substeps,
            0,
            self.num_substeps - 1,
        )
        return pose_history[sample_indices]

    def _score_from_sampled_pose(
        self, sampled_pose: jax.Array, window: WindowArrays
    ) -> ScoreTerms:
        sim_xy = sampled_pose[:, :2]
        sim_yaw = sampled_pose[:, 2]
        sim_speed = jnp.linalg.norm(sampled_pose[:, 3:5], axis=-1)
        sim_yaw_rate = sampled_pose[:, 5]

        dt = jnp.diff(
            window.relative_sample_times_s, prepend=window.relative_sample_times_s[:1]
        )
        dt = jnp.maximum(dt, self.substep_dt_seconds)
        target_dxy = jnp.diff(window.target_xy, axis=0, prepend=window.target_xy[:1])
        target_speed = jnp.linalg.norm(target_dxy, axis=-1) / dt
        target_yaw_rate = (
            _wrap_angle(window.target_yaw - jnp.roll(window.target_yaw, 1)) / dt
        )

        valid_mask = window.sample_mask & window.score_mask
        pos_err = jnp.linalg.norm(sim_xy - window.target_xy, axis=-1)
        yaw_err = jnp.abs(_wrap_angle(sim_yaw - window.target_yaw))
        vel_err = jnp.abs(sim_speed - target_speed)
        yaw_rate_err = jnp.abs(sim_yaw_rate - target_yaw_rate)

        position_term = _safe_masked_mean(
            _pseudo_huber(pos_err / POS_SCALE), valid_mask
        )
        yaw_term = _safe_masked_mean(_pseudo_huber(yaw_err / YAW_SCALE), valid_mask)
        velocity_term = _safe_masked_mean(
            _pseudo_huber(vel_err / VEL_SCALE), valid_mask
        )
        yaw_rate_term = _safe_masked_mean(
            _pseudo_huber(yaw_rate_err / YAW_RATE_SCALE), valid_mask
        )
        last_index = jnp.maximum(jnp.sum(valid_mask.astype(jnp.int32)) - 1, 0)
        endpoint_term = _pseudo_huber(pos_err[last_index] / POS_SCALE) + _pseudo_huber(
            yaw_err[last_index] / YAW_SCALE
        )
        return ScoreTerms(
            total=(
                POS_WEIGHT * position_term
                + YAW_WEIGHT * yaw_term
                + VEL_WEIGHT * velocity_term
                + YAW_RATE_WEIGHT * yaw_rate_term
                + ENDPOINT_WEIGHT * endpoint_term
            ),
            position=position_term,
            yaw=yaw_term,
            linear_velocity=velocity_term,
            angular_velocity=yaw_rate_term,
            endpoint=endpoint_term,
        )

    def _simulate_window(self, latent: jax.Array, window: WindowArrays) -> ScoreTerms:
        sampled_pose = self._simulate_pose_history(latent, window)
        return self._score_from_sampled_pose(sampled_pose, window)

    def _summarize_window(
        self, latent: jax.Array, window: WindowArrays
    ) -> SummaryTerms:
        sampled_pose = self._simulate_pose_history(latent, window)
        score = self._score_from_sampled_pose(sampled_pose, window)
        valid_mask = window.sample_mask & window.score_mask
        pos_err = jnp.linalg.norm(sampled_pose[:, :2] - window.target_xy, axis=-1)
        yaw_err = jnp.abs(_wrap_angle(sampled_pose[:, 2] - window.target_yaw))
        last_index = jnp.maximum(jnp.sum(valid_mask.astype(jnp.int32)) - 1, 0)
        metrics = WindowMetrics(
            position_mae_m=_safe_masked_mean(pos_err, valid_mask),
            position_rmse_m=jnp.sqrt(_safe_masked_mean(pos_err * pos_err, valid_mask)),
            yaw_mae_rad=_safe_masked_mean(yaw_err, valid_mask),
            yaw_mae_deg=_safe_masked_mean(yaw_err, valid_mask) * RAD_TO_DEG,
            endpoint_position_m=pos_err[last_index],
            endpoint_yaw_rad=yaw_err[last_index],
            endpoint_yaw_deg=yaw_err[last_index] * RAD_TO_DEG,
        )
        return SummaryTerms(score=score, metrics=metrics)

    def _aggregate_score_terms(self, terms: ScoreTerms) -> ScoreTerms:
        return ScoreTerms(*(jnp.mean(component) for component in terms))

    def build_population_evaluator(self, windows: WindowArrays):
        simulate_all_windows = jax.vmap(self._simulate_window, in_axes=(None, 0))

        def evaluate_candidate(latent: jax.Array) -> ScoreTerms:
            return self._aggregate_score_terms(simulate_all_windows(latent, windows))

        if len(self.devices) == 1:
            return jax.jit(jax.vmap(evaluate_candidate))

        @partial(jax.pmap, devices=self.devices)
        def evaluate_sharded(latent_shard: jax.Array) -> ScoreTerms:
            return jax.vmap(evaluate_candidate)(latent_shard)

        return evaluate_sharded

    def build_summary_evaluator(self, windows: WindowArrays):
        return jax.jit(jax.vmap(self._summarize_window, in_axes=(None, 0)))

    def evaluate_population(
        self, latents: jax.Array, windows: WindowArrays, population_eval=None
    ) -> dict[str, jax.Array]:
        evaluate = population_eval or self.build_population_evaluator(windows)
        if len(self.devices) == 1:
            terms = evaluate(latents)
        else:
            num_devices = len(self.devices)
            population_size = int(latents.shape[0])
            shard_size = (population_size + num_devices - 1) // num_devices
            padded_size = shard_size * num_devices
            pad = padded_size - population_size
            if pad > 0:
                latents = jnp.pad(latents, ((0, pad), (0, 0)))
            sharded = latents.reshape(num_devices, shard_size, latents.shape[-1])
            sharded_terms = evaluate(sharded)
            terms = ScoreTerms(
                *(
                    component.reshape(padded_size)[:population_size]
                    for component in sharded_terms
                )
            )
        return {
            "total": terms.total,
            "position": terms.position,
            "yaw": terms.yaw,
            "linear_velocity": terms.linear_velocity,
            "angular_velocity": terms.angular_velocity,
            "endpoint": terms.endpoint,
        }

    def evaluate_summary(
        self,
        latent: jax.Array,
        batch: WindowBatch,
        windows: WindowArrays | None = None,
        summary_eval=None,
    ) -> EvaluationSummary:
        prepared = windows or self.prepare_batch(batch)
        evaluate = summary_eval or self.build_summary_evaluator(prepared)
        per_window_terms = evaluate(latent, prepared)
        total = jax.device_get(per_window_terms.score.total)
        position = jax.device_get(per_window_terms.score.position)
        yaw = jax.device_get(per_window_terms.score.yaw)
        linear_velocity = jax.device_get(per_window_terms.score.linear_velocity)
        angular_velocity = jax.device_get(per_window_terms.score.angular_velocity)
        endpoint = jax.device_get(per_window_terms.score.endpoint)
        metrics = {
            "position_mae_m": float(
                jax.device_get(per_window_terms.metrics.position_mae_m).mean()
            ),
            "position_rmse_m": float(
                jax.device_get(per_window_terms.metrics.position_rmse_m).mean()
            ),
            "yaw_mae_rad": float(
                jax.device_get(per_window_terms.metrics.yaw_mae_rad).mean()
            ),
            "yaw_mae_deg": float(
                jax.device_get(per_window_terms.metrics.yaw_mae_deg).mean()
            ),
            "endpoint_position_m": float(
                jax.device_get(per_window_terms.metrics.endpoint_position_m).mean()
            ),
            "endpoint_yaw_rad": float(
                jax.device_get(per_window_terms.metrics.endpoint_yaw_rad).mean()
            ),
            "endpoint_yaw_deg": float(
                jax.device_get(per_window_terms.metrics.endpoint_yaw_deg).mean()
            ),
        }
        maneuver_scores: dict[str, float] = {}
        for maneuver in sorted({item.maneuver for item in batch.metadata}):
            indices = [
                i for i, item in enumerate(batch.metadata) if item.maneuver == maneuver
            ]
            maneuver_scores[maneuver] = float(total[indices].mean())
        return EvaluationSummary(
            score=ScoreBreakdown(
                total=float(total.mean()),
                position=float(position.mean()),
                yaw=float(yaw.mean()),
                linear_velocity=float(linear_velocity.mean()),
                angular_velocity=float(angular_velocity.mean()),
                endpoint=float(endpoint.mean()),
            ),
            metrics=metrics,
            maneuver_scores=maneuver_scores,
            windows=len(batch.metadata),
        )
