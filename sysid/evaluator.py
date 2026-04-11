from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from environment.config import EnvironmentConfig
from environment.environment import TagEnvironment
from environment.motor import apply_deadzone
from environment.mujoco_data import JointDofSlices, JointQposSlices, yaw_to_quaternion
from environment.randomization import randomize_model
from sysid.params import (
    MAX_COMMAND_DELAY_SUBSTEPS,
    MAX_OBSERVATION_DELAY_SUBSTEPS,
    build_domain_and_pipeline,
)
from sysid.types import EvaluationSummary, ScoreBreakdown, WindowBatch


LOCAL_POS_SCALE = 0.05
LOCAL_YAW_SCALE = 0.15
INCREMENT_HORIZONS_S = (0.2, 0.5, 1.0)
INCREMENT_POS_SCALES = (0.04, 0.06, 0.10)
INCREMENT_YAW_SCALES = (0.10, 0.14, 0.22)
INCREMENT_WEIGHTS = (0.45, 0.35, 0.20)
ANCHOR_HORIZONS_S = (0.5, 1.0)
ANCHOR_POS_SCALES = (0.07, 0.10)
ANCHOR_YAW_SCALES = (0.14, 0.20)
ANCHOR_WEIGHTS = (0.7, 0.3)
TRANSITION_THRESHOLD = 0.08
TRANSITION_RESPONSE_MIN_S = 0.05
TRANSITION_RESPONSE_MAX_S = 0.35
LOCAL_WEIGHT = 0.35
INCREMENT_WEIGHT = 0.35
TRANSITION_WEIGHT = 0.20
ANCHOR_WEIGHT = 0.10
PSEUDO_HUBER_DELTA = 1.0
EVADER_PARK_POS = jnp.array([5.0, 5.0, 0.0299], dtype=jnp.float32)
RAD_TO_DEG = 180.0 / jnp.pi


class WindowArrays(NamedTuple):
    relative_sample_times_s: jax.Array
    score_relative_times_s: jax.Array
    time_weights: jax.Array
    target_xy: jax.Array
    target_yaw: jax.Array
    sample_mask: jax.Array
    score_mask: jax.Array
    increment_future_indices: jax.Array
    increment_future_mask: jax.Array
    anchor_indices: jax.Array
    anchor_mask: jax.Array
    transition_sample_mask: jax.Array
    command_substeps: jax.Array
    initial_pose: jax.Array
    initial_velocity: jax.Array
    initial_command: jax.Array


class ScoreTerms(NamedTuple):
    total: jax.Array
    local: jax.Array
    increment: jax.Array
    transition: jax.Array
    anchor: jax.Array


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


def _piecewise_time_weights(score_relative_times_s: jax.Array) -> jax.Array:
    return jnp.where(
        score_relative_times_s < 0.5,
        1.0,
        jnp.where(
            score_relative_times_s < 1.5,
            0.7,
            jnp.where(score_relative_times_s < 3.0, 0.4, 0.2),
        ),
    ).astype(jnp.float32)


def _safe_weighted_mean(
    values: jax.Array, weights: jax.Array, mask: jax.Array
) -> jax.Array:
    masked_weights = weights * mask.astype(jnp.float32)
    return jnp.sum(values * masked_weights) / jnp.maximum(masked_weights.sum(), 1.0)


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

    def _build_future_lookup(
        self,
        relative_times_s,
        score_mask,
        horizons_s: tuple[float, ...],
    ) -> tuple[jax.Array, jax.Array]:
        num_windows, max_samples = relative_times_s.shape
        future_indices = np.zeros(
            (num_windows, max_samples, len(horizons_s)), dtype=np.int32
        )
        future_mask = np.zeros(
            (num_windows, max_samples, len(horizons_s)), dtype=np.bool_
        )
        for window_index in range(num_windows):
            valid_indices = [
                idx
                for idx, is_scored in enumerate(score_mask[window_index])
                if is_scored
            ]
            valid_times = [
                float(relative_times_s[window_index, idx]) for idx in valid_indices
            ]
            for source_position, source_index in enumerate(valid_indices):
                source_time = valid_times[source_position]
                for horizon_index, horizon_s in enumerate(horizons_s):
                    target_time = source_time + horizon_s
                    target_index = next(
                        (
                            valid_indices[pos]
                            for pos, valid_time in enumerate(valid_times)
                            if valid_time >= target_time
                        ),
                        None,
                    )
                    if target_index is None:
                        continue
                    future_indices[window_index, source_index, horizon_index] = (
                        target_index
                    )
                    future_mask[window_index, source_index, horizon_index] = True
        return jnp.asarray(future_indices), jnp.asarray(future_mask)

    def _build_anchor_lookup(
        self, relative_times_s, score_mask
    ) -> tuple[jax.Array, jax.Array]:
        num_windows = relative_times_s.shape[0]
        anchor_indices = np.zeros((num_windows, len(ANCHOR_HORIZONS_S)), dtype=np.int32)
        anchor_mask = np.zeros((num_windows, len(ANCHOR_HORIZONS_S)), dtype=np.bool_)
        for window_index in range(num_windows):
            valid_indices = [
                idx
                for idx, is_scored in enumerate(score_mask[window_index])
                if is_scored
            ]
            if not valid_indices:
                continue
            valid_times = [
                float(relative_times_s[window_index, idx]) for idx in valid_indices
            ]
            score_start_time = valid_times[0]
            for anchor_index, anchor_s in enumerate(ANCHOR_HORIZONS_S):
                target_index = next(
                    (
                        valid_indices[pos]
                        for pos, valid_time in enumerate(valid_times)
                        if valid_time >= score_start_time + anchor_s
                    ),
                    None,
                )
                if target_index is None:
                    continue
                anchor_indices[window_index, anchor_index] = target_index
                anchor_mask[window_index, anchor_index] = True
        return jnp.asarray(anchor_indices), jnp.asarray(anchor_mask)

    def _build_transition_sample_mask(self, batch: WindowBatch) -> jax.Array:
        num_windows, max_samples = batch.relative_sample_times_s.shape
        transition_mask = np.zeros((num_windows, max_samples), dtype=np.bool_)
        substep_times = (
            np.arange(batch.command_substeps.shape[1], dtype=np.float32)
            * self.substep_dt_seconds
        )
        for window_index in range(num_windows):
            commands = batch.command_substeps[window_index]
            deltas = np.linalg.norm(commands[1:] - commands[:-1], axis=-1)
            transition_times = substep_times[1:][deltas > TRANSITION_THRESHOLD]
            if len(transition_times) == 0:
                continue
            sample_times = batch.relative_sample_times_s[window_index]
            sample_score_mask = batch.score_mask[window_index]
            sample_transition = np.any(
                (
                    sample_times[:, None] - transition_times[None, :]
                    >= TRANSITION_RESPONSE_MIN_S
                )
                & (
                    sample_times[:, None] - transition_times[None, :]
                    <= TRANSITION_RESPONSE_MAX_S
                ),
                axis=1,
            )
            transition_mask[window_index] = sample_transition & sample_score_mask
        return jnp.asarray(transition_mask)

    def prepare_batch(self, batch: WindowBatch) -> WindowArrays:
        relative_sample_times_s = jnp.asarray(batch.relative_sample_times_s)
        score_mask = jnp.asarray(batch.score_mask)
        sample_mask = jnp.asarray(batch.sample_mask)
        score_start_time = jnp.min(
            jnp.where(score_mask, relative_sample_times_s, jnp.inf),
            axis=1,
            keepdims=True,
        )
        score_relative_times_s = jnp.maximum(
            relative_sample_times_s - score_start_time,
            0.0,
        )
        time_weights = _piecewise_time_weights(
            score_relative_times_s
        ) * score_mask.astype(jnp.float32)
        increment_future_indices, increment_future_mask = self._build_future_lookup(
            batch.relative_sample_times_s,
            batch.score_mask,
            INCREMENT_HORIZONS_S,
        )
        anchor_indices, anchor_mask = self._build_anchor_lookup(
            batch.relative_sample_times_s,
            batch.score_mask,
        )
        transition_sample_mask = self._build_transition_sample_mask(batch)
        return WindowArrays(
            relative_sample_times_s=relative_sample_times_s,
            score_relative_times_s=score_relative_times_s,
            time_weights=time_weights,
            target_xy=jnp.asarray(batch.target_xy),
            target_yaw=jnp.asarray(batch.target_yaw),
            sample_mask=sample_mask,
            score_mask=score_mask,
            increment_future_indices=increment_future_indices,
            increment_future_mask=increment_future_mask,
            anchor_indices=anchor_indices,
            anchor_mask=anchor_mask,
            transition_sample_mask=transition_sample_mask,
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
        motor_deadzone = domain_params.chaser.motor_deadzone

        def substep(
            carry: tuple[mjx.Data, jax.Array], proposed_action: jax.Array
        ) -> tuple[tuple[mjx.Data, jax.Array], jax.Array]:
            data, action_buffer = carry
            action_buffer = (
                jnp.roll(action_buffer, shift=-1, axis=0).at[-1].set(proposed_action)
            )
            delayed_action = action_buffer[-1 - pipeline_params.action_delay_substeps]
            motor_command = apply_deadzone(delayed_action, motor_deadzone)
            data = data.replace(
                ctrl=jnp.concatenate(
                    [motor_command, jnp.zeros((2,), dtype=jnp.float32)]
                )
            )
            data = mjx.step(model, data)
            return (data, action_buffer), _extract_pose(
                data.qpos[self.qpos_slices.chaser_root],
                data.qvel[self.dof_slices.chaser_root],
            )

        _, pose_history = jax.lax.scan(
            substep,
            (initial_data, initial_buffer),
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
        valid_mask = window.sample_mask & window.score_mask
        time_weights = window.time_weights
        pos_err = jnp.linalg.norm(sim_xy - window.target_xy, axis=-1)
        yaw_err = jnp.abs(_wrap_angle(sim_yaw - window.target_yaw))

        local_term = _safe_weighted_mean(
            _pseudo_huber(pos_err / LOCAL_POS_SCALE)
            + 0.7 * _pseudo_huber(yaw_err / LOCAL_YAW_SCALE),
            time_weights,
            valid_mask,
        )

        increment_weights = jnp.asarray(INCREMENT_WEIGHTS, dtype=jnp.float32)
        increment_pos_scales = jnp.asarray(INCREMENT_POS_SCALES, dtype=jnp.float32)
        increment_yaw_scales = jnp.asarray(INCREMENT_YAW_SCALES, dtype=jnp.float32)

        increment_terms = []
        for horizon_index in range(len(INCREMENT_HORIZONS_S)):
            future_indices = window.increment_future_indices[:, horizon_index]
            future_mask = valid_mask & window.increment_future_mask[:, horizon_index]
            sim_future_xy = sim_xy[future_indices]
            sim_future_yaw = sim_yaw[future_indices]
            target_future_xy = window.target_xy[future_indices]
            target_future_yaw = window.target_yaw[future_indices]
            dpos_err = jnp.linalg.norm(
                (sim_future_xy - sim_xy) - (target_future_xy - window.target_xy),
                axis=-1,
            )
            dyaw_err = jnp.abs(
                _wrap_angle(
                    (sim_future_yaw - sim_yaw) - (target_future_yaw - window.target_yaw)
                )
            )
            increment_terms.append(
                _safe_weighted_mean(
                    _pseudo_huber(dpos_err / increment_pos_scales[horizon_index])
                    + 0.7
                    * _pseudo_huber(dyaw_err / increment_yaw_scales[horizon_index]),
                    time_weights,
                    future_mask,
                )
            )
        increment_term = jnp.sum(increment_weights * jnp.stack(increment_terms))

        transition_local = _safe_weighted_mean(
            _pseudo_huber(pos_err / 0.04) + 0.8 * _pseudo_huber(yaw_err / 0.12),
            time_weights,
            valid_mask & window.transition_sample_mask,
        )
        transition_horizon_index = 1
        transition_future_indices = window.increment_future_indices[
            :, transition_horizon_index
        ]
        transition_future_mask = (
            valid_mask
            & window.transition_sample_mask
            & window.increment_future_mask[:, transition_horizon_index]
        )
        transition_dpos_err = jnp.linalg.norm(
            (sim_xy[transition_future_indices] - sim_xy)
            - (window.target_xy[transition_future_indices] - window.target_xy),
            axis=-1,
        )
        transition_dyaw_err = jnp.abs(
            _wrap_angle(
                (sim_yaw[transition_future_indices] - sim_yaw)
                - (window.target_yaw[transition_future_indices] - window.target_yaw)
            )
        )
        transition_increment = _safe_weighted_mean(
            _pseudo_huber(transition_dpos_err / 0.06)
            + 0.8 * _pseudo_huber(transition_dyaw_err / 0.14),
            time_weights,
            transition_future_mask,
        )
        transition_term = transition_local + 0.5 * transition_increment

        anchor_weights = jnp.asarray(ANCHOR_WEIGHTS, dtype=jnp.float32)
        anchor_pos_scales = jnp.asarray(ANCHOR_POS_SCALES, dtype=jnp.float32)
        anchor_yaw_scales = jnp.asarray(ANCHOR_YAW_SCALES, dtype=jnp.float32)
        anchor_pos_err = pos_err[window.anchor_indices]
        anchor_yaw_err = yaw_err[window.anchor_indices]
        anchor_loss = _pseudo_huber(
            anchor_pos_err / anchor_pos_scales
        ) + 0.7 * _pseudo_huber(anchor_yaw_err / anchor_yaw_scales)
        anchor_mask = window.anchor_mask.astype(jnp.float32)
        anchor_term = jnp.sum(anchor_weights * anchor_mask * anchor_loss) / jnp.maximum(
            jnp.sum(anchor_weights * anchor_mask), 1.0
        )

        return ScoreTerms(
            total=(
                LOCAL_WEIGHT * local_term
                + INCREMENT_WEIGHT * increment_term
                + TRANSITION_WEIGHT * transition_term
                + ANCHOR_WEIGHT * anchor_term
            ),
            local=local_term,
            increment=increment_term,
            transition=transition_term,
            anchor=anchor_term,
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
            "local": terms.local,
            "increment": terms.increment,
            "transition": terms.transition,
            "anchor": terms.anchor,
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
        local = jax.device_get(per_window_terms.score.local)
        increment = jax.device_get(per_window_terms.score.increment)
        transition = jax.device_get(per_window_terms.score.transition)
        anchor = jax.device_get(per_window_terms.score.anchor)
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
                local=float(local.mean()),
                increment=float(increment.mean()),
                transition=float(transition.mean()),
                anchor=float(anchor.mean()),
            ),
            metrics=metrics,
            maneuver_scores=maneuver_scores,
            windows=len(batch.metadata),
        )
