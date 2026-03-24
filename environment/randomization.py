from typing import cast

import jax
import jax.numpy as jnp
from jax import random

from environment.config import EnvironmentConfig
from environment.curriculum import curriculum_scale
from environment.geometry import circle_intersects_box
from environment.mujoco_data import yaw_to_quaternion
from environment.types import (
    AgentDynamicsParams,
    DomainParams,
    ObstacleState,
    PipelineParams,
)


def _scaled_range(
    min_value: float, max_value: float, scale: jax.Array
) -> tuple[jax.Array, jax.Array]:
    low = 1.0 + (min_value - 1.0) * scale
    high = 1.0 + (max_value - 1.0) * scale
    return low, high


def _sample_scale(
    rng: jax.Array, min_value: float, max_value: float, scale: jax.Array
) -> jax.Array:
    low, high = _scaled_range(min_value, max_value, scale)
    return random.uniform(rng, (), minval=low, maxval=high)


def _sample_centered(
    rng: jax.Array, max_abs_value: float, scale: jax.Array
) -> jax.Array:
    return random.uniform(
        rng,
        (),
        minval=-max_abs_value * scale,
        maxval=max_abs_value * scale,
    )


def _sample_agent_dynamics(
    rng: jax.Array, config: EnvironmentConfig, scale: jax.Array
) -> AgentDynamicsParams:
    dyn = config.dynamics_randomization
    (
        mass_rng,
        com_x_rng,
        com_y_rng,
        track_rng,
        wheel_friction_rng,
        wheel_scrub_rng,
        caster_friction_rng,
        frictionloss_rng,
        strength_rng,
        back_emf_rng,
        balance_rng,
    ) = random.split(rng, 11)

    return AgentDynamicsParams(
        mass_scale=_sample_scale(
            mass_rng, dyn.chassis_mass_scale_min, dyn.chassis_mass_scale_max, scale
        ),
        com_offset_xy=jnp.array(
            [
                _sample_centered(com_x_rng, dyn.com_offset_x_max, scale),
                _sample_centered(com_y_rng, dyn.com_offset_y_max, scale),
            ]
        ),
        track_width_scale=_sample_scale(
            track_rng,
            dyn.track_width_scale_min,
            dyn.track_width_scale_max,
            scale,
        ),
        wheel_friction_scale=_sample_scale(
            wheel_friction_rng,
            dyn.wheel_friction_scale_min,
            dyn.wheel_friction_scale_max,
            scale,
        ),
        wheel_scrub_scale=_sample_scale(
            wheel_scrub_rng,
            dyn.wheel_scrub_scale_min,
            dyn.wheel_scrub_scale_max,
            scale,
        ),
        caster_friction_scale=_sample_scale(
            caster_friction_rng,
            dyn.caster_friction_scale_min,
            dyn.caster_friction_scale_max,
            scale,
        ),
        wheel_frictionloss_scale=_sample_scale(
            frictionloss_rng,
            dyn.wheel_frictionloss_scale_min,
            dyn.wheel_frictionloss_scale_max,
            scale,
        ),
        motor_strength_scale=_sample_scale(
            strength_rng,
            dyn.motor_strength_scale_min,
            dyn.motor_strength_scale_max,
            scale,
        ),
        back_emf_scale=_sample_scale(
            back_emf_rng,
            dyn.back_emf_scale_min,
            dyn.back_emf_scale_max,
            scale,
        ),
        motor_balance=_sample_centered(balance_rng, dyn.motor_balance_delta_max, scale),
    )


def sample_domain_params(
    rng: jax.Array, config: EnvironmentConfig, curriculum_progress: jax.Array
) -> DomainParams:
    if not config.dynamics_randomization.enabled:
        scale = jnp.asarray(0.0, dtype=jnp.float32)
    else:
        scale = curriculum_scale(
            curriculum_progress,
            config.curriculum.dynamics_start,
            config.curriculum.full_progress,
            config.curriculum.exponent,
        )
    chaser_rng, evader_rng = random.split(rng, 2)
    return DomainParams(
        chaser=_sample_agent_dynamics(chaser_rng, config, scale),
        evader=_sample_agent_dynamics(evader_rng, config, scale),
    )


def domain_params_distance(
    domain_params: DomainParams, config: EnvironmentConfig
) -> jax.Array:
    dyn = config.dynamics_randomization

    def agent_distance(agent: AgentDynamicsParams) -> jax.Array:
        parts = jnp.asarray(
            [
                (agent.mass_scale - 1.0) / max(dyn.chassis_mass_scale_max - 1.0, 1e-6),
                agent.com_offset_xy[0] / max(dyn.com_offset_x_max, 1e-6),
                agent.com_offset_xy[1] / max(dyn.com_offset_y_max, 1e-6),
                (agent.track_width_scale - 1.0)
                / max(dyn.track_width_scale_max - 1.0, 1e-6),
                (agent.wheel_friction_scale - 1.0)
                / max(dyn.wheel_friction_scale_max - 1.0, 1e-6),
                (agent.wheel_scrub_scale - 1.0)
                / max(dyn.wheel_scrub_scale_max - 1.0, 1e-6),
                (agent.caster_friction_scale - 1.0)
                / max(abs(dyn.caster_friction_scale_min - 1.0), 1e-6),
                (agent.wheel_frictionloss_scale - 1.0)
                / max(dyn.wheel_frictionloss_scale_max - 1.0, 1e-6),
                (agent.motor_strength_scale - 1.0)
                / max(dyn.motor_strength_scale_max - 1.0, 1e-6),
                (agent.back_emf_scale - 1.0)
                / max(abs(dyn.back_emf_scale_min - 1.0), 1e-6),
                agent.motor_balance / max(dyn.motor_balance_delta_max, 1e-6),
            ],
            dtype=jnp.float32,
        )
        return jnp.linalg.norm(parts)

    return 0.5 * (
        agent_distance(domain_params.chaser) + agent_distance(domain_params.evader)
    )


def sample_pipeline_params(
    rng: jax.Array, config: EnvironmentConfig, curriculum_progress: jax.Array
) -> PipelineParams:
    if not config.pipeline_randomization.enabled:
        scale = jnp.asarray(0.0, dtype=jnp.float32)
    else:
        scale = curriculum_scale(
            curriculum_progress,
            config.curriculum.pipeline_start,
            config.curriculum.full_progress,
            config.curriculum.exponent,
        )
    (
        action_delay_rng,
        obs_delay_rng,
        action_drop_rng,
        frame_drop_rng,
        stale_rng,
        position_noise_rng,
        yaw_noise_rng,
        stale_steps_rng,
    ) = random.split(rng, 8)
    pipe = config.pipeline_randomization
    max_action_delay = jnp.int32(jnp.floor(scale * pipe.max_action_delay_substeps))
    max_observation_delay = jnp.int32(
        jnp.floor(scale * pipe.max_observation_delay_substeps)
    )
    max_stale_steps = jnp.int32(
        jnp.maximum(0, jnp.floor(scale * pipe.max_stale_observation_steps))
    )

    sampled_max_stale_steps = jax.lax.cond(
        max_stale_steps > 0,
        lambda _: random.randint(stale_steps_rng, (), 1, max_stale_steps + 1),
        lambda _: jnp.int32(0),
        operand=None,
    )

    action_delay_substeps = random.randint(
        action_delay_rng, (), 0, max_action_delay + 1
    )
    observation_delay_substeps = random.randint(
        obs_delay_rng, (), 0, max_observation_delay + 1
    )
    return PipelineParams(
        action_delay_substeps=jnp.int32(action_delay_substeps),
        observation_delay_substeps=jnp.int32(observation_delay_substeps),
        action_drop_probability=random.uniform(
            action_drop_rng,
            (),
            minval=0.0,
            maxval=scale * pipe.action_drop_probability_max,
        ),
        frame_drop_probability=random.uniform(
            frame_drop_rng,
            (),
            minval=0.0,
            maxval=scale * pipe.frame_drop_probability_max,
        ),
        stale_observation_probability=random.uniform(
            stale_rng,
            (),
            minval=0.0,
            maxval=scale * pipe.stale_observation_probability_max,
        ),
        position_noise_std=random.uniform(
            position_noise_rng,
            (),
            minval=0.0,
            maxval=scale * pipe.position_noise_std_max,
        ),
        yaw_noise_std=random.uniform(
            yaw_noise_rng, (), minval=0.0, maxval=scale * pipe.yaw_noise_std_max
        ),
        max_stale_steps=jnp.int32(sampled_max_stale_steps),
    )


def sample_layout_state(
    rng: jax.Array, config: EnvironmentConfig, curriculum_progress: jax.Array
) -> ObstacleState:
    if config.max_obstacles == 0:
        return ObstacleState(
            positions_xy=jnp.zeros((0, 2), dtype=jnp.float32),
            yaws=jnp.zeros((0,), dtype=jnp.float32),
            active=jnp.zeros((0,), dtype=bool),
        )

    layout_scale = curriculum_scale(
        curriculum_progress,
        config.curriculum.layout_start,
        config.curriculum.full_progress,
        config.curriculum.exponent,
    )
    (
        key_obs_count,
        key_obs_pos,
        key_obs_yaw,
    ) = random.split(rng, 3)

    obstacle_cfg = config.layout_randomization
    available_max_obstacles = jnp.int32(
        jnp.floor(layout_scale * obstacle_cfg.max_obstacles)
    )
    n_active = random.randint(key_obs_count, (), 0, available_max_obstacles + 1)
    obstacle_active = jnp.arange(config.max_obstacles) < n_active

    obstacle_half_extent = config.obstacle_width / 2
    obstacle_clearance_radius = jnp.sqrt(2.0) * obstacle_half_extent
    obstacle_margin = obstacle_clearance_radius
    obstacle_lower = jnp.array(
        [
            -(config.arena_width / 2) + obstacle_margin,
            -(config.arena_height / 2) + obstacle_margin,
        ]
    )
    obstacle_upper = jnp.array(
        [
            (config.arena_width / 2) - obstacle_margin,
            (config.arena_height / 2) - obstacle_margin,
        ]
    )

    obstacle_pool = random.uniform(
        key_obs_pos,
        (obstacle_cfg.obstacle_candidate_pool, 2),
        minval=obstacle_lower,
        maxval=obstacle_upper,
    )

    min_obstacle_separation = (
        obstacle_cfg.obstacle_separation_factor * obstacle_clearance_radius
    )

    def choose_obstacles(
        carry: tuple[jax.Array, jax.Array], candidate_xy: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        chosen, count = carry
        distances = jnp.linalg.norm(chosen - candidate_xy, axis=1)
        valid_slots = jnp.arange(config.max_obstacles) < count
        masked_distances = jnp.asarray(jnp.where(valid_slots, distances, jnp.inf))
        min_distance = cast(jax.Array, masked_distances.min())
        accept = (count == 0) | (min_distance >= min_obstacle_separation)
        write_index = jnp.minimum(count, config.max_obstacles - 1)
        chosen = jnp.where(
            accept & (count < config.max_obstacles),
            chosen.at[write_index].set(candidate_xy),
            chosen,
        )
        count = count + (accept & (count < config.max_obstacles))
        return (chosen, count), None

    initial_obstacles = jnp.zeros((config.max_obstacles, 2))
    (obstacle_positions_xy, _), _ = jax.lax.scan(
        choose_obstacles, (initial_obstacles, jnp.int32(0)), obstacle_pool
    )
    obstacle_yaws = random.uniform(
        key_obs_yaw, (config.max_obstacles,), minval=-jnp.pi, maxval=jnp.pi
    )
    return ObstacleState(
        positions_xy=obstacle_positions_xy,
        yaws=obstacle_yaws,
        active=obstacle_active,
    )


def sample_spawn_state(
    rng: jax.Array,
    config: EnvironmentConfig,
    obstacle_state: ObstacleState,
    model_nq: int,
    model_nv: int,
    chaser_root_qpos_adr: int,
    evader_root_qpos_adr: int,
    chaser_caster_qpos_slice: slice,
    evader_caster_qpos_slice: slice,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    (
        key_chaser_candidates,
        key_evader_candidates,
        key_chaser_yaw,
        key_evader_yaw,
        key_separation_fallback,
    ) = random.split(rng, 5)

    obstacle_half_extent = config.obstacle_width / 2
    agent_margin = config.wall_margin_factor * config.agent_radius
    agent_lower = jnp.array(
        [
            -(config.arena_width / 2) + agent_margin,
            -(config.arena_height / 2) + agent_margin,
        ]
    )
    agent_upper = jnp.array(
        [
            (config.arena_width / 2) - agent_margin,
            (config.arena_height / 2) - agent_margin,
        ]
    )

    candidate_pool = config.layout_randomization.spawn_candidate_pool
    chaser_candidates = random.uniform(
        key_chaser_candidates,
        (candidate_pool, 2),
        minval=agent_lower,
        maxval=agent_upper,
    )
    evader_candidates = random.uniform(
        key_evader_candidates,
        (candidate_pool, 2),
        minval=agent_lower,
        maxval=agent_upper,
    )

    def choose_agent(candidates: jax.Array) -> jax.Array:
        overlaps = jax.vmap(
            lambda candidate: jax.vmap(
                circle_intersects_box, in_axes=(None, None, 0, 0, None)
            )(
                candidate,
                config.agent_radius,
                obstacle_state.positions_xy,
                obstacle_state.yaws,
                obstacle_half_extent,
            )
        )(candidates)
        blocked = jnp.any(overlaps & obstacle_state.active[None, :], axis=1)
        valid = ~blocked
        idx = jnp.argmax(valid)
        return candidates[idx]

    chaser_xy = choose_agent(chaser_candidates)
    evader_xy = choose_agent(evader_candidates)

    delta = evader_xy - chaser_xy
    dist = cast(jax.Array, jnp.linalg.norm(delta))
    fallback_dir = random.uniform(
        key_separation_fallback, (2,), minval=-1.0, maxval=1.0
    )
    fallback_dir = fallback_dir / jnp.maximum(jnp.linalg.norm(fallback_dir), 1e-6)
    direction = cast(jax.Array, jnp.where(dist < 1e-6, fallback_dir, delta / dist))

    too_close = dist < config.minimum_starting_separation
    pushed_evader = jnp.clip(
        chaser_xy + direction * config.minimum_starting_separation,
        agent_lower,
        agent_upper,
    )
    evader_xy = jnp.where(too_close, pushed_evader, evader_xy)

    new_dist = cast(jax.Array, jnp.linalg.norm(evader_xy - chaser_xy))
    still_too_close = new_dist < config.minimum_starting_separation
    pushed_chaser = jnp.clip(
        evader_xy - direction * config.minimum_starting_separation,
        agent_lower,
        agent_upper,
    )
    chaser_xy = jnp.where(still_too_close, pushed_chaser, chaser_xy)

    chaser_yaw = random.uniform(key_chaser_yaw, (), minval=-jnp.pi, maxval=jnp.pi)
    evader_yaw = random.uniform(key_evader_yaw, (), minval=-jnp.pi, maxval=jnp.pi)

    qpos = jnp.zeros(model_nq)
    qvel = jnp.zeros(model_nv)
    z = config.agent_z
    identity_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])

    qpos = qpos.at[chaser_root_qpos_adr : chaser_root_qpos_adr + 3].set(
        jnp.array([chaser_xy[0], chaser_xy[1], z])
    )
    qpos = qpos.at[chaser_root_qpos_adr + 3 : chaser_root_qpos_adr + 7].set(
        yaw_to_quaternion(chaser_yaw)
    )
    qpos = qpos.at[chaser_caster_qpos_slice].set(identity_quaternion)

    qpos = qpos.at[evader_root_qpos_adr : evader_root_qpos_adr + 3].set(
        jnp.array([evader_xy[0], evader_xy[1], z])
    )
    qpos = qpos.at[evader_root_qpos_adr + 3 : evader_root_qpos_adr + 7].set(
        yaw_to_quaternion(evader_yaw)
    )
    qpos = qpos.at[evader_caster_qpos_slice].set(identity_quaternion)

    initial_distance = jnp.linalg.norm(chaser_xy - evader_xy)
    return qpos, qvel, initial_distance.astype(jnp.float32)
