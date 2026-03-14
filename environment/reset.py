from typing import cast

import jax
import jax.numpy as jnp
from jax import random

from environment.config import EnvironmentConfig
from environment.geometry import circle_intersects_box
from environment.mujoco_data import yaw_to_quaternion
from environment.types import ObstacleState


def sample_reset_state(
    rng: jax.Array,
    config: EnvironmentConfig,
    model_nq: int,
    model_nv: int,
    chaser_root_qpos_adr: int,
    evader_root_qpos_adr: int,
    chaser_caster_qpos_slice: slice,
    evader_caster_qpos_slice: slice,
) -> tuple[jax.Array, jax.Array, jax.Array, ObstacleState]:
    (
        key_obs_count,
        key_obs_pos,
        key_obs_yaw,
        key_chaser_candidates,
        key_evader_candidates,
        key_chaser_yaw,
        key_evader_yaw,
        key_separation_fallback,
    ) = random.split(rng, 8)

    n_active = random.randint(key_obs_count, (), 0, config.max_obstacles + 1)
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

    n_pool = 64
    obstacle_pool = random.uniform(
        key_obs_pos, (n_pool, 2), minval=obstacle_lower, maxval=obstacle_upper
    )

    min_obstacle_separation = 2.2 * obstacle_clearance_radius

    def choose_obstacles(
        carry: tuple[jax.Array, jax.Array], candidate_xy: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        chosen, count = carry
        distances = jnp.linalg.norm(chosen - candidate_xy, axis=1)
        valid_slots = jnp.arange(config.max_obstacles) < count
        min_distance = jnp.min(jnp.where(valid_slots, distances, jnp.inf))
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
    obstacle_state = ObstacleState(
        positions_xy=obstacle_positions_xy,
        yaws=obstacle_yaws,
        active=obstacle_active,
    )

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

    n_candidates = 64
    chaser_candidates = random.uniform(
        key_chaser_candidates,
        (n_candidates, 2),
        minval=agent_lower,
        maxval=agent_upper,
    )
    evader_candidates = random.uniform(
        key_evader_candidates,
        (n_candidates, 2),
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
    dist = jnp.linalg.norm(delta)
    fallback_dir = random.uniform(
        key_separation_fallback, (2,), minval=-1.0, maxval=1.0
    )
    fallback_dir = fallback_dir / jnp.maximum(jnp.linalg.norm(fallback_dir), 1e-6)
    direction = jnp.where(dist < 1e-6, fallback_dir, delta / dist)

    too_close = dist < config.minimum_starting_separation
    pushed_evader = jnp.clip(
        chaser_xy + direction * config.minimum_starting_separation,
        agent_lower,
        agent_upper,
    )
    evader_xy = jnp.where(too_close, pushed_evader, evader_xy)

    new_dist = jnp.linalg.norm(evader_xy - chaser_xy)
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

    initial_distance = cast(jax.Array, jnp.linalg.norm(chaser_xy - evader_xy))
    return qpos, qvel, initial_distance, obstacle_state
