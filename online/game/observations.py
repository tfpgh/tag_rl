from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from environment.config import EnvironmentConfig
from environment.geometry import raycast_scene
from online.core.state import BoardState, Pose2D


@dataclass(frozen=True, slots=True)
class LiveObservations:
    chaser: jnp.ndarray
    evader: jnp.ndarray


def _pose_xy(pose: Pose2D) -> jnp.ndarray:
    return jnp.asarray([pose.x_m, pose.y_m], dtype=jnp.float32)


def _build_obstacle_arrays(
    board_state: BoardState, env_config: EnvironmentConfig
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    max_obstacles = env_config.max_obstacles
    positions = jnp.zeros((max_obstacles, 2), dtype=jnp.float32)
    yaws = jnp.zeros((max_obstacles,), dtype=jnp.float32)
    active = jnp.zeros((max_obstacles,), dtype=bool)
    for index, obstacle in enumerate(board_state.obstacles[:max_obstacles]):
        positions = positions.at[index].set(
            jnp.asarray([obstacle.pose.x_m, obstacle.pose.y_m], dtype=jnp.float32)
        )
        yaws = yaws.at[index].set(jnp.asarray(obstacle.pose.yaw_rad, dtype=jnp.float32))
        active = active.at[index].set(True)
    return positions, yaws, active


def build_live_observations(
    board_state: BoardState,
    env_config: EnvironmentConfig,
    episode_progress: float,
) -> LiveObservations | None:
    if not board_state.calibration.valid:
        return None
    if board_state.chaser is None or board_state.evader is None:
        return None
    if not board_state.chaser.visible or not board_state.evader.visible:
        return None

    chaser = board_state.chaser
    evader = board_state.evader
    obstacle_positions, obstacle_yaws, obstacle_active = _build_obstacle_arrays(
        board_state, env_config
    )
    ray_angles = jnp.linspace(0, 2 * jnp.pi, env_config.n_rays, endpoint=False)
    ray_directions = jnp.stack([jnp.cos(ray_angles), jnp.sin(ray_angles)], axis=-1)
    progress = jnp.asarray([episode_progress], dtype=jnp.float32)

    chaser_dist, chaser_type = raycast_scene(
        _pose_xy(chaser),
        ray_directions,
        jnp.asarray(chaser.yaw_rad, dtype=jnp.float32),
        _pose_xy(evader),
        env_config.agent_radius,
        obstacle_positions,
        obstacle_yaws,
        obstacle_active,
        env_config,
    )
    evader_dist, evader_type = raycast_scene(
        _pose_xy(evader),
        ray_directions,
        jnp.asarray(evader.yaw_rad, dtype=jnp.float32),
        _pose_xy(chaser),
        env_config.agent_radius,
        obstacle_positions,
        obstacle_yaws,
        obstacle_active,
        env_config,
    )
    return LiveObservations(
        chaser=jnp.concatenate([chaser_dist, chaser_type, progress]),
        evader=jnp.concatenate([evader_dist, evader_type, progress]),
    )
