from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from environment.config import EnvironmentConfig
from environment.geometry import raycast_scene
from online.types import Pose2D, WorldState


class ObservationBuilder:
    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config
        ray_angles = jnp.linspace(0, 2 * jnp.pi, config.n_rays, endpoint=False)
        self._ray_directions = jnp.stack(
            [jnp.cos(ray_angles), jnp.sin(ray_angles)], axis=-1
        )
        self._max_steps = config.episode_max_length * config.action_frequency

    def _obstacle_arrays(
        self, world: WorldState
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        max_obstacles = self.config.max_obstacles
        positions = np.zeros((max_obstacles, 2), dtype=np.float32)
        yaws = np.zeros((max_obstacles,), dtype=np.float32)
        active = np.zeros((max_obstacles,), dtype=bool)
        for index, obstacle in enumerate(world.obstacles[:max_obstacles]):
            positions[index] = np.asarray(
                [obstacle.pose.x, obstacle.pose.y], dtype=np.float32
            )
            yaws[index] = np.float32(obstacle.pose.yaw)
            active[index] = True
        return jnp.asarray(positions), jnp.asarray(yaws), jnp.asarray(active)

    def _obs_for(
        self,
        self_pose: Pose2D,
        other_pose: Pose2D,
        world: WorldState,
        episode_progress: float,
    ) -> np.ndarray:
        obstacle_positions, obstacle_yaws, obstacle_active = self._obstacle_arrays(
            world
        )
        ray_distances, ray_types = raycast_scene(
            jnp.asarray([self_pose.x, self_pose.y], dtype=jnp.float32),
            self._ray_directions,
            jnp.asarray(self_pose.yaw, dtype=jnp.float32),
            jnp.asarray([other_pose.x, other_pose.y], dtype=jnp.float32),
            self.config.agent_radius,
            obstacle_positions,
            obstacle_yaws,
            obstacle_active,
            self.config,
        )
        obs = jnp.concatenate(
            [
                ray_distances,
                ray_types,
                jnp.asarray([episode_progress], dtype=jnp.float32),
            ]
        )
        return np.asarray(obs, dtype=np.float32)

    def build(
        self, world: WorldState, episode_step: int
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if world.chaser is None or world.chaser.filtered_pose is None:
            raise ValueError("missing chaser pose")
        if world.evader is None or world.evader.filtered_pose is None:
            raise ValueError("missing evader pose")
        progress = min(1.0, episode_step / max(1, self._max_steps))
        return (
            self._obs_for(
                world.chaser.filtered_pose, world.evader.filtered_pose, world, progress
            ),
            self._obs_for(
                world.evader.filtered_pose, world.chaser.filtered_pose, world, progress
            ),
            progress,
        )
