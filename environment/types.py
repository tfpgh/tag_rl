from typing import NamedTuple

from mujoco import mjx

import jax


class AgentKinematics(NamedTuple):
    position_xy: jax.Array
    yaw: jax.Array
    forward_velocity: jax.Array
    angular_velocity: jax.Array


class ObstacleState(NamedTuple):
    positions_xy: jax.Array
    yaws: jax.Array
    active: jax.Array


class TagEnvironmentState(NamedTuple):
    mjx_data: mjx.Data
    obstacle_positions_xy: jax.Array
    obstacle_yaws: jax.Array
    obstacle_active: jax.Array
    step_count: jax.Array
    tagged: jax.Array
    prev_distance: jax.Array


class TagEnvironmentStepInfo(NamedTuple):
    tagged: jax.Array
    time_up: jax.Array
    distance: jax.Array
    step_count: jax.Array
    is_frozen: jax.Array
    chaser_collision: jax.Array
    evader_collision: jax.Array
