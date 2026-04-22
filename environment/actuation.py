from __future__ import annotations

import jax
import jax.numpy as jnp

from environment.mjcf import WHEEL_LATERAL_OFFSET, WHEEL_RADIUS
from environment.mujoco_data import quaternion_to_yaw
from environment.types import AgentDynamicsParams, AgentModelIndices


def _motor_curve(command: jax.Array, sharpness: jax.Array) -> jax.Array:
    sharpness = jnp.maximum(jnp.asarray(sharpness, dtype=jnp.float32), 0.0)
    command = jnp.asarray(command, dtype=jnp.float32)
    return jax.lax.cond(
        sharpness < 1e-4,
        lambda _: command,
        lambda _: jnp.tanh(sharpness * command) / jnp.tanh(sharpness),
        operand=None,
    )


def shape_agent_actions(
    proposed_action: jax.Array,
    qpos: jax.Array,
    qvel: jax.Array,
    agent_params: AgentDynamicsParams,
    agent_indices: AgentModelIndices,
    root_qpos_slice: slice,
    root_dof_slice: slice,
) -> jax.Array:
    shaped_action = _motor_curve(proposed_action, agent_params.motor_curve_sharpness)

    root_qpos = qpos[root_qpos_slice]
    root_qvel = qvel[root_dof_slice]
    yaw = quaternion_to_yaw(root_qpos[3:7])
    forward_velocity = root_qvel[0] * jnp.cos(yaw) + root_qvel[1] * jnp.sin(yaw)
    yaw_rate = root_qvel[5]

    wheel_radius = WHEEL_RADIUS * agent_params.wheel_radius_scale
    track_half_width = WHEEL_LATERAL_OFFSET * agent_params.track_width_scale
    wheel_ground_speed = jnp.array(
        [
            forward_velocity - yaw_rate * track_half_width,
            forward_velocity + yaw_rate * track_half_width,
        ],
        dtype=jnp.float32,
    )
    wheel_surface_speed = wheel_radius * jnp.array(
        [
            qvel[agent_indices.left_wheel_dof_id],
            qvel[agent_indices.right_wheel_dof_id],
        ],
        dtype=jnp.float32,
    )

    threshold = jnp.maximum(agent_params.traction_slip_threshold, 1e-4)
    slip_excess = jnp.maximum(
        0.0,
        jnp.abs(wheel_surface_speed - wheel_ground_speed) - threshold,
    )
    slip_ratio = slip_excess / jnp.maximum(jnp.abs(wheel_ground_speed), threshold)
    traction_loss = 1.0 / (1.0 + agent_params.traction_loss_strength * slip_ratio)
    traction_scale = jnp.maximum(agent_params.traction_min_scale, traction_loss)
    traction_scale = jnp.where(jnp.abs(shaped_action) > 1e-3, traction_scale, 1.0)
    return jnp.clip(shaped_action * traction_scale, -1.0, 1.0)
