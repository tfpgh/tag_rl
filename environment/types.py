from typing import NamedTuple

import jax
from mujoco import mjx


class AgentKinematics(NamedTuple):
    position_xy: jax.Array
    yaw: jax.Array
    forward_velocity: jax.Array
    angular_velocity: jax.Array


class ObstacleState(NamedTuple):
    positions_xy: jax.Array
    yaws: jax.Array
    active: jax.Array


class AgentDynamicsParams(NamedTuple):
    mass_scale: jax.Array
    com_offset_xy: jax.Array
    track_width_scale: jax.Array
    wheel_longitudinal_offset: jax.Array
    wheel_radius_scale: jax.Array
    wheel_slide_friction_scale: jax.Array
    wheel_slide_friction_balance: jax.Array
    wheel_torsional_friction_scale: jax.Array
    wheel_rolling_friction_scale: jax.Array
    caster_radius_scale: jax.Array
    caster_offset_x: jax.Array
    caster_slide_friction_scale: jax.Array
    caster_torsional_friction_scale: jax.Array
    wheel_joint_damping_scale: jax.Array
    wheel_joint_damping_balance: jax.Array
    wheel_joint_frictionloss_scale: jax.Array
    wheel_joint_frictionloss_balance: jax.Array
    wheel_armature_scale: jax.Array
    motor_strength_scale: jax.Array
    back_emf_scale: jax.Array
    motor_balance: jax.Array
    motor_deadzone: jax.Array
    motor_time_constant_seconds: jax.Array


class DomainParams(NamedTuple):
    chaser: AgentDynamicsParams
    evader: AgentDynamicsParams


class PipelineParams(NamedTuple):
    action_delay_substeps: jax.Array
    observation_delay_substeps: jax.Array
    action_drop_probability: jax.Array
    frame_drop_probability: jax.Array
    stale_observation_probability: jax.Array
    position_noise_std: jax.Array
    yaw_noise_std: jax.Array
    max_stale_steps: jax.Array


class AgentModelIndices(NamedTuple):
    body_id: int
    left_wheel_body_id: int
    right_wheel_body_id: int
    left_wheel_geom_id: int
    right_wheel_geom_id: int
    caster_body_id: int
    caster_geom_id: int
    left_wheel_dof_id: int
    right_wheel_dof_id: int
    left_actuator_id: int
    right_actuator_id: int


class ModelIndices(NamedTuple):
    floor_geom_id: int
    chaser: AgentModelIndices
    evader: AgentModelIndices


class TagEnvironmentState(NamedTuple):
    mjx_data: mjx.Data
    domain_params: DomainParams
    pipeline_params: PipelineParams
    obstacle_positions_xy: jax.Array
    obstacle_yaws: jax.Array
    obstacle_active: jax.Array
    observation_buffer: jax.Array
    action_buffer: jax.Array
    last_applied_actions: jax.Array
    last_measured_agent_positions_xy: jax.Array
    last_measured_agent_yaws: jax.Array
    last_measured_obstacle_positions_xy: jax.Array
    last_measured_obstacle_yaws: jax.Array
    stale_observation_steps_remaining: jax.Array
    step_count: jax.Array
    prev_distance: jax.Array


class TagEnvironmentStepInfo(NamedTuple):
    tagged: jax.Array
    time_up: jax.Array
    distance: jax.Array
    step_count: jax.Array
    obstacle_count: jax.Array
    chaser_collision: jax.Array
    evader_collision: jax.Array
    action_delay_substeps: jax.Array
    observation_delay_substeps: jax.Array
    action_drop_probability: jax.Array
    frame_drop_probability: jax.Array
    stale_observation_probability: jax.Array
    position_noise_std: jax.Array
    yaw_noise_std: jax.Array
    chaser_mass_scale: jax.Array
    evader_mass_scale: jax.Array
    chaser_motor_balance: jax.Array
    evader_motor_balance: jax.Array
    chaser_motor_strength_scale: jax.Array
    evader_motor_strength_scale: jax.Array
