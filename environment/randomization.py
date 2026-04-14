from typing import cast

import jax
import jax.numpy as jnp
from jax import random
from mujoco import mjx

from environment.config import EnvironmentConfig
from environment.curriculum import curriculum_scale
from environment.geometry import circle_intersects_box
from environment.mjcf import (
    CASTER_BODY_Z_OFFSET,
    CASTER_RADIUS,
    WHEEL_BODY_Z_OFFSET,
    WHEEL_RADIUS,
)
from environment.mujoco_data import yaw_to_quaternion
from environment.types import (
    AgentDynamicsParams,
    DomainParams,
    ModelIndices,
    ObstacleState,
    PipelineParams,
)


def _scaled_range(
    min_value: float, max_value: float, scale: jax.Array, nominal: float = 1.0
) -> tuple[jax.Array, jax.Array]:
    low = nominal + (min_value - nominal) * scale
    high = nominal + (max_value - nominal) * scale
    return low, high


def _sample_scale(
    rng: jax.Array,
    min_value: float,
    max_value: float,
    scale: jax.Array,
    nominal: float = 1.0,
) -> jax.Array:
    low, high = _scaled_range(min_value, max_value, scale, nominal)
    return random.uniform(rng, (), minval=low, maxval=high)


def _sample_centered(
    rng: jax.Array, max_abs_value: float, scale: jax.Array, nominal: float = 0.0
) -> jax.Array:
    return random.uniform(
        rng,
        (),
        minval=nominal - max_abs_value * scale,
        maxval=nominal + max_abs_value * scale,
    )


def _sample_agent_dynamics(
    rng: jax.Array, config: EnvironmentConfig, scale: jax.Array
) -> AgentDynamicsParams:
    dyn = config.dynamics_randomization
    (
        mass_rng,
        com_x_rng,
        com_z_rng,
        com_y_rng,
        track_rng,
        wheel_radius_rng,
        wheel_slide_friction_rng,
        wheel_torsional_friction_rng,
        wheel_rolling_friction_rng,
        body_contact_friction_rng,
        caster_radius_rng,
        caster_offset_rng,
        caster_slide_friction_rng,
        caster_torsional_friction_rng,
        wheel_damping_rng,
        wheel_frictionloss_rng,
        wheel_armature_rng,
        strength_rng,
        back_emf_rng,
        balance_rng,
        time_constant_rng,
    ) = random.split(rng, 21)

    return AgentDynamicsParams(
        mass_scale=_sample_scale(
            mass_rng, dyn.chassis_mass_scale_min, dyn.chassis_mass_scale_max, scale
        ),
        com_offset_xyz=jnp.array(
            [
                _sample_centered(
                    com_x_rng, dyn.com_offset_x_delta, scale, dyn.com_offset_x_nominal
                ),
                _sample_centered(com_y_rng, dyn.com_offset_y_max, scale),
                _sample_centered(
                    com_z_rng, dyn.com_offset_z_delta, scale, dyn.com_offset_z_nominal
                ),
            ]
        ),
        track_width_scale=_sample_scale(
            track_rng,
            dyn.track_width_scale_min,
            dyn.track_width_scale_max,
            scale,
            dyn.track_width_scale_nominal,
        ),
        wheel_radius_scale=_sample_scale(
            wheel_radius_rng,
            dyn.wheel_radius_scale_min,
            dyn.wheel_radius_scale_max,
            scale,
            dyn.wheel_radius_scale_nominal,
        ),
        wheel_slide_friction_scale=_sample_scale(
            wheel_slide_friction_rng,
            dyn.wheel_slide_friction_scale_min,
            dyn.wheel_slide_friction_scale_max,
            scale,
            dyn.wheel_slide_friction_scale_nominal,
        ),
        wheel_torsional_friction_scale=_sample_scale(
            wheel_torsional_friction_rng,
            dyn.wheel_torsional_friction_scale_min,
            dyn.wheel_torsional_friction_scale_max,
            scale,
            dyn.wheel_torsional_friction_scale_nominal,
        ),
        wheel_rolling_friction_scale=_sample_scale(
            wheel_rolling_friction_rng,
            dyn.wheel_rolling_friction_scale_min,
            dyn.wheel_rolling_friction_scale_max,
            scale,
            dyn.wheel_rolling_friction_scale_nominal,
        ),
        body_contact_friction_scale=_sample_scale(
            body_contact_friction_rng,
            dyn.body_contact_friction_scale_min,
            dyn.body_contact_friction_scale_max,
            scale,
            dyn.body_contact_friction_scale_nominal,
        ),
        caster_radius_scale=_sample_scale(
            caster_radius_rng,
            dyn.caster_radius_scale_min,
            dyn.caster_radius_scale_max,
            scale,
            dyn.caster_radius_scale_nominal,
        ),
        caster_offset_x=_sample_centered(
            caster_offset_rng,
            dyn.caster_offset_x_delta,
            scale,
            dyn.caster_offset_x_nominal,
        ),
        caster_slide_friction_scale=_sample_scale(
            caster_slide_friction_rng,
            dyn.caster_slide_friction_scale_min,
            dyn.caster_slide_friction_scale_max,
            scale,
            dyn.caster_slide_friction_scale_nominal,
        ),
        caster_torsional_friction_scale=_sample_scale(
            caster_torsional_friction_rng,
            dyn.caster_torsional_friction_scale_min,
            dyn.caster_torsional_friction_scale_max,
            scale,
            dyn.caster_torsional_friction_scale_nominal,
        ),
        wheel_joint_damping_scale=_sample_scale(
            wheel_damping_rng,
            dyn.wheel_joint_damping_scale_min,
            dyn.wheel_joint_damping_scale_max,
            scale,
            dyn.wheel_joint_damping_scale_nominal,
        ),
        wheel_joint_frictionloss_scale=_sample_scale(
            wheel_frictionloss_rng,
            dyn.wheel_joint_frictionloss_scale_min,
            dyn.wheel_joint_frictionloss_scale_max,
            scale,
            dyn.wheel_joint_frictionloss_scale_nominal,
        ),
        wheel_armature_scale=_sample_scale(
            wheel_armature_rng,
            dyn.wheel_armature_scale_min,
            dyn.wheel_armature_scale_max,
            scale,
            dyn.wheel_armature_scale_nominal,
        ),
        motor_strength_scale=_sample_scale(
            strength_rng,
            dyn.motor_strength_scale_min,
            dyn.motor_strength_scale_max,
            scale,
            dyn.motor_strength_scale_nominal,
        ),
        back_emf_scale=_sample_scale(
            back_emf_rng,
            dyn.back_emf_scale_min,
            dyn.back_emf_scale_max,
            scale,
            dyn.back_emf_scale_nominal,
        ),
        motor_balance=_sample_centered(
            balance_rng,
            dyn.motor_balance_delta_max,
            scale,
            dyn.motor_balance_nominal,
        ),
        motor_time_constant_seconds=_sample_scale(
            time_constant_rng,
            dyn.motor_time_constant_seconds_min,
            dyn.motor_time_constant_seconds_max,
            scale,
            dyn.motor_time_constant_seconds_nominal,
        ),
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


def sample_pipeline_params(
    rng: jax.Array, config: EnvironmentConfig, curriculum_progress: jax.Array
) -> PipelineParams:
    if not config.pipeline_randomization.enabled:
        return PipelineParams(
            action_delay_substeps=jnp.int32(0),
            observation_delay_substeps=jnp.int32(0),
            action_drop_probability=jnp.float32(0.0),
            frame_drop_probability=jnp.float32(0.0),
            stale_observation_probability=jnp.float32(0.0),
            position_noise_std=jnp.float32(0.0),
            yaw_noise_std=jnp.float32(0.0),
            max_stale_steps=jnp.int32(0),
        )
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
    delta_action = jnp.int32(jnp.floor(scale * pipe.action_delay_substeps_delta))
    min_action_delay = jnp.maximum(
        0, jnp.int32(pipe.nominal_action_delay_substeps) - delta_action
    )
    max_action_delay = jnp.int32(pipe.nominal_action_delay_substeps) + delta_action
    delta_obs = jnp.int32(jnp.floor(scale * pipe.observation_delay_substeps_delta))
    min_observation_delay = jnp.maximum(
        0, jnp.int32(pipe.nominal_observation_delay_substeps) - delta_obs
    )
    max_observation_delay = (
        jnp.int32(pipe.nominal_observation_delay_substeps) + delta_obs
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
        action_delay_rng, (), min_action_delay, max_action_delay + 1
    )
    observation_delay_substeps = random.randint(
        obs_delay_rng, (), min_observation_delay, max_observation_delay + 1
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
    available_max_obstacles = jnp.where(
        layout_scale > 0.0,
        jnp.int32(jnp.ceil(layout_scale * obstacle_cfg.max_obstacles)),
        jnp.int32(0),
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
    domain_params: DomainParams,
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
    chaser_z = jnp.maximum(
        WHEEL_BODY_Z_OFFSET + WHEEL_RADIUS * domain_params.chaser.wheel_radius_scale,
        CASTER_BODY_Z_OFFSET + CASTER_RADIUS * domain_params.chaser.caster_radius_scale,
    )
    evader_z = jnp.maximum(
        WHEEL_BODY_Z_OFFSET + WHEEL_RADIUS * domain_params.evader.wheel_radius_scale,
        CASTER_BODY_Z_OFFSET + CASTER_RADIUS * domain_params.evader.caster_radius_scale,
    )
    identity_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])

    qpos = qpos.at[chaser_root_qpos_adr : chaser_root_qpos_adr + 3].set(
        jnp.array([chaser_xy[0], chaser_xy[1], chaser_z])
    )
    qpos = qpos.at[chaser_root_qpos_adr + 3 : chaser_root_qpos_adr + 7].set(
        yaw_to_quaternion(chaser_yaw)
    )
    qpos = qpos.at[chaser_caster_qpos_slice].set(identity_quaternion)

    qpos = qpos.at[evader_root_qpos_adr : evader_root_qpos_adr + 3].set(
        jnp.array([evader_xy[0], evader_xy[1], evader_z])
    )
    qpos = qpos.at[evader_root_qpos_adr + 3 : evader_root_qpos_adr + 7].set(
        yaw_to_quaternion(evader_yaw)
    )
    qpos = qpos.at[evader_caster_qpos_slice].set(identity_quaternion)

    initial_distance = cast(jax.Array, jnp.linalg.norm(chaser_xy - evader_xy))
    return qpos, qvel, initial_distance


def _apply_agent_dynamics(
    base_model: mjx.Model,
    model_indices: ModelIndices,
    geom_friction: jax.Array,
    geom_size: jax.Array,
    body_mass: jax.Array,
    body_pos: jax.Array,
    body_ipos: jax.Array,
    body_inertia: jax.Array,
    dof_damping: jax.Array,
    dof_frictionloss: jax.Array,
    dof_armature: jax.Array,
    actuator_gainprm: jax.Array,
    actuator_biasprm: jax.Array,
    actuator_dynprm: jax.Array,
    agent_params: AgentDynamicsParams,
    agent_indices,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    left_motor_scale = agent_params.motor_strength_scale * (
        1.0 + agent_params.motor_balance
    )
    right_motor_scale = agent_params.motor_strength_scale * (
        1.0 - agent_params.motor_balance
    )
    left_emf_scale = agent_params.back_emf_scale * (1.0 + agent_params.motor_balance)
    right_emf_scale = agent_params.back_emf_scale * (1.0 - agent_params.motor_balance)

    for body_geom_id in (
        agent_indices.chassis_base_geom_id,
        agent_indices.bumper_geom_id,
    ):
        base_friction = base_model.geom_friction[body_geom_id]
        geom_friction = geom_friction.at[body_geom_id].set(
            jnp.maximum(
                1e-6,
                base_friction.at[0].set(
                    base_friction[0] * agent_params.body_contact_friction_scale
                ),
            )
        )

    for wheel_geom_id in (
        agent_indices.left_wheel_geom_id,
        agent_indices.right_wheel_geom_id,
    ):
        base_friction = base_model.geom_friction[wheel_geom_id]
        geom_friction = geom_friction.at[wheel_geom_id].set(
            jnp.maximum(
                1e-6,
                jnp.asarray(
                    [
                        base_friction[0] * agent_params.wheel_slide_friction_scale,
                        base_friction[1] * agent_params.wheel_torsional_friction_scale,
                        base_friction[2] * agent_params.wheel_rolling_friction_scale,
                    ],
                    dtype=jnp.float32,
                ),
            )
        )
        base_size = base_model.geom_size[wheel_geom_id]
        geom_size = geom_size.at[wheel_geom_id].set(
            base_size.at[0].set(base_size[0] * agent_params.wheel_radius_scale)
        )
    geom_friction = geom_friction.at[agent_indices.caster_geom_id].set(
        jnp.maximum(
            1e-6,
            jnp.asarray(
                [
                    base_model.geom_friction[agent_indices.caster_geom_id, 0]
                    * agent_params.caster_slide_friction_scale,
                    base_model.geom_friction[agent_indices.caster_geom_id, 1]
                    * agent_params.caster_torsional_friction_scale,
                    base_model.geom_friction[agent_indices.caster_geom_id, 2]
                    * agent_params.caster_torsional_friction_scale,
                ],
                dtype=jnp.float32,
            ),
        )
    )
    base_caster_size = base_model.geom_size[agent_indices.caster_geom_id]
    geom_size = geom_size.at[agent_indices.caster_geom_id].set(
        base_caster_size.at[0].set(
            base_caster_size[0] * agent_params.caster_radius_scale
        )
    )

    left_body_pos = base_model.body_pos[agent_indices.left_wheel_body_id]
    right_body_pos = base_model.body_pos[agent_indices.right_wheel_body_id]
    body_pos = body_pos.at[agent_indices.left_wheel_body_id].set(
        left_body_pos.at[1].set(left_body_pos[1] * agent_params.track_width_scale)
    )
    body_pos = body_pos.at[agent_indices.right_wheel_body_id].set(
        right_body_pos.at[1].set(right_body_pos[1] * agent_params.track_width_scale)
    )
    caster_body_pos = base_model.body_pos[agent_indices.caster_body_id]
    body_pos = body_pos.at[agent_indices.caster_body_id].set(
        caster_body_pos.at[0].set(caster_body_pos[0] + agent_params.caster_offset_x)
    )

    body_mass = body_mass.at[agent_indices.body_id].set(
        jnp.maximum(
            1e-6, base_model.body_mass[agent_indices.body_id] * agent_params.mass_scale
        )
    )
    base_ipos = base_model.body_ipos[agent_indices.body_id]
    body_ipos = body_ipos.at[agent_indices.body_id].set(
        base_ipos
        + jnp.array(
            [
                agent_params.com_offset_xyz[0],
                agent_params.com_offset_xyz[1],
                agent_params.com_offset_xyz[2],
            ],
            dtype=base_ipos.dtype,
        )
    )
    body_inertia = body_inertia.at[agent_indices.body_id].set(
        jnp.maximum(
            1e-7,
            base_model.body_inertia[agent_indices.body_id] * agent_params.mass_scale,
        )
    )

    for dof_id in (
        agent_indices.left_wheel_dof_id,
        agent_indices.right_wheel_dof_id,
    ):
        dof_damping = dof_damping.at[dof_id].set(
            jnp.maximum(
                1e-7,
                base_model.dof_damping[dof_id] * agent_params.wheel_joint_damping_scale,
            )
        )
        dof_frictionloss = dof_frictionloss.at[dof_id].set(
            jnp.maximum(
                1e-7,
                base_model.dof_frictionloss[dof_id]
                * agent_params.wheel_joint_frictionloss_scale,
            )
        )
        dof_armature = dof_armature.at[dof_id].set(
            jnp.maximum(
                1e-9,
                base_model.dof_armature[dof_id] * agent_params.wheel_armature_scale,
            )
        )

    actuator_gainprm = actuator_gainprm.at[agent_indices.left_actuator_id, 0].set(
        base_model.actuator_gainprm[agent_indices.left_actuator_id, 0]
        * left_motor_scale
    )
    actuator_gainprm = actuator_gainprm.at[agent_indices.right_actuator_id, 0].set(
        base_model.actuator_gainprm[agent_indices.right_actuator_id, 0]
        * right_motor_scale
    )
    actuator_biasprm = actuator_biasprm.at[agent_indices.left_actuator_id, 2].set(
        base_model.actuator_biasprm[agent_indices.left_actuator_id, 2] * left_emf_scale
    )
    actuator_biasprm = actuator_biasprm.at[agent_indices.right_actuator_id, 2].set(
        base_model.actuator_biasprm[agent_indices.right_actuator_id, 2]
        * right_emf_scale
    )
    actuator_dynprm = actuator_dynprm.at[agent_indices.left_actuator_id, 0].set(
        jnp.maximum(1e-5, agent_params.motor_time_constant_seconds)
    )
    actuator_dynprm = actuator_dynprm.at[agent_indices.right_actuator_id, 0].set(
        jnp.maximum(1e-5, agent_params.motor_time_constant_seconds)
    )

    return (
        geom_friction,
        geom_size,
        body_mass,
        body_pos,
        body_ipos,
        body_inertia,
        dof_damping,
        dof_frictionloss,
        dof_armature,
        actuator_gainprm,
        actuator_biasprm,
        actuator_dynprm,
    )


def randomize_model(
    base_model: mjx.Model, domain_params: DomainParams, model_indices: ModelIndices
) -> mjx.Model:
    geom_friction = base_model.geom_friction
    geom_size = base_model.geom_size
    body_mass = base_model.body_mass
    body_pos = base_model.body_pos
    body_ipos = base_model.body_ipos
    body_inertia = base_model.body_inertia
    dof_damping = base_model.dof_damping
    dof_frictionloss = base_model.dof_frictionloss
    dof_armature = base_model.dof_armature
    actuator_gainprm = base_model.actuator_gainprm
    actuator_biasprm = base_model.actuator_biasprm
    actuator_dynprm = base_model.actuator_dynprm

    (
        geom_friction,
        geom_size,
        body_mass,
        body_pos,
        body_ipos,
        body_inertia,
        dof_damping,
        dof_frictionloss,
        dof_armature,
        actuator_gainprm,
        actuator_biasprm,
        actuator_dynprm,
    ) = _apply_agent_dynamics(
        base_model,
        model_indices,
        geom_friction,
        geom_size,
        body_mass,
        body_pos,
        body_ipos,
        body_inertia,
        dof_damping,
        dof_frictionloss,
        dof_armature,
        actuator_gainprm,
        actuator_biasprm,
        actuator_dynprm,
        domain_params.chaser,
        model_indices.chaser,
    )
    (
        geom_friction,
        geom_size,
        body_mass,
        body_pos,
        body_ipos,
        body_inertia,
        dof_damping,
        dof_frictionloss,
        dof_armature,
        actuator_gainprm,
        actuator_biasprm,
        actuator_dynprm,
    ) = _apply_agent_dynamics(
        base_model,
        model_indices,
        geom_friction,
        geom_size,
        body_mass,
        body_pos,
        body_ipos,
        body_inertia,
        dof_damping,
        dof_frictionloss,
        dof_armature,
        actuator_gainprm,
        actuator_biasprm,
        actuator_dynprm,
        domain_params.evader,
        model_indices.evader,
    )

    return cast(
        mjx.Model,
        base_model.tree_replace(
            {
                "geom_friction": geom_friction,
                "geom_size": geom_size,
                "body_mass": body_mass,
                "body_pos": body_pos,
                "body_ipos": body_ipos,
                "body_inertia": body_inertia,
                "dof_damping": dof_damping,
                "dof_frictionloss": dof_frictionloss,
                "dof_armature": dof_armature,
                "actuator_gainprm": actuator_gainprm,
                "actuator_biasprm": actuator_biasprm,
                "actuator_dynprm": actuator_dynprm,
            }
        ),
    )
