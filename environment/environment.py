from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from environment.config import EnvironmentConfig
from environment.geometry import (
    circle_hits_any_obstacle,
    point_out_of_bounds,
    raycast_scene,
)
from environment.mjcf import generate_mjcf
from environment.mujoco_data import JointDofSlices, JointQposSlices, quaternion_to_yaw
from environment.randomization import (
    randomize_model,
    sample_domain_params,
    sample_layout_state,
    sample_pipeline_params,
    sample_spawn_state,
)
from environment.types import (
    AgentKinematics,
    AgentModelIndices,
    ModelIndices,
    TagEnvironmentState,
    TagEnvironmentStepInfo,
)


def observation_size(config: EnvironmentConfig) -> int:
    return 1 + 2 * config.n_rays


def _wrap_angle(angle: jax.Array) -> jax.Array:
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


class TagEnvironment:
    def __init__(self, config: EnvironmentConfig) -> None:
        config.validate()
        self.config = config

        xml = generate_mjcf(config, mode="training")
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mjx_model = mjx.put_model(self.mj_model)

        self.joint_qpos_slices = JointQposSlices(self.mj_model)
        self.joint_dof_slices = JointDofSlices(self.mj_model)

        ray_angles = jnp.linspace(0, 2 * jnp.pi, config.n_rays, endpoint=False)
        self.ray_directions = jnp.stack(
            [jnp.cos(ray_angles), jnp.sin(ray_angles)], axis=-1
        )
        self.substeps_per_action = round(
            1.0 / (config.action_frequency * self.mj_model.opt.timestep)
        )
        self.max_steps = config.episode_max_length * config.action_frequency
        self.freeze_steps = config.chaser_freeze_seconds * config.action_frequency
        self.tag_distance = 2 * config.agent_radius * config.tag_distance_factor
        self.obs_size = observation_size(config)
        self.max_action_buffer_len = (
            config.pipeline_randomization.max_action_delay_substeps + 1
        )
        self.max_observation_buffer_len = (
            config.pipeline_randomization.max_observation_delay_substeps + 1
        )

        self._template_mjx_data = mjx.make_data(self.mj_model)
        self.model_indices = self._build_model_indices()

    def _build_agent_model_indices(self, prefix: str) -> AgentModelIndices:
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, prefix)
        left_wheel_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_left_wheel_geom"
        )
        right_wheel_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_right_wheel_geom"
        )
        caster_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_caster_ball_geom"
        )
        left_joint_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_left_wheel_joint"
        )
        right_joint_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_right_wheel_joint"
        )
        left_actuator_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}_left_motor"
        )
        right_actuator_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}_right_motor"
        )
        return AgentModelIndices(
            body_id=body_id,
            left_wheel_geom_id=left_wheel_geom_id,
            right_wheel_geom_id=right_wheel_geom_id,
            caster_geom_id=caster_geom_id,
            left_wheel_dof_id=int(self.mj_model.jnt_dofadr[left_joint_id]),
            right_wheel_dof_id=int(self.mj_model.jnt_dofadr[right_joint_id]),
            left_actuator_id=left_actuator_id,
            right_actuator_id=right_actuator_id,
        )

    def _build_model_indices(self) -> ModelIndices:
        floor_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        return ModelIndices(
            floor_geom_id=floor_geom_id,
            chaser=self._build_agent_model_indices("chaser"),
            evader=self._build_agent_model_indices("evader"),
        )

    def forward(self, mjx_model: mjx.Model, mjx_data: mjx.Data) -> mjx.Data:
        return mjx.forward(mjx_model, mjx_data)

    def _agent_kinematics(self, mjx_data: mjx.Data, is_chaser: bool) -> AgentKinematics:
        qpos = mjx_data.qpos
        qvel = mjx_data.qvel

        if is_chaser:
            root_qpos = qpos[self.joint_qpos_slices.chaser_root]
            root_qvel = qvel[self.joint_dof_slices.chaser_root]
        else:
            root_qpos = qpos[self.joint_qpos_slices.evader_root]
            root_qvel = qvel[self.joint_dof_slices.evader_root]

        yaw = quaternion_to_yaw(root_qpos[3:7])
        forward_velocity = (
            root_qvel[0] * jnp.cos(yaw) + root_qvel[1] * jnp.sin(yaw)
        ) / self.config.agent_max_linear_velocity
        angular_velocity = root_qvel[5] / self.config.agent_max_angular_velocity

        return AgentKinematics(
            position_xy=root_qpos[:2],
            yaw=yaw,
            forward_velocity=forward_velocity,
            angular_velocity=angular_velocity,
        )

    def _agent_pair_kinematics(
        self, mjx_data: mjx.Data
    ) -> tuple[AgentKinematics, AgentKinematics]:
        return (
            self._agent_kinematics(mjx_data, is_chaser=True),
            self._agent_kinematics(mjx_data, is_chaser=False),
        )

    def _compute_observation_from_measurement(
        self,
        measured_self_position_xy: jax.Array,
        measured_self_yaw: jax.Array,
        measured_other_position_xy: jax.Array,
        measured_obstacle_positions_xy: jax.Array,
        measured_obstacle_yaws: jax.Array,
        obstacle_active: jax.Array,
        step_count: jax.Array,
    ) -> jax.Array:
        ray_distances, ray_types = raycast_scene(
            measured_self_position_xy,
            self.ray_directions,
            measured_self_yaw,
            measured_other_position_xy,
            self.config.agent_radius,
            measured_obstacle_positions_xy,
            measured_obstacle_yaws,
            obstacle_active,
            self.config,
        )
        episode_progress = step_count.astype(jnp.float32) / self.max_steps
        return jnp.concatenate(
            [ray_distances, ray_types, jnp.array([episode_progress])]
        )

    def _compute_observation_pair_from_measurement(
        self,
        measured_agent_positions_xy: jax.Array,
        measured_agent_yaws: jax.Array,
        measured_obstacle_positions_xy: jax.Array,
        measured_obstacle_yaws: jax.Array,
        obstacle_active: jax.Array,
        step_count: jax.Array,
    ) -> jax.Array:
        chaser_obs = self._compute_observation_from_measurement(
            measured_agent_positions_xy[0],
            measured_agent_yaws[0],
            measured_agent_positions_xy[1],
            measured_obstacle_positions_xy,
            measured_obstacle_yaws,
            obstacle_active,
            step_count,
        )
        evader_obs = self._compute_observation_from_measurement(
            measured_agent_positions_xy[1],
            measured_agent_yaws[1],
            measured_agent_positions_xy[0],
            measured_obstacle_positions_xy,
            measured_obstacle_yaws,
            obstacle_active,
            step_count,
        )
        return jnp.stack([chaser_obs, evader_obs], axis=0)

    def _empty_action_buffer(self) -> jax.Array:
        return jnp.zeros((self.max_action_buffer_len, 2, 2), dtype=jnp.float32)

    def _filled_observation_buffer(self, observations: jax.Array) -> jax.Array:
        return jnp.repeat(
            observations[None, :, :], self.max_observation_buffer_len, axis=0
        )

    def _push_buffer(self, buffer: jax.Array, value: jax.Array) -> jax.Array:
        return jnp.roll(buffer, shift=-1, axis=0).at[-1].set(value)

    def _select_delayed(
        self, buffer: jax.Array, delay_substeps: jax.Array
    ) -> jax.Array:
        return buffer[-1 - delay_substeps]

    def _expand_actions_to_substeps(self, proposed_actions: jax.Array) -> jax.Array:
        return jnp.repeat(
            proposed_actions[None, :, :], self.substeps_per_action, axis=0
        )

    def _prepare_action_substep_sequence(
        self, state: TagEnvironmentState, proposed_actions: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        expanded_actions = self._expand_actions_to_substeps(proposed_actions)
        fallback_actions = jnp.where(
            self.config.pipeline_randomization.hold_last_action_on_drop,
            state.last_applied_actions,
            jnp.zeros_like(proposed_actions),
        )
        expanded_fallback_actions = self._expand_actions_to_substeps(fallback_actions)
        return expanded_actions, expanded_fallback_actions

    def _sample_measurement(
        self,
        state: TagEnvironmentState,
        true_agent_positions_xy: jax.Array,
        true_agent_yaws: jax.Array,
        rng: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        pipeline = state.pipeline_params
        (
            agent_pos_rng,
            agent_yaw_rng,
            obstacle_pos_rng,
            obstacle_yaw_rng,
            frame_drop_rng,
            stale_rng,
            stale_length_rng,
        ) = jax.random.split(rng, 7)

        noisy_agent_positions_xy = (
            true_agent_positions_xy
            + jax.random.normal(agent_pos_rng, true_agent_positions_xy.shape)
            * pipeline.position_noise_std
        )
        noisy_agent_yaws = _wrap_angle(
            true_agent_yaws
            + jax.random.normal(agent_yaw_rng, true_agent_yaws.shape)
            * pipeline.yaw_noise_std
        )
        noisy_obstacle_positions_xy = (
            state.obstacle_positions_xy
            + jax.random.normal(obstacle_pos_rng, state.obstacle_positions_xy.shape)
            * pipeline.position_noise_std
        )
        noisy_obstacle_yaws = _wrap_angle(
            state.obstacle_yaws
            + jax.random.normal(obstacle_yaw_rng, state.obstacle_yaws.shape)
            * pipeline.yaw_noise_std
        )

        frame_drop = jax.random.bernoulli(
            frame_drop_rng, pipeline.frame_drop_probability
        )
        start_stale = jax.random.bernoulli(
            stale_rng, pipeline.stale_observation_probability
        )
        sampled_stale_steps = jax.lax.cond(
            pipeline.max_stale_steps > 0,
            lambda _: jax.random.randint(
                stale_length_rng, (), 1, pipeline.max_stale_steps + 1
            ),
            lambda _: jnp.int32(0),
            operand=None,
        )
        currently_stale = state.stale_observation_steps_remaining > 0
        hold_measurement = currently_stale | frame_drop | start_stale

        measured_agent_positions_xy = jnp.where(
            hold_measurement,
            state.last_measured_agent_positions_xy,
            noisy_agent_positions_xy,
        )
        measured_agent_yaws = jnp.where(
            hold_measurement,
            state.last_measured_agent_yaws,
            noisy_agent_yaws,
        )
        measured_obstacle_positions_xy = jnp.where(
            hold_measurement,
            state.last_measured_obstacle_positions_xy,
            noisy_obstacle_positions_xy,
        )
        measured_obstacle_yaws = jnp.where(
            hold_measurement,
            state.last_measured_obstacle_yaws,
            noisy_obstacle_yaws,
        )

        next_stale_steps_remaining = jnp.where(
            currently_stale,
            state.stale_observation_steps_remaining - 1,
            jnp.where(start_stale, sampled_stale_steps - 1, jnp.int32(0)),
        )

        return (
            measured_agent_positions_xy,
            measured_agent_yaws,
            measured_obstacle_positions_xy,
            measured_obstacle_yaws,
            jnp.int32(next_stale_steps_remaining),
        )

    def _collision_flags(
        self,
        chaser_position_xy: jax.Array,
        evader_position_xy: jax.Array,
        obstacle_positions_xy: jax.Array,
        obstacle_yaws: jax.Array,
        obstacle_active: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        obstacle_half_extent = self.config.obstacle_width / 2

        chaser_collision = point_out_of_bounds(
            chaser_position_xy, self.config, self.config.agent_radius
        ) | circle_hits_any_obstacle(
            chaser_position_xy,
            self.config.agent_radius,
            obstacle_positions_xy,
            obstacle_yaws,
            obstacle_active,
            obstacle_half_extent,
        )
        evader_collision = point_out_of_bounds(
            evader_position_xy, self.config, self.config.agent_radius
        ) | circle_hits_any_obstacle(
            evader_position_xy,
            self.config.agent_radius,
            obstacle_positions_xy,
            obstacle_yaws,
            obstacle_active,
            obstacle_half_extent,
        )
        return chaser_collision, evader_collision

    def compute_observations(
        self, state: TagEnvironmentState
    ) -> tuple[jax.Array, jax.Array]:
        delayed_observations = self._select_delayed(
            state.observation_buffer, state.pipeline_params.observation_delay_substeps
        )
        return delayed_observations[0], delayed_observations[1]

    def _make_reset_state(
        self, rng: jax.Array, curriculum_progress: jax.Array
    ) -> tuple[TagEnvironmentState, jax.Array, jax.Array]:
        layout_rng, domain_rng, pipeline_rng, spawn_rng, measurement_rng = (
            jax.random.split(rng, 5)
        )
        obstacle_state = sample_layout_state(
            layout_rng, self.config, curriculum_progress
        )
        domain_params = sample_domain_params(
            domain_rng, self.config, curriculum_progress
        )
        pipeline_params = sample_pipeline_params(
            pipeline_rng, self.config, curriculum_progress
        )
        qpos, qvel, initial_distance = sample_spawn_state(
            spawn_rng,
            self.config,
            obstacle_state,
            self.mj_model.nq,
            self.mj_model.nv,
            self.joint_qpos_slices.chaser_root.start,
            self.joint_qpos_slices.evader_root.start,
            self.joint_qpos_slices.chaser_caster_ball_joint,
            self.joint_qpos_slices.evader_caster_ball_joint,
        )
        randomized_model = randomize_model(
            self.mjx_model, domain_params, self.model_indices
        )
        mjx_data = self.forward(
            randomized_model, self._template_mjx_data.replace(qpos=qpos, qvel=qvel)
        )
        chaser, evader = self._agent_pair_kinematics(mjx_data)
        true_agent_positions_xy = jnp.stack(
            [chaser.position_xy, evader.position_xy], axis=0
        )
        true_agent_yaws = jnp.stack([chaser.yaw, evader.yaw], axis=0)

        zero_state = TagEnvironmentState(
            mjx_data=mjx_data,
            domain_params=domain_params,
            pipeline_params=pipeline_params,
            obstacle_positions_xy=obstacle_state.positions_xy,
            obstacle_yaws=obstacle_state.yaws,
            obstacle_active=obstacle_state.active,
            observation_buffer=jnp.zeros(
                (self.max_observation_buffer_len, 2, self.obs_size), dtype=jnp.float32
            ),
            action_buffer=self._empty_action_buffer(),
            last_applied_actions=jnp.zeros((2, 2), dtype=jnp.float32),
            last_measured_agent_positions_xy=true_agent_positions_xy,
            last_measured_agent_yaws=true_agent_yaws,
            last_measured_obstacle_positions_xy=obstacle_state.positions_xy,
            last_measured_obstacle_yaws=obstacle_state.yaws,
            stale_observation_steps_remaining=jnp.int32(0),
            step_count=jnp.int32(0),
            prev_distance=initial_distance,
        )
        (
            measured_agent_positions_xy,
            measured_agent_yaws,
            measured_obstacle_positions_xy,
            measured_obstacle_yaws,
            stale_steps_remaining,
        ) = self._sample_measurement(
            zero_state, true_agent_positions_xy, true_agent_yaws, measurement_rng
        )
        observations = self._compute_observation_pair_from_measurement(
            measured_agent_positions_xy,
            measured_agent_yaws,
            measured_obstacle_positions_xy,
            measured_obstacle_yaws,
            obstacle_state.active,
            jnp.int32(0),
        )
        state = zero_state._replace(
            observation_buffer=self._filled_observation_buffer(observations),
            last_measured_agent_positions_xy=measured_agent_positions_xy,
            last_measured_agent_yaws=measured_agent_yaws,
            last_measured_obstacle_positions_xy=measured_obstacle_positions_xy,
            last_measured_obstacle_yaws=measured_obstacle_yaws,
            stale_observation_steps_remaining=stale_steps_remaining,
        )
        return state, observations[0], observations[1]

    def step_physics(
        self,
        state: TagEnvironmentState,
        chaser_action: jax.Array,
        evader_action: jax.Array,
        rng: jax.Array,
    ) -> tuple[
        TagEnvironmentState,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        TagEnvironmentStepInfo,
    ]:
        config = self.config

        chaser_action = jnp.clip(chaser_action, -1.0, 1.0)
        evader_action = jnp.clip(evader_action, -1.0, 1.0)
        proposed_actions = jnp.stack([chaser_action, evader_action], axis=0)

        is_frozen = state.step_count < self.freeze_steps
        proposed_actions = proposed_actions.at[0].set(
            jnp.where(is_frozen, jnp.zeros(2), proposed_actions[0])
        )
        action_rng, measurement_rng = jax.random.split(rng)
        proposed_action_substeps, fallback_action_substeps = (
            self._prepare_action_substep_sequence(state, proposed_actions)
        )
        action_drop = jax.random.bernoulli(
            action_rng, state.pipeline_params.action_drop_probability
        )

        randomized_model = randomize_model(
            self.mjx_model, state.domain_params, self.model_indices
        )
        measurement_substep_rngs = jax.random.split(
            measurement_rng, self.substeps_per_action
        )

        def substep(
            carry: tuple[
                mjx.Data,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
            inputs: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[
            tuple[
                mjx.Data,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
            None,
        ]:
            (
                data,
                action_buffer,
                observation_buffer,
                last_applied_actions,
                last_measured_agent_positions_xy,
                last_measured_agent_yaws,
                last_measured_obstacle_positions_xy,
                last_measured_obstacle_yaws,
                stale_steps_remaining,
            ) = carry
            proposed_action_substep, fallback_action_substep, substep_rng = inputs

            action_buffer = self._push_buffer(action_buffer, proposed_action_substep)
            delayed_actions = self._select_delayed(
                action_buffer, state.pipeline_params.action_delay_substeps
            )
            applied_actions = jnp.where(
                state.pipeline_params.action_drop_probability > 0.0,
                jnp.where(
                    action_drop,
                    fallback_action_substep,
                    delayed_actions,
                ),
                delayed_actions,
            )
            control_actions = jnp.concatenate([applied_actions[0], applied_actions[1]])
            data = data.replace(ctrl=control_actions)
            data = mjx.step(randomized_model, data)

            chaser, evader = self._agent_pair_kinematics(data)
            true_agent_positions_xy = jnp.stack(
                [chaser.position_xy, evader.position_xy], axis=0
            )
            true_agent_yaws = jnp.stack([chaser.yaw, evader.yaw], axis=0)
            measurement_state = state._replace(
                mjx_data=data,
                action_buffer=action_buffer,
                observation_buffer=observation_buffer,
                last_applied_actions=last_applied_actions,
                last_measured_agent_positions_xy=last_measured_agent_positions_xy,
                last_measured_agent_yaws=last_measured_agent_yaws,
                last_measured_obstacle_positions_xy=last_measured_obstacle_positions_xy,
                last_measured_obstacle_yaws=last_measured_obstacle_yaws,
                stale_observation_steps_remaining=stale_steps_remaining,
            )
            (
                measured_agent_positions_xy,
                measured_agent_yaws,
                measured_obstacle_positions_xy,
                measured_obstacle_yaws,
                next_stale_steps_remaining,
            ) = self._sample_measurement(
                measurement_state, true_agent_positions_xy, true_agent_yaws, substep_rng
            )
            current_observations = self._compute_observation_pair_from_measurement(
                measured_agent_positions_xy,
                measured_agent_yaws,
                measured_obstacle_positions_xy,
                measured_obstacle_yaws,
                state.obstacle_active,
                state.step_count + 1,
            )
            observation_buffer = self._push_buffer(
                observation_buffer, current_observations
            )
            return (
                data,
                action_buffer,
                observation_buffer,
                applied_actions,
                measured_agent_positions_xy,
                measured_agent_yaws,
                measured_obstacle_positions_xy,
                measured_obstacle_yaws,
                next_stale_steps_remaining,
            ), None

        (
            (
                mjx_data,
                action_buffer,
                observation_buffer,
                applied_actions,
                measured_agent_positions_xy,
                measured_agent_yaws,
                measured_obstacle_positions_xy,
                measured_obstacle_yaws,
                stale_steps_remaining,
            ),
            _,
        ) = jax.lax.scan(
            substep,
            (
                state.mjx_data,
                state.action_buffer,
                state.observation_buffer,
                state.last_applied_actions,
                state.last_measured_agent_positions_xy,
                state.last_measured_agent_yaws,
                state.last_measured_obstacle_positions_xy,
                state.last_measured_obstacle_yaws,
                state.stale_observation_steps_remaining,
            ),
            (
                proposed_action_substeps,
                fallback_action_substeps,
                measurement_substep_rngs,
            ),
        )

        step_count = state.step_count + 1
        obstacle_positions_xy = state.obstacle_positions_xy
        obstacle_yaws = state.obstacle_yaws
        obstacle_active = state.obstacle_active
        chaser, evader = self._agent_pair_kinematics(mjx_data)
        distance = jnp.linalg.norm(chaser.position_xy - evader.position_xy)
        tagged = distance < self.tag_distance
        chaser_collision, evader_collision = self._collision_flags(
            chaser.position_xy,
            evader.position_xy,
            obstacle_positions_xy,
            obstacle_yaws,
            obstacle_active,
        )
        time_up = step_count >= self.max_steps
        done = tagged | time_up | chaser_collision | evader_collision

        distance_shaping = (
            state.prev_distance - config.distance_shaping_gamma * distance
        )

        chaser_reward = (
            config.win_reward * tagged
            - config.win_reward * time_up
            + config.chaser_time_reward
            + config.distance_shaping_scale * distance_shaping
            - config.collision_penalty * chaser_collision
        )
        evader_reward = (
            -config.win_reward * tagged
            + config.win_reward * time_up
            + config.evader_time_reward
            - config.distance_shaping_scale * distance_shaping
            - config.collision_penalty * evader_collision
        )

        interim_state = state._replace(
            mjx_data=mjx_data,
            action_buffer=action_buffer,
            last_applied_actions=applied_actions,
            step_count=step_count,
            prev_distance=distance,
        )
        delayed_observations = self._select_delayed(
            observation_buffer, state.pipeline_params.observation_delay_substeps
        )

        new_state = interim_state._replace(
            observation_buffer=observation_buffer,
            last_measured_agent_positions_xy=measured_agent_positions_xy,
            last_measured_agent_yaws=measured_agent_yaws,
            last_measured_obstacle_positions_xy=measured_obstacle_positions_xy,
            last_measured_obstacle_yaws=measured_obstacle_yaws,
            stale_observation_steps_remaining=stale_steps_remaining,
        )
        info = TagEnvironmentStepInfo(
            tagged=tagged,
            time_up=time_up,
            distance=distance,
            step_count=step_count,
            obstacle_count=obstacle_active.astype(jnp.int32).sum(),
            chaser_collision=chaser_collision,
            evader_collision=evader_collision,
            action_delay_substeps=state.pipeline_params.action_delay_substeps,
            observation_delay_substeps=state.pipeline_params.observation_delay_substeps,
            action_drop_probability=state.pipeline_params.action_drop_probability,
            frame_drop_probability=state.pipeline_params.frame_drop_probability,
            stale_observation_probability=state.pipeline_params.stale_observation_probability,
            position_noise_std=state.pipeline_params.position_noise_std,
            yaw_noise_std=state.pipeline_params.yaw_noise_std,
            floor_friction_scale=state.domain_params.floor_friction_scale,
            chaser_mass_scale=state.domain_params.chaser.mass_scale,
            evader_mass_scale=state.domain_params.evader.mass_scale,
            chaser_motor_balance=state.domain_params.chaser.motor_balance,
            evader_motor_balance=state.domain_params.evader.motor_balance,
            chaser_motor_strength_scale=state.domain_params.chaser.motor_strength_scale,
            evader_motor_strength_scale=state.domain_params.evader.motor_strength_scale,
        )
        return (
            new_state,
            delayed_observations[0],
            delayed_observations[1],
            chaser_reward,
            evader_reward,
            done,
            info,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, rng: jax.Array, curriculum_progress: jax.Array
    ) -> tuple[TagEnvironmentState, jax.Array, jax.Array]:
        return self._make_reset_state(rng, curriculum_progress)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: TagEnvironmentState,
        chaser_action: jax.Array,
        evader_action: jax.Array,
        rng: jax.Array,
    ) -> tuple[
        TagEnvironmentState,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        TagEnvironmentStepInfo,
    ]:
        return self.step_physics(state, chaser_action, evader_action, rng)
