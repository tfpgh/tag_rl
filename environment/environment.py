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
from environment.reset import sample_reset_state
from environment.types import (
    AgentKinematics,
    TagEnvironmentState,
    TagEnvironmentStepInfo,
)


def observation_size(config: EnvironmentConfig) -> int:
    return 3 + 2 * config.n_rays


class TagEnvironment:
    def __init__(self, config: EnvironmentConfig, n_envs: int) -> None:
        self.config = config
        self.n_envs = n_envs

        xml = generate_mjcf(config, mode="training")
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mjx_model = mjx.put_model(self.mj_model)

        self.joint_qpos_slices = JointQposSlices(self.mj_model)
        self.joint_dof_slices = JointDofSlices(self.mj_model)

        self.ray_angles = jnp.linspace(0, 2 * jnp.pi, config.n_rays, endpoint=False)
        self.substeps_per_action = round(
            1.0 / (config.action_frequency * self.mj_model.opt.timestep)
        )
        self.max_steps = config.episode_max_length * config.action_frequency
        self.freeze_steps = config.chaser_freeze_seconds * config.action_frequency
        self.tag_distance = 2 * config.agent_radius * config.tag_distance_factor

        self._template_mjx_data = mjx.make_data(self.mj_model)

    def forward(self, mjx_data: mjx.Data) -> mjx.Data:
        return mjx.forward(self.mjx_model, mjx_data)

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

    def _compute_observation(
        self,
        state: TagEnvironmentState,
        is_chaser: bool,
    ) -> jax.Array:
        me = self._agent_kinematics(state.mjx_data, is_chaser=is_chaser)
        other = self._agent_kinematics(state.mjx_data, is_chaser=not is_chaser)
        ray_distances, ray_types = raycast_scene(
            me.position_xy,
            self.ray_angles,
            me.yaw,
            other.position_xy,
            self.config.agent_radius,
            state.obstacle_positions_xy,
            state.obstacle_yaws,
            state.obstacle_active,
            self.config,
        )
        episode_progress = state.step_count.astype(jnp.float32) / self.max_steps
        return jnp.concatenate(
            [
                jnp.array([me.forward_velocity, me.angular_velocity]),
                ray_distances,
                ray_types,
                jnp.array([episode_progress]),
            ]
        )

    def _collision_flags(
        self, state: TagEnvironmentState
    ) -> tuple[jax.Array, jax.Array]:
        obstacle_half_extent = self.config.obstacle_width / 2
        chaser = self._agent_kinematics(state.mjx_data, is_chaser=True)
        evader = self._agent_kinematics(state.mjx_data, is_chaser=False)

        chaser_collision = point_out_of_bounds(
            chaser.position_xy, self.config, self.config.agent_radius
        ) | circle_hits_any_obstacle(
            chaser.position_xy,
            self.config.agent_radius,
            state.obstacle_positions_xy,
            state.obstacle_yaws,
            state.obstacle_active,
            obstacle_half_extent,
        )
        evader_collision = point_out_of_bounds(
            evader.position_xy, self.config, self.config.agent_radius
        ) | circle_hits_any_obstacle(
            evader.position_xy,
            self.config.agent_radius,
            state.obstacle_positions_xy,
            state.obstacle_yaws,
            state.obstacle_active,
            obstacle_half_extent,
        )
        return chaser_collision, evader_collision

    def reset_state(
        self, rng: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        qpos, qvel, initial_distance, obstacle_state = sample_reset_state(
            rng,
            self.config,
            self.mj_model.nq,
            self.mj_model.nv,
            self.joint_qpos_slices.chaser_root.start,
            self.joint_qpos_slices.evader_root.start,
            self.joint_qpos_slices.chaser_caster_ball_joint,
            self.joint_qpos_slices.evader_caster_ball_joint,
        )
        return (
            qpos,
            qvel,
            initial_distance,
            obstacle_state.positions_xy,
            obstacle_state.yaws,
            obstacle_state.active,
        )

    def compute_observations(
        self, state: TagEnvironmentState
    ) -> tuple[jax.Array, jax.Array]:
        chaser_obs = self._compute_observation(state, is_chaser=True)
        evader_obs = self._compute_observation(state, is_chaser=False)
        return chaser_obs, evader_obs

    def step_physics(
        self,
        state: TagEnvironmentState,
        chaser_action: jax.Array,
        evader_action: jax.Array,
    ) -> tuple[
        TagEnvironmentState, jax.Array, jax.Array, jax.Array, TagEnvironmentStepInfo
    ]:
        config = self.config

        chaser_action = jnp.clip(chaser_action, -1.0, 1.0)
        evader_action = jnp.clip(evader_action, -1.0, 1.0)

        is_frozen = state.step_count < self.freeze_steps
        chaser_action = jnp.where(is_frozen, jnp.zeros(2), chaser_action)

        control_actions = jnp.concatenate([chaser_action, evader_action])
        mjx_data = state.mjx_data.replace(ctrl=control_actions)

        def substep(data: mjx.Data, _: None) -> tuple[mjx.Data, None]:
            return mjx.step(self.mjx_model, data), None

        mjx_data, _ = jax.lax.scan(
            substep, mjx_data, None, length=self.substeps_per_action
        )

        step_count = state.step_count + 1
        new_state = TagEnvironmentState(
            mjx_data=mjx_data,
            obstacle_positions_xy=state.obstacle_positions_xy,
            obstacle_yaws=state.obstacle_yaws,
            obstacle_active=state.obstacle_active,
            step_count=step_count,
            tagged=state.tagged,
            prev_distance=state.prev_distance,
        )

        chaser = self._agent_kinematics(new_state.mjx_data, is_chaser=True)
        evader = self._agent_kinematics(new_state.mjx_data, is_chaser=False)
        distance = jnp.linalg.norm(chaser.position_xy - evader.position_xy)
        tagged = distance < self.tag_distance
        chaser_collision, evader_collision = self._collision_flags(new_state)
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

        new_state = TagEnvironmentState(
            mjx_data=mjx_data,
            obstacle_positions_xy=state.obstacle_positions_xy,
            obstacle_yaws=state.obstacle_yaws,
            obstacle_active=state.obstacle_active,
            step_count=step_count,
            tagged=tagged,
            prev_distance=distance,
        )
        info = TagEnvironmentStepInfo(
            tagged=tagged,
            time_up=time_up,
            distance=distance,
            step_count=step_count,
            is_frozen=is_frozen,
            chaser_collision=chaser_collision,
            evader_collision=evader_collision,
        )
        return new_state, chaser_reward, evader_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> tuple[TagEnvironmentState, jax.Array, jax.Array]:
        (
            qpos,
            qvel,
            initial_distance,
            obstacle_positions_xy,
            obstacle_yaws,
            obstacle_active,
        ) = self.reset_state(rng)
        mjx_data = self.forward(self._template_mjx_data.replace(qpos=qpos, qvel=qvel))
        state = TagEnvironmentState(
            mjx_data=mjx_data,
            obstacle_positions_xy=obstacle_positions_xy,
            obstacle_yaws=obstacle_yaws,
            obstacle_active=obstacle_active,
            step_count=jnp.int32(0),
            tagged=jnp.bool_(False),
            prev_distance=initial_distance,
        )
        chaser_obs, evader_obs = self.compute_observations(state)
        return state, chaser_obs, evader_obs

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: TagEnvironmentState,
        chaser_action: jax.Array,
        evader_action: jax.Array,
    ) -> tuple[
        TagEnvironmentState,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        TagEnvironmentStepInfo,
    ]:
        new_state, chaser_reward, evader_reward, done, info = self.step_physics(
            state, chaser_action, evader_action
        )
        chaser_obs, evader_obs = self.compute_observations(new_state)
        return (
            new_state,
            chaser_obs,
            evader_obs,
            chaser_reward,
            evader_reward,
            done,
            info,
        )
