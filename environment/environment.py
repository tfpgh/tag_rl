from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from jax import random
from mujoco import mjx

from environment.config import EnvironmentConfig
from environment.mjcf import generate_mjcf
from environment.mujoco_data import (
    JointDofSlices,
    JointQposSlices,
    quaternion_to_yaw,
    yaw_to_quaternion,
)


def observation_size(config: EnvironmentConfig) -> int:
    """
    Observation vector size.

    Layout:
        [0] linear velocity (normalized [-1, 1])
        [1] angular velocity (normalized [-1, 1])
        [2:2+N] ray distances (normalized [0, 1])
        [2+N:2+2N] ray hit type encoding (0 for environment, 1 for agent)
        [2+2N] episode progress (normalized [0, 1])
    """
    return 3 + 2 * config.n_rays


class TagEnvironmentState(NamedTuple):
    mjx_data: mjx.Data

    step_count: jax.Array  # int
    tagged: jax.Array  # bool
    prev_distance: jax.Array  # float


class TagEnvironmentStepInfo(NamedTuple):
    tagged: jax.Array  # bool
    time_up: jax.Array  # bool
    distance: jax.Array  # float
    step_count: jax.Array  # int
    is_frozen: jax.Array  # bool


class TagEnvironment:
    def __init__(self, config: EnvironmentConfig, n_envs: int) -> None:
        self.config = config
        self.n_envs = n_envs

        xml = generate_mjcf(config)
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mjx_model = mjx.put_model(self.mj_model, impl="warp")

        self.joint_qpos_slices = JointQposSlices(self.mj_model)
        self.joint_dof_slices = JointDofSlices(self.mj_model)

        # Geom groups for ray hits
        self.geom_groups = jnp.array(self.mj_model.geom_group, dtype=jnp.int32)

        # Body IDs for ray exclusion (skip own body geoms)
        chaser_root = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "chaser")
        evader_root = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "evader")
        self.chaser_body_ids = self._get_body_subtree(chaser_root)
        self.evader_body_ids = self._get_body_subtree(evader_root)

        self.ray_angles = jnp.linspace(0, 2 * jnp.pi, config.n_rays, endpoint=False)

        # Derived step timings
        self.substeps_per_action = round(
            1.0 / (config.action_frequency * self.mj_model.opt.timestep)
        )
        self.max_steps = config.episode_max_length * config.action_frequency
        self.freeze_steps = config.chaser_freeze_seconds * config.action_frequency

        # Derived geometry
        self.tag_distance = 2 * config.agent_radius * config.tag_distance_factor
        self.arena_diagonal = jnp.sqrt(
            (config.arena_width) ** 2 + (config.arena_height) ** 2
        )

        # Pre-allocate mjx.Data template (reused by reset to avoid tracer leaks)
        self._template_mjx_data = mjx.make_data(
            self.mj_model,
            impl="warp",
            naconmax=20 * n_envs,
            njmax=100,
        )

    def _get_body_subtree(self, root_id: int) -> list[int]:
        """Get all body IDs in the subtree rooted at root_id (inclusive)."""
        subtree = [root_id]
        for i in range(self.mj_model.nbody):
            if i != root_id:
                current = i
                while current != 0:
                    current = int(self.mj_model.body_parentid[current])
                    if current == root_id:
                        subtree.append(i)
                        break
        return sorted(subtree)

    def _cast_rays(
        self,
        mjx_model: mjx.Model,
        mjx_data: mjx.Data,
        agent_position: jax.Array,
        agent_yaw: jax.Array,
        bodyexclude: list[int],
    ) -> tuple[jax.Array, jax.Array]:
        """
        Cast config.n_rays horizontally using mjx.ray

        Returns:
            distances: [n_rays] normalized to [0, 1]
            types: [n_ray] GEOM_GROUP_WALL / GEOM_GROUP_AGENT
        """

        angles = self.ray_angles + agent_yaw
        directions = jnp.stack(
            [
                jnp.cos(angles),
                jnp.sin(angles),
                jnp.zeros_like(angles),
            ],
            axis=-1,
        )

        origins = jnp.broadcast_to(agent_position, directions.shape)

        def _cast_single_ray(
            origin: jax.Array, direction: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            distance, geom_id = mjx.ray(
                mjx_model,
                mjx_data,
                origin,
                direction,
                bodyexclude=bodyexclude,
            )

            geom_group = self.geom_groups[geom_id]
            normalized_distance = jnp.clip(distance / self.arena_diagonal, 0.0, 1.0)
            return normalized_distance, geom_group

        return jax.vmap(_cast_single_ray)(origins, directions)

    def _compute_observation(
        self,
        mjx_model: mjx.Model,
        mjx_data: mjx.Data,
        is_chaser: bool,
        step_count: jax.Array,
    ):
        """
        Build observation vector for one agent.
        Reads position/quaternion from qpos, velocity from qvel.
        """
        config = self.config
        qpos = mjx_data.qpos
        qvel = mjx_data.qvel

        if is_chaser:
            my_position = qpos[self.joint_qpos_slices.chaser_root][:3]
            my_quaternion = qpos[self.joint_qpos_slices.chaser_root][3:7]
            my_velocity = qvel[self.joint_dof_slices.chaser_root][:3]
            my_angular_velocity = qvel[self.joint_dof_slices.chaser_root][3:6]
        else:
            my_position = qpos[self.joint_qpos_slices.evader_root][:3]
            my_quaternion = qpos[self.joint_qpos_slices.evader_root][3:7]
            my_velocity = qvel[self.joint_dof_slices.evader_root][:3]
            my_angular_velocity = qvel[self.joint_dof_slices.evader_root][3:6]

        my_yaw = quaternion_to_yaw(my_quaternion)

        # Don't care about perfect clip
        forward_velocity = my_velocity[0] * jnp.cos(my_yaw) + my_velocity[1] * jnp.sin(
            my_yaw
        )
        normalized_velocity = forward_velocity / config.agent_max_linear_velocity
        normalized_angular_velocity = (
            my_angular_velocity[2] / config.agent_max_angular_velocity
        )

        ray_distances, ray_types = self._cast_rays(
            mjx_model,
            mjx_data,
            my_position,
            my_yaw,
            bodyexclude=self.chaser_body_ids if is_chaser else self.evader_body_ids,
        )

        episode_progress = step_count.astype(jnp.float32) / self.max_steps

        return jnp.concatenate(
            [
                jnp.array([normalized_velocity, normalized_angular_velocity]),
                ray_distances,
                ray_types,
                jnp.array([episode_progress]),
            ]
        )

    def _reset_positions(self, rng: jax.Array):
        """Generate random qpos/qvel for a reset. Returns (qpos, qvel, initial_distance)."""
        config = self.config
        key_chaser_position, key_evader_position, key_chaser_yaw, key_evader_yaw = (
            random.split(rng, 4)
        )

        # Move agents to random positions
        margin = config.wall_margin_factor * config.agent_radius
        lower_bounds = jnp.array(
            [-(config.arena_width / 2) + margin, -(config.arena_height / 2) + margin]
        )
        upper_bounds = jnp.array(
            [(config.arena_width / 2) - margin, (config.arena_height / 2) - margin]
        )

        chaser_xy = random.uniform(
            key_chaser_position, (2,), minval=lower_bounds, maxval=upper_bounds
        )
        evader_xy = random.uniform(
            key_evader_position, (2,), minval=lower_bounds, maxval=upper_bounds
        )

        # Guarantee minimum separation
        delta = evader_xy - chaser_xy
        dist = jnp.linalg.norm(delta)

        # If exactly coincident, pick a random direction
        random_dir = random.uniform(key_evader_yaw, (2,), minval=-1.0, maxval=1.0)
        random_dir = random_dir / jnp.linalg.norm(random_dir)
        direction = jnp.where(dist < 1e-6, random_dir, delta / dist)

        too_close = dist < config.minimum_starting_separation

        # Phase 1: push evader away from chaser
        evader_candidate = chaser_xy + direction * config.minimum_starting_separation
        evader_candidate = jnp.clip(evader_candidate, lower_bounds, upper_bounds)
        evader_xy = jnp.where(too_close, evader_candidate, evader_xy)

        # Phase 2: if clamp pulled evader back, push chaser the other way
        new_dist = jnp.linalg.norm(evader_xy - chaser_xy)
        still_too_close = new_dist < config.minimum_starting_separation
        chaser_candidate = evader_xy - direction * config.minimum_starting_separation
        chaser_candidate = jnp.clip(chaser_candidate, lower_bounds, upper_bounds)
        chaser_xy = jnp.where(still_too_close, chaser_candidate, chaser_xy)

        chaser_yaw = random.uniform(key_chaser_yaw, (), minval=-jnp.pi, maxval=jnp.pi)
        evader_yaw = random.uniform(key_evader_yaw, (), minval=-jnp.pi, maxval=jnp.pi)

        z = config.agent_z

        identity_quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])

        qpos = jnp.zeros(self.mj_model.nq)

        # Chaser freejoint
        chaser_adr = self.joint_qpos_slices.chaser_root.start
        qpos = qpos.at[chaser_adr : chaser_adr + 3].set(
            jnp.array([chaser_xy[0], chaser_xy[1], z])
        )
        qpos = qpos.at[chaser_adr + 3 : chaser_adr + 7].set(
            yaw_to_quaternion(chaser_yaw)
        )
        qpos = qpos.at[self.joint_qpos_slices.chaser_caster_ball_joint].set(
            identity_quaternion
        )

        # Evader freejoint
        evader_adr = self.joint_qpos_slices.evader_root.start
        qpos = qpos.at[evader_adr : evader_adr + 3].set(
            jnp.array([evader_xy[0], evader_xy[1], z])
        )
        qpos = qpos.at[evader_adr + 3 : evader_adr + 7].set(
            yaw_to_quaternion(evader_yaw)
        )
        qpos = qpos.at[self.joint_qpos_slices.evader_caster_ball_joint].set(
            identity_quaternion
        )

        qvel = jnp.zeros(self.mj_model.nv)

        initial_distance = jnp.linalg.norm(chaser_xy - evader_xy)

        return qpos, qvel, initial_distance

    def reset_state(self, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Generate random qpos/qvel only. NO forward, NO obs.

        Returns (qpos, qvel, initial_distance).
        """
        return self._reset_positions(rng)

    def forward(self, mjx_data: mjx.Data) -> mjx.Data:
        """Thin wrapper around mjx.forward."""
        return mjx.forward(self.mjx_model, mjx_data)

    def compute_observations(
        self, mjx_data: mjx.Data, step_count: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Compute observations for both agents from forwarded state."""
        chaser_obs = self._compute_observation(
            self.mjx_model, mjx_data, is_chaser=True, step_count=step_count
        )
        evader_obs = self._compute_observation(
            self.mjx_model, mjx_data, is_chaser=False, step_count=step_count
        )
        return chaser_obs, evader_obs

    def step_physics(
        self,
        state: TagEnvironmentState,
        chaser_action: jax.Array,
        evader_action: jax.Array,
    ) -> tuple[
        TagEnvironmentState, jax.Array, jax.Array, jax.Array, TagEnvironmentStepInfo
    ]:
        """
        Physics + rewards only. NO forward, NO obs.

        Returns (new_state, chaser_reward, evader_reward, done, info).
        The new_state contains un-forwarded mjx_data.
        """
        config = self.config

        chaser_action = jnp.clip(chaser_action, -1.0, 1.0)
        evader_action = jnp.clip(evader_action, -1.0, 1.0)

        # Freeze chaser during freeze period
        is_frozen = state.step_count < self.freeze_steps
        chaser_action = jnp.where(is_frozen, jnp.zeros(2), chaser_action)

        control_actions = jnp.concatenate([chaser_action, evader_action])

        # Substep physics
        mjx_data = state.mjx_data.replace(ctrl=control_actions)

        def substep(data: mjx.Data, _) -> tuple[mjx.Data, None]:
            data = mjx.step(self.mjx_model, data)
            return data, None

        mjx_data, _ = jax.lax.scan(
            substep, mjx_data, None, length=self.substeps_per_action
        )

        # Read positions directly from qpos (no forward needed)
        chaser_xy = mjx_data.qpos[self.joint_qpos_slices.chaser_root][:2]
        evader_xy = mjx_data.qpos[self.joint_qpos_slices.evader_root][:2]

        distance = jnp.linalg.norm(chaser_xy - evader_xy)
        tagged = distance < self.tag_distance

        # Termination
        step_count = state.step_count + 1
        time_up = step_count >= self.max_steps
        done = tagged | time_up

        # Potential-based reward shaping: gamma * phi(s') - phi(s) where phi = -distance
        distance_shaping = (
            state.prev_distance - config.distance_shaping_gamma * distance
        )

        chaser_reward = (
            config.win_reward * tagged
            - config.win_reward * time_up
            - config.time_reward
            + config.distance_shaping_scale * distance_shaping
        )
        evader_reward = (
            -config.win_reward * tagged
            + config.win_reward * time_up
            + config.time_reward
            - config.distance_shaping_scale * distance_shaping
        )

        chaser_reward: jax.Array = jnp.where(is_frozen, 0.0, chaser_reward)  # type: ignore[assignment]

        new_state = TagEnvironmentState(
            mjx_data=mjx_data,
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
        )

        return new_state, chaser_reward, evader_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> tuple[TagEnvironmentState, jax.Array, jax.Array]:
        """Reset environment. Returns (state, chaser_obs, evader_obs).

        Convenience method composing reset_state → forward → compute_observations.
        """
        qpos, qvel, initial_distance = self.reset_state(rng)

        mjx_data = self._template_mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = self.forward(mjx_data)

        state = TagEnvironmentState(
            mjx_data=mjx_data,
            step_count=jnp.int32(0),
            tagged=jnp.bool_(False),
            prev_distance=initial_distance,
        )

        chaser_obs, evader_obs = self.compute_observations(mjx_data, jnp.int32(0))

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
        """
        Step the environment.

        Convenience method composing step_physics → forward → compute_observations.
        Returns (state, chaser_obs, evader_obs, chaser_reward, evader_reward, done, info)
        """
        new_state, chaser_reward, evader_reward, done, info = self.step_physics(
            state, chaser_action, evader_action
        )

        mjx_data = self.forward(new_state.mjx_data)
        new_state = new_state._replace(mjx_data=mjx_data)

        chaser_obs, evader_obs = self.compute_observations(
            mjx_data, new_state.step_count
        )

        return (
            new_state,
            chaser_obs,
            evader_obs,
            chaser_reward,
            evader_reward,
            done,
            info,
        )
