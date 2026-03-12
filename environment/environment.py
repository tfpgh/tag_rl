from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from jax import random
from mujoco import mjx

from environment.config import EnvironmentConfig
from environment.mjcf import GEOM_GROUP_WALL, WALL_HEIGHT, generate_mjcf
from environment.mjx_ray_patch import patch_mjx_cylinder_rays
from environment.mujoco_data import (
    JointDofSlices,
    JointQposSlices,
    quaternion_to_yaw,
    yaw_to_quaternion,
)

patch_mjx_cylinder_rays()


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
    chaser_collision: jax.Array  # bool
    evader_collision: jax.Array  # bool


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
        chaser_root = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "chaser"
        )
        evader_root = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "evader"
        )
        self.chaser_body_ids = self._get_body_subtree(chaser_root)
        self.evader_body_ids = self._get_body_subtree(evader_root)

        # Obstacle mocap body indices (for mocap_pos / mocap_quat)
        self.obstacle_mocap_ids: list[int] = []
        for i in range(config.max_obstacles):
            body_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}"
            )
            self.obstacle_mocap_ids.append(int(self.mj_model.body_mocapid[body_id]))
        self.obstacle_clearance = config.obstacle_width / 2 + config.agent_radius

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
        self.collision_epsilon = config.collision_epsilon_factor * config.agent_radius / float(self.arena_diagonal)

        # Pre-allocate mjx.Data template (reused by reset to avoid tracer leaks)
        self._template_mjx_data = mjx.make_data(
            self.mj_model,
            impl="warp",
            naconmax=(14 + 4 * config.max_obstacles) * n_envs,
            njmax=80,
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

    def collision_from_obs(self, obs: jax.Array) -> jax.Array:
        """Check if any ray hits a wall/obstacle closer than agent_radius (normalized)."""
        n = self.config.n_rays
        ray_distances = obs[2 : 2 + n]
        ray_types = obs[2 + n : 2 + 2 * n]
        return jnp.any(
            (ray_types == GEOM_GROUP_WALL) & (ray_distances < self.collision_epsilon)
        )

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

        # --- Obstacle placement (farthest-point sampling) ---
        n_active = random.randint(key_obs_count, (), 0, config.max_obstacles + 1)

        obs_margin = config.obstacle_width / 2
        obs_lower = jnp.array(
            [
                -(config.arena_width / 2) + obs_margin,
                -(config.arena_height / 2) + obs_margin,
            ]
        )
        obs_upper = jnp.array(
            [
                (config.arena_width / 2) - obs_margin,
                (config.arena_height / 2) - obs_margin,
            ]
        )

        n_pool = 64
        pool = random.uniform(
            key_obs_pos, (n_pool, 2), minval=obs_lower, maxval=obs_upper
        )

        def _farthest_step(carry, _):
            chosen, n_chosen, available = carry
            dists = jnp.linalg.norm(
                pool[:, None, :] - chosen[None, :, :], axis=-1
            )  # (n_pool, max_obs)
            slot_mask = jnp.arange(config.max_obstacles) < n_chosen
            dists = jnp.where(slot_mask[None, :], dists, jnp.inf)
            min_dists = jnp.min(dists, axis=1)  # pyright: ignore[reportArgumentType]
            min_dists = jnp.where(available, min_dists, -jnp.inf)
            idx = jnp.argmax(min_dists)
            chosen = chosen.at[n_chosen].set(pool[idx])
            available = available.at[idx].set(False)
            return (chosen, n_chosen + 1, available), None

        init_chosen = jnp.zeros((config.max_obstacles, 2))
        init_chosen = init_chosen.at[0].set(pool[0])
        init_available = jnp.ones(n_pool, dtype=jnp.bool_).at[0].set(False)

        (obstacle_xy, _, _), _ = jax.lax.scan(
            _farthest_step,
            (init_chosen, jnp.int32(1), init_available),
            None,
            length=config.max_obstacles - 1,
        )

        # Random yaw per obstacle
        obstacle_yaws = random.uniform(
            key_obs_yaw, (config.max_obstacles,), minval=-jnp.pi, maxval=jnp.pi
        )

        active_mask = jnp.arange(config.max_obstacles) < n_active
        obstacle_z_active = WALL_HEIGHT / 2
        obstacle_z = jnp.where(active_mask, obstacle_z_active, -10.0)

        # --- Agent placement (candidate-based, JAX-friendly) ---
        n_candidates = 64
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

        clearance = self.obstacle_clearance

        def _pick_valid(candidates: jax.Array) -> jax.Array:
            # candidates: (n_candidates, 2), obstacle_xy: (max_obstacles, 2)
            # Check overlap with each active obstacle (axis-aligned box + circle approx)
            dx = jnp.abs(
                candidates[:, None, 0] - obstacle_xy[None, :, 0]
            )  # (n_cand, max_obs)
            dy = jnp.abs(candidates[:, None, 1] - obstacle_xy[None, :, 1])
            overlap = (dx < clearance) & (dy < clearance) & active_mask[None, :]
            any_overlap = jnp.any(overlap, axis=1)  # (n_candidates,)
            valid = ~any_overlap
            # Pick first valid; fallback to index 0 if none valid
            idx = jnp.argmax(valid)
            return candidates[idx]

        chaser_xy = _pick_valid(chaser_candidates)
        evader_xy = _pick_valid(evader_candidates)

        # --- Minimum separation (existing push-apart logic) ---
        delta = evader_xy - chaser_xy
        dist = jnp.linalg.norm(delta)

        random_dir = random.uniform(
            key_separation_fallback, (2,), minval=-1.0, maxval=1.0
        )
        random_dir = random_dir / jnp.linalg.norm(random_dir + 1e-8)
        direction = jnp.where(dist < 1e-6, random_dir, delta / dist)

        too_close = dist < config.minimum_starting_separation

        evader_candidate = chaser_xy + direction * config.minimum_starting_separation
        evader_candidate = jnp.clip(evader_candidate, agent_lower, agent_upper)
        evader_xy = jnp.where(too_close, evader_candidate, evader_xy)

        new_dist = jnp.linalg.norm(evader_xy - chaser_xy)
        still_too_close = new_dist < config.minimum_starting_separation
        chaser_candidate = evader_xy - direction * config.minimum_starting_separation
        chaser_candidate = jnp.clip(chaser_candidate, agent_lower, agent_upper)
        chaser_xy = jnp.where(still_too_close, chaser_candidate, chaser_xy)

        chaser_yaw = random.uniform(key_chaser_yaw, (), minval=-jnp.pi, maxval=jnp.pi)
        evader_yaw = random.uniform(key_evader_yaw, (), minval=-jnp.pi, maxval=jnp.pi)

        # --- Build qpos ---
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

        # Obstacle mocap positions: (max_obstacles, 3)
        obstacle_pos = jnp.stack(
            [obstacle_xy[:, 0], obstacle_xy[:, 1], obstacle_z], axis=-1
        )
        # Random yaw quaternions: (max_obstacles, 4)
        obstacle_quat = jax.vmap(yaw_to_quaternion)(obstacle_yaws)

        qvel = jnp.zeros(self.mj_model.nv)

        initial_distance = jnp.linalg.norm(chaser_xy - evader_xy)

        return qpos, qvel, initial_distance, obstacle_pos, obstacle_quat

    def reset_state(
        self, rng: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Generate random qpos/qvel only. NO forward, NO obs.

        Returns (qpos, qvel, initial_distance, obstacle_pos, obstacle_quat).
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
            chaser_collision=jnp.bool_(False),
            evader_collision=jnp.bool_(False),
        )

        return new_state, chaser_reward, evader_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> tuple[TagEnvironmentState, jax.Array, jax.Array]:
        """Reset environment. Returns (state, chaser_obs, evader_obs).

        Convenience method composing reset_state → forward → compute_observations.
        """
        qpos, qvel, initial_distance, obstacle_pos, obstacle_quat = self.reset_state(
            rng
        )

        # Set obstacle positions via mocap arrays
        mocap_pos = self._template_mjx_data.mocap_pos
        mocap_quat = self._template_mjx_data.mocap_quat
        for i in range(self.config.max_obstacles):
            mid = self.obstacle_mocap_ids[i]
            mocap_pos = mocap_pos.at[mid].set(obstacle_pos[i])
            mocap_quat = mocap_quat.at[mid].set(obstacle_quat[i])

        mjx_data = self._template_mjx_data.replace(
            qpos=qpos, qvel=qvel, mocap_pos=mocap_pos, mocap_quat=mocap_quat
        )
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
