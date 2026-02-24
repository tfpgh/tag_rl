if __name__ == "__main__":
    import os

    os.environ["MUJOCO_GL"] = "egl"


from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from jax import random
from jax._src.pjit import JitWrapped
from mujoco import mjx
from tqdm import tqdm

from environment.config import EnvironmentConfig
from environment.mjcf import generate_mjcf
from environment.mujoco_data import (
    JointQposSlices,
    SensorSlices,
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
    """
    return 2 + 2 * config.n_rays


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

        self.sensor_slices = SensorSlices(self.mj_model)
        self.joint_qpos_slices = JointQposSlices(self.mj_model)

        # Geom groups for ray hits
        self.geom_groups = jnp.array(self.mj_model.geom_group, dtype=jnp.int32)

        self.ray_angles = jnp.linspace(0, 2 * jnp.pi, config.n_rays, endpoint=False)

        # Derived step timings
        self.substeps_per_action = round(
            1.0 / (config.action_frequency * self.mj_model.opt.timestep)
        )
        self.max_steps = config.episode_max_length * config.action_frequency
        self.freeze_steps = config.chaser_freeze_seconds * config.action_frequency

        # Derived geometry
        self.tag_distance = 2 * config.agent_radius * config.tag_distance_factor
        self.ray_origin_offset = config.agent_radius + 0.001
        self.arena_diagonal = jnp.sqrt(
            (config.arena_width) ** 2 + (config.arena_height) ** 2
        )

    def _cast_rays(
        self,
        mjx_model: mjx.Model,
        mjx_data: mjx.Data,
        agent_position: jax.Array,
        agent_yaw: jax.Array,
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

        origins = agent_position + directions * (self.ray_origin_offset)

        def _cast_single_ray(
            origin: jax.Array, direction: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            distance, geom_id = mjx.ray(
                mjx_model,
                mjx_data,
                origin,
                direction,
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
    ):
        """
        Build observation vector for one agent
        """
        config = self.config
        sensor_data = mjx_data.sensordata
        sensor_slices = self.sensor_slices

        if is_chaser:
            my_position = sensor_data[sensor_slices.chaser_position]
            my_quaternion = sensor_data[sensor_slices.chaser_quaternion]
            my_velocity = sensor_data[sensor_slices.chaser_velocity]
            my_angular_velocity = sensor_data[sensor_slices.chaser_angular_velocity]
        else:
            my_position = sensor_data[sensor_slices.evader_position]
            my_quaternion = sensor_data[sensor_slices.evader_quaternion]
            my_velocity = sensor_data[sensor_slices.evader_velocity]
            my_angular_velocity = sensor_data[sensor_slices.evader_angular_velocity]

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
        )

        return jnp.concatenate(
            [
                jnp.array([normalized_velocity, normalized_angular_velocity]),
                ray_distances,
                ray_types,
            ]
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> tuple[TagEnvironmentState, jax.Array, jax.Array]:
        """Reset environment. Returns (state, chaser_obs, evader_obs)."""
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

        mjx_data = mjx.make_data(
            self.mj_model,
            impl="warp",
            naconmax=20 * self.n_envs,
            njmax=100,
        )
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = mjx.forward(self.mjx_model, mjx_data)

        initial_distance = jnp.linalg.norm(chaser_xy - evader_xy)

        state = TagEnvironmentState(
            mjx_data=mjx_data,
            step_count=jnp.int32(0),
            tagged=jnp.bool_(False),
            prev_distance=initial_distance,
        )

        chaser_obs = self._compute_observation(self.mjx_model, mjx_data, is_chaser=True)
        evader_obs = self._compute_observation(
            self.mjx_model, mjx_data, is_chaser=False
        )

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

        Returns (state, chaser_obs, evader_obs, chaser_reward, evader_reward, done, info)"""
        config = self.config

        chaser_action = jnp.clip(chaser_action, -1.0, 1.0)
        evader_action = jnp.clip(evader_action, -1.0, 1.0)

        # Freeze chaser during freeze period
        is_frozen = state.step_count < self.freeze_steps
        chaser_action = jnp.where(is_frozen, jnp.zeros(2), chaser_action)

        control_actions = jnp.concatenate([chaser_action, evader_action])

        # Substep physics
        mjx_data = state.mjx_data.replace(ctrl=control_actions)

        # Closure can't access self, didn't realize this was a thing in Python
        sensor_slices = self.sensor_slices

        def substep(data: mjx.Data, _) -> tuple[mjx.Data, None]:
            data = mjx.step(self.mjx_model, data)

            return data, None

        mjx_data, _ = jax.lax.scan(
            substep, mjx_data, None, length=self.substeps_per_action
        )
        mjx_data = mjx.forward(self.mjx_model, mjx_data)

        sensor_data = mjx_data.sensordata

        chaser_xy = sensor_data[sensor_slices.chaser_position][:2]
        evader_xy = sensor_data[sensor_slices.evader_position][:2]

        distance = jnp.linalg.norm(chaser_xy - evader_xy)
        tagged = distance < self.tag_distance

        # Termination
        step_count = state.step_count + 1
        time_up = step_count >= self.max_steps
        done = tagged | time_up

        distance_delta = state.prev_distance - distance  # Positive = getting closer

        chaser_reward = (
            config.win_reward * tagged
            - config.win_reward * time_up
            - config.time_reward
            + config.distance_shaping_scale * distance_delta
        )
        evader_reward = (
            -config.win_reward * tagged
            + config.win_reward * time_up
            + config.time_reward
            - config.distance_shaping_scale * distance_delta
        )

        chaser_reward: jax.Array = jnp.where(is_frozen, 0.0, chaser_reward)

        new_state = TagEnvironmentState(
            mjx_data=mjx_data,
            step_count=step_count,
            tagged=tagged,
            prev_distance=distance,
        )

        chaser_obs = self._compute_observation(self.mjx_model, mjx_data, is_chaser=True)
        evader_obs = self._compute_observation(
            self.mjx_model, mjx_data, is_chaser=False
        )

        info = TagEnvironmentStepInfo(
            tagged=tagged,
            time_up=time_up,
            distance=distance,
            step_count=step_count,
            is_frozen=is_frozen,
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


def make_auto_reset_step(env: TagEnvironment) -> JitWrapped:
    @jax.jit
    def auto_step(
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
        state, chaser_obs, evader_obs, chaser_reward, evader_reward, done, info = (
            env.step(state, chaser_action, evader_action)
        )

        rng, reset_key = random.split(rng)
        reset_state, reset_chaser_obs, reset_evader_obs = env.reset(reset_key)

        state = jax.tree.map(
            lambda fresh, old: jnp.where(done, fresh, old), reset_state, state
        )
        chaser_obs = jnp.where(done, reset_chaser_obs, chaser_obs)
        evader_obs = jnp.where(done, reset_evader_obs, evader_obs)

        return state, chaser_obs, evader_obs, chaser_reward, evader_reward, done, info

    return auto_step


# Testing
if __name__ == "__main__":
    import os
    import time

    os.environ["MUJOCO_GL"] = "egl"

    config = EnvironmentConfig()

    config.arena_width = 0.5
    config.arena_height = 0.5
    config.wall_margin_factor = 1.2

    N = 4096
    env = TagEnvironment(config, N)

    print(f"obs size:     {observation_size(config)}")
    print(
        f"nq={env.mj_model.nq}  nv={env.mj_model.nv}  "
        f"nu={env.mj_model.nu}  ngeom={env.mj_model.ngeom}"
    )
    print(f"substeps:     {env.substeps_per_action}")
    print(f"max_steps:    {env.max_steps}")
    print(f"freeze_steps: {env.freeze_steps}")
    print(f"tag_distance: {env.tag_distance:.4f}")

    rng = random.PRNGKey(42)

    # Single env
    rng, k = random.split(rng)
    state, obs_c, obs_e = env.reset(k)
    print(f"\nobs shapes: chaser={obs_c.shape}  evader={obs_e.shape}")

    rng, k = random.split(rng)
    act = jnp.full((4,), 0.05)
    state, obs_c, obs_e, rew_c, rew_e, done, info = env.step(state, act[:2], act[2:])
    print(
        f"step 1: dist={info.distance:.4f}  tagged={info.tagged}  "
        f"frozen={info.is_frozen}  rew_c={rew_c:.6f}  rew_e={rew_e:.6f}"
    )

    # Vectorized test
    print(f"\nVectorized ({N} envs)...")
    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)

    rng, k = random.split(rng)
    keys = random.split(k, N)

    t0 = time.time()
    states, obs_cs, obs_es = v_reset(keys)
    jax.block_until_ready(obs_cs)
    print(f"  reset (incl JIT): {time.time() - t0:.1f}s")

    rng, k = random.split(rng)
    batch_act = jnp.full((N, 4), 0.05)

    waypoint_interval = 10
    rng, k1, k2 = random.split(rng, 3)
    prev_actions = random.uniform(k1, (N, 4), minval=-0.5, maxval=0.5)
    next_actions = random.uniform(k2, (N, 4), minval=-0.5, maxval=0.5)

    t0 = time.time()
    states, *_ = v_step(states, batch_act[:, :2], batch_act[:, 2:])
    jax.block_until_ready(states.step_count)
    print(f"  first step (incl JIT): {time.time() - t0:.1f}s")

    # Benchmark post-JIT
    n_iters = 100
    t0 = time.time()
    for i in tqdm(range(n_iters)):
        if i % waypoint_interval == 0 and i > 0:
            prev_actions = next_actions
            rng, k = random.split(rng)
            next_actions = random.uniform(k, (N, 4), minval=-0.5, maxval=0.5)
        t = (i % waypoint_interval) / waypoint_interval
        batch_act = prev_actions + t * (next_actions - prev_actions)

        states, obs_cs, obs_es, rew_cs, rew_es, dones, infos = v_step(
            states, batch_act[:, :2], batch_act[:, 2:]
        )
        jax.block_until_ready(states.step_count)

    elapsed = time.time() - t0
    sps = (n_iters * N) / elapsed
    print(
        f"\n  Throughput: {sps:,.0f} env steps/sec ({n_iters} x {N} in {elapsed:.2f}s)"
    )

    import imageio
    import numpy as np

    print("\nRecording video...")
    rng, k = random.split(rng)
    state, chaser_obs, evader_obs = env.reset(k)
    auto_step = make_auto_reset_step(env)

    # Action linear interpolation
    waypoint_interval = 10
    rng, k_c, k_e = random.split(rng, 3)
    prev_chaser = random.uniform(k_c, (2,), minval=-0.5, maxval=0.5)
    prev_evader = random.uniform(k_e, (2,), minval=-0.5, maxval=0.5)
    rng, k_c, k_e = random.split(rng, 3)
    next_chaser = random.uniform(k_c, (2,), minval=-0.5, maxval=0.5)
    next_evader = random.uniform(k_e, (2,), minval=-0.5, maxval=0.5)

    frames = []
    renderer = mujoco.Renderer(env.mj_model, height=1080, width=1920)
    mj_data = mujoco.MjData(env.mj_model)

    n_video_steps = 1_200
    for i in range(n_video_steps):
        # Manually copy qpos/qvel from MJX to CPU MjData
        mj_data.qpos[:] = np.array(state.mjx_data.qpos)
        mj_data.qvel[:] = np.array(state.mjx_data.qvel)
        mujoco.mj_forward(env.mj_model, mj_data)
        renderer.update_scene(mj_data)
        frames.append(renderer.render())

        # Random actions in [-1, 1]
        rng, k_c, k_e, k_reset = random.split(rng, 4)
        if i % waypoint_interval == 0 and i > 0:
            prev_chaser = next_chaser
            prev_evader = next_evader
            rng, k_c, k_e = random.split(rng, 3)
            next_chaser = random.uniform(k_c, (2,), minval=-0.5, maxval=0.5)
            next_evader = random.uniform(k_e, (2,), minval=-0.5, maxval=0.5)
        t = (i % waypoint_interval) / waypoint_interval
        chaser_action = prev_chaser + t * (next_chaser - prev_chaser)
        evader_action = prev_evader + t * (next_evader - prev_evader)

        # Step with auto-reset
        state, chaser_obs, evader_obs, chaser_reward, evader_reward, done, info = (
            auto_step(state, chaser_action, evader_action, k_reset)
        )

        if done:
            print(
                f"  step {i}: tagged={info.tagged}, time_up={info.time_up}, "
                f"dist={info.distance:.3f}, steps={info.step_count} -> auto reset"
            )

    renderer.close()
    imageio.mimsave("debug.mp4", frames, fps=int(config.action_frequency))
    print(f"Saved {len(frames)} frames to debug.mp4")
