from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from environment.config import EnvironmentConfig
from environment.environment import (
    TagEnvironment,
    TagEnvironmentState,
    TagEnvironmentStepInfo,
)
from rl.config import RLConfig


class Transition(NamedTuple):
    done: jax.Array
    chaser_action: jax.Array
    evader_action: jax.Array
    chaser_value: jax.Array
    evader_value: jax.Array
    chaser_reward: jax.Array
    evader_reward: jax.Array
    chaser_log_prob: jax.Array
    evader_log_prob: jax.Array
    chaser_obs: jax.Array
    evader_obs: jax.Array
    info: TagEnvironmentStepInfo


class AgentTransition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array


class RunnerState(NamedTuple):
    chaser_ts: TrainState
    evader_ts: TrainState
    env_state: TagEnvironmentState
    chaser_obs: jax.Array
    evader_obs: jax.Array
    done: jax.Array
    chaser_hstate: jax.Array
    evader_hstate: jax.Array
    rng: jax.Array


class RolloutCarry(NamedTuple):
    chaser_ts: TrainState
    evader_ts: TrainState
    env_state: TagEnvironmentState
    chaser_obs: jax.Array
    evader_obs: jax.Array
    done: jax.Array
    chaser_hstate: jax.Array
    evader_hstate: jax.Array
    rng: jax.Array


ActorCriticModule = Any


def auto_reset_step(
    env: TagEnvironment,
    env_config: EnvironmentConfig,
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
    stepped, chaser_reward, evader_reward, done, info = env.step_physics(
        state, chaser_action, evader_action
    )

    reset_qpos, reset_qvel, reset_distance, reset_obs_pos, reset_obs_quat = (
        env.reset_state(rng)
    )

    new_mocap_pos = jnp.where(done, reset_obs_pos, stepped.mjx_data.mocap_pos)
    new_mocap_quat = jnp.where(done, reset_obs_quat, stepped.mjx_data.mocap_quat)
    new_mjx_data = stepped.mjx_data.replace(
        qpos=jnp.where(done, reset_qpos, stepped.mjx_data.qpos),
        qvel=jnp.where(done, reset_qvel, stepped.mjx_data.qvel),
        time=jnp.where(
            done, jnp.zeros_like(stepped.mjx_data.time), stepped.mjx_data.time
        ),
        mocap_pos=new_mocap_pos,
        mocap_quat=new_mocap_quat,
    )
    step_count = jnp.asarray(jnp.where(done, jnp.int32(0), stepped.step_count))

    new_mjx_data = env.forward(new_mjx_data)
    chaser_obs, evader_obs = env.compute_observations(new_mjx_data, step_count)

    chaser_collision = env.collision_from_obs(chaser_obs)
    evader_collision = env.collision_from_obs(evader_obs)
    chaser_reward = (
        chaser_reward
        - env_config.collision_penalty * chaser_collision * ~info.is_frozen
    )
    evader_reward = evader_reward - env_config.collision_penalty * evader_collision
    info = info._replace(
        chaser_collision=chaser_collision,
        evader_collision=evader_collision,
    )

    tagged = jnp.asarray(jnp.where(done, jnp.bool_(False), stepped.tagged))
    state = TagEnvironmentState(
        mjx_data=new_mjx_data,
        step_count=step_count,
        tagged=tagged,
        prev_distance=jnp.where(done, reset_distance, stepped.prev_distance),
    )
    return state, chaser_obs, evader_obs, chaser_reward, evader_reward, done, info


def collect_trajectories(
    runner_state: RunnerState,
    env: TagEnvironment,
    env_config: EnvironmentConfig,
    rl_config: RLConfig,
    chaser_network: ActorCriticModule,
    evader_network: ActorCriticModule,
) -> tuple[RunnerState, Transition]:
    def _env_step(carry: RolloutCarry, _: None) -> tuple[RolloutCarry, Transition]:
        (
            chaser_ts,
            evader_ts,
            env_state,
            chaser_obs,
            evader_obs,
            done,
            chaser_hstate,
            evader_hstate,
            rng,
        ) = carry

        last_done = done
        rng, chaser_rng, evader_rng, step_rng = jax.random.split(rng, 4)

        chaser_ac_in = (
            chaser_obs[np.newaxis, :],
            done[np.newaxis, :],
        )
        chaser_hstate, chaser_pi, chaser_value = chaser_network.apply(  # type: ignore[reportAssignmentType]
            chaser_ts.params, chaser_hstate, chaser_ac_in
        )
        chaser_action_raw = chaser_pi.sample(seed=chaser_rng)  # type: ignore[reportAttributeAccessIssue]
        chaser_log_prob = chaser_pi.log_prob(chaser_action_raw)  # type: ignore[reportAttributeAccessIssue]
        chaser_value = chaser_value.squeeze(0)
        chaser_action_raw = chaser_action_raw.squeeze(0)
        chaser_log_prob = chaser_log_prob.squeeze(0)
        chaser_action = jnp.clip(chaser_action_raw, -1.0, 1.0)

        evader_ac_in = (
            evader_obs[np.newaxis, :],
            done[np.newaxis, :],
        )
        evader_hstate, evader_pi, evader_value = evader_network.apply(  # type: ignore[reportAssignmentType]
            evader_ts.params, evader_hstate, evader_ac_in
        )
        evader_action_raw = evader_pi.sample(seed=evader_rng)  # type: ignore[reportAttributeAccessIssue]
        evader_log_prob = evader_pi.log_prob(evader_action_raw)  # type: ignore[reportAttributeAccessIssue]
        evader_value = evader_value.squeeze(0)
        evader_action_raw = evader_action_raw.squeeze(0)
        evader_log_prob = evader_log_prob.squeeze(0)
        evader_action = jnp.clip(evader_action_raw, -1.0, 1.0)

        step_rngs = jax.random.split(step_rng, rl_config.num_envs)
        (
            env_state,
            next_chaser_obs,
            next_evader_obs,
            chaser_reward,
            evader_reward,
            done,
            info,
        ) = jax.vmap(auto_reset_step, in_axes=(None, None, 0, 0, 0, 0))(
            env,
            env_config,
            env_state,
            chaser_action,
            evader_action,
            step_rngs,
        )

        transition = Transition(
            done=last_done,
            chaser_action=chaser_action,
            evader_action=evader_action,
            chaser_value=chaser_value,
            evader_value=evader_value,
            chaser_reward=chaser_reward,
            evader_reward=evader_reward,
            chaser_log_prob=chaser_log_prob,
            evader_log_prob=evader_log_prob,
            chaser_obs=chaser_obs,
            evader_obs=evader_obs,
            info=info,
        )
        carry = RolloutCarry(
            chaser_ts,
            evader_ts,
            env_state,
            next_chaser_obs,
            next_evader_obs,
            done,
            chaser_hstate,
            evader_hstate,
            rng,
        )
        return carry, transition

    final_carry, traj_batch = jax.lax.scan(
        _env_step, RolloutCarry(*runner_state), None, rl_config.num_steps
    )
    return RunnerState(*final_carry), traj_batch


def bootstrap_values(
    runner_state: RunnerState,
    chaser_network: ActorCriticModule,
    evader_network: ActorCriticModule,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    (
        chaser_ts,
        evader_ts,
        _,
        last_chaser_obs,
        last_evader_obs,
        last_done,
        chaser_hstate,
        evader_hstate,
        _,
    ) = runner_state

    chaser_ac_in = (
        last_chaser_obs[np.newaxis, :],
        last_done[np.newaxis, :],
    )
    _, _, chaser_last_val = chaser_network.apply(  # type: ignore[reportAssignmentType]
        chaser_ts.params, chaser_hstate, chaser_ac_in
    )

    evader_ac_in = (
        last_evader_obs[np.newaxis, :],
        last_done[np.newaxis, :],
    )
    _, _, evader_last_val = evader_network.apply(  # type: ignore[reportAssignmentType]
        evader_ts.params, evader_hstate, evader_ac_in
    )

    return chaser_last_val.squeeze(0), evader_last_val.squeeze(0), last_done


def split_agent_trajectories(
    traj_batch: Transition,
) -> tuple[AgentTransition, AgentTransition]:
    chaser_traj = AgentTransition(
        done=traj_batch.done,
        action=traj_batch.chaser_action,
        value=traj_batch.chaser_value,
        reward=traj_batch.chaser_reward,
        log_prob=traj_batch.chaser_log_prob,
        obs=traj_batch.chaser_obs,
    )
    evader_traj = AgentTransition(
        done=traj_batch.done,
        action=traj_batch.evader_action,
        value=traj_batch.evader_value,
        reward=traj_batch.evader_reward,
        log_prob=traj_batch.evader_log_prob,
        obs=traj_batch.evader_obs,
    )
    return chaser_traj, evader_traj
