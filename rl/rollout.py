from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from environment.environment import (
    TagEnvironment,
    TagEnvironmentState,
    TagEnvironmentStepInfo,
)
from rl.config import RLConfig
from rl.policy import PolicyModule


class Transition(NamedTuple):
    done: jax.Array
    chaser_action: jax.Array
    chaser_action_raw: jax.Array
    evader_action: jax.Array
    evader_action_raw: jax.Array
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
    prev_done: jax.Array
    chaser_hstate: jax.Array
    evader_hstate: jax.Array
    rng: jax.Array


class RolloutCarry(NamedTuple):
    chaser_ts: TrainState
    evader_ts: TrainState
    env_state: TagEnvironmentState
    chaser_obs: jax.Array
    evader_obs: jax.Array
    prev_done: jax.Array
    chaser_hstate: jax.Array
    evader_hstate: jax.Array
    rng: jax.Array


def auto_reset_step(
    env: TagEnvironment,
    state: TagEnvironmentState,
    chaser_action: jax.Array,
    evader_action: jax.Array,
    rng: jax.Array,
    curriculum_progress: jax.Array,
) -> tuple[
    TagEnvironmentState,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    TagEnvironmentStepInfo,
]:
    step_rng, reset_rng = jax.random.split(rng)
    (
        stepped,
        chaser_obs,
        evader_obs,
        chaser_reward,
        evader_reward,
        done,
        info,
    ) = env.step(state, chaser_action, evader_action, step_rng)

    # Guard against DR-induced physics instability before it can poison GAE/PPO.
    physics_ok = (
        jnp.isfinite(chaser_obs).all()
        & jnp.isfinite(evader_obs).all()
        & jnp.isfinite(chaser_reward)
        & jnp.isfinite(evader_reward)
        & jnp.isfinite(info.distance)
        & ~info.physics_invalid
        & ~info.reward_invalid
    )
    chaser_reward = jnp.where(physics_ok, chaser_reward, jnp.float32(0.0))
    evader_reward = jnp.where(physics_ok, evader_reward, jnp.float32(0.0))
    effective_done = done | ~physics_ok

    def reset_branch(_: None) -> tuple[TagEnvironmentState, jax.Array, jax.Array]:
        return env.reset(reset_rng, curriculum_progress)

    def continue_branch(_: None) -> tuple[TagEnvironmentState, jax.Array, jax.Array]:
        return stepped, chaser_obs, evader_obs

    state, chaser_obs, evader_obs = jax.lax.cond(
        effective_done, reset_branch, continue_branch, operand=None
    )
    return (
        state,
        chaser_obs,
        evader_obs,
        chaser_reward,
        evader_reward,
        effective_done,
        info,
    )


def collect_trajectories(
    runner_state: RunnerState,
    env: TagEnvironment,
    rl_config: RLConfig,
    chaser_network: PolicyModule,
    evader_network: PolicyModule,
    curriculum_progress: jax.Array,
) -> tuple[RunnerState, Transition]:
    def _env_step(carry: RolloutCarry, _: None) -> tuple[RolloutCarry, Transition]:
        (
            chaser_ts,
            evader_ts,
            env_state,
            chaser_obs,
            evader_obs,
            prev_done,
            chaser_hstate,
            evader_hstate,
            rng,
        ) = carry

        transition_done = prev_done
        rng, chaser_rng, evader_rng, step_rng = jax.random.split(rng, 4)

        chaser_ac_in = (
            chaser_obs[None, :],
            prev_done[None, :],
        )
        chaser_hstate, chaser_pi, chaser_value = chaser_network.apply(
            chaser_ts.params, chaser_hstate, chaser_ac_in
        )
        chaser_action_raw = chaser_pi.sample(seed=chaser_rng)
        chaser_log_prob = jnp.asarray(chaser_pi.log_prob(chaser_action_raw))
        chaser_value = chaser_value.squeeze(0)
        chaser_action_raw = chaser_action_raw.squeeze(0)
        chaser_log_prob = chaser_log_prob.squeeze(0)
        chaser_action = jnp.clip(chaser_action_raw, -1.0, 1.0)

        evader_ac_in = (
            evader_obs[None, :],
            prev_done[None, :],
        )
        evader_hstate, evader_pi, evader_value = evader_network.apply(
            evader_ts.params, evader_hstate, evader_ac_in
        )
        evader_action_raw = evader_pi.sample(seed=evader_rng)
        evader_log_prob = jnp.asarray(evader_pi.log_prob(evader_action_raw))
        evader_value = evader_value.squeeze(0)
        evader_action_raw = evader_action_raw.squeeze(0)
        evader_log_prob = evader_log_prob.squeeze(0)
        evader_action = jnp.clip(evader_action_raw, -1.0, 1.0)

        step_rngs = jax.random.split(step_rng, chaser_action.shape[0])
        (
            env_state,
            next_chaser_obs,
            next_evader_obs,
            chaser_reward,
            evader_reward,
            next_prev_done,
            info,
        ) = jax.vmap(auto_reset_step, in_axes=(None, 0, 0, 0, 0, None))(
            env,
            env_state,
            chaser_action,
            evader_action,
            step_rngs,
            curriculum_progress,
        )

        transition = Transition(
            done=transition_done,
            chaser_action=chaser_action,
            chaser_action_raw=chaser_action_raw,
            evader_action=evader_action,
            evader_action_raw=evader_action_raw,
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
            next_prev_done,
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
    chaser_network: PolicyModule,
    evader_network: PolicyModule,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    (
        chaser_ts,
        evader_ts,
        _,
        last_chaser_obs,
        last_evader_obs,
        last_prev_done,
        chaser_hstate,
        evader_hstate,
        _,
    ) = runner_state

    chaser_ac_in = (
        last_chaser_obs[None, :],
        last_prev_done[None, :],
    )
    _, _, chaser_last_val = chaser_network.apply(
        chaser_ts.params, chaser_hstate, chaser_ac_in
    )

    evader_ac_in = (
        last_evader_obs[None, :],
        last_prev_done[None, :],
    )
    _, _, evader_last_val = evader_network.apply(
        evader_ts.params, evader_hstate, evader_ac_in
    )

    return chaser_last_val.squeeze(0), evader_last_val.squeeze(0), last_prev_done


def split_agent_trajectories(
    traj_batch: Transition,
) -> tuple[AgentTransition, AgentTransition]:
    chaser_traj = AgentTransition(
        done=traj_batch.done,
        action=traj_batch.chaser_action_raw,
        value=traj_batch.chaser_value,
        reward=traj_batch.chaser_reward,
        log_prob=traj_batch.chaser_log_prob,
        obs=traj_batch.chaser_obs,
    )
    evader_traj = AgentTransition(
        done=traj_batch.done,
        action=traj_batch.evader_action_raw,
        value=traj_batch.evader_value,
        reward=traj_batch.evader_reward,
        log_prob=traj_batch.evader_log_prob,
        obs=traj_batch.evader_obs,
    )
    return chaser_traj, evader_traj
