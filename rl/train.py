from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from environment.config import EnvironmentConfig
from environment.environment import (
    TagEnvironment,
    TagEnvironmentState,
    TagEnvironmentStepInfo,
    observation_size,
)
from rl.config import RLConfig
from rl.models import ActorCriticRNN, ScannedRNN


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


def make_train(rl_config: RLConfig, env_config: EnvironmentConfig):
    num_updates = rl_config.total_timesteps // rl_config.num_steps // rl_config.num_envs

    env = TagEnvironment(env_config, rl_config.num_envs)
    obs_size = observation_size(env_config)
    action_dim = (2,)

    chaser_network = ActorCriticRNN(
        action_dim=action_dim, hidden_size=rl_config.hidden_size
    )
    evader_network = ActorCriticRNN(
        action_dim=action_dim, hidden_size=rl_config.hidden_size
    )

    def _auto_reset_step(state, chaser_action, evader_action, rng):
        stepped, chaser_obs, evader_obs, chaser_reward, evader_reward, done, info = (
            env.step(state, chaser_action, evader_action)
        )
        reset_state, reset_chaser_obs, reset_evader_obs = env.reset(rng)

        # Swap only essential state fields — avoids jax.tree.map over DataWarp
        # which leaks tracers on non-vmappable contact fields.
        # Derived quantities are recomputed by mjx.step()/forward() next step.
        new_mjx_data = stepped.mjx_data.replace(
            qpos=jnp.where(done, reset_state.mjx_data.qpos, stepped.mjx_data.qpos),
            qvel=jnp.where(done, reset_state.mjx_data.qvel, stepped.mjx_data.qvel),
            time=jnp.where(done, reset_state.mjx_data.time, stepped.mjx_data.time),
        )
        state = TagEnvironmentState(
            mjx_data=new_mjx_data,
            step_count=jnp.where(done, reset_state.step_count, stepped.step_count),
            tagged=jnp.where(done, reset_state.tagged, stepped.tagged),
            prev_distance=jnp.where(
                done, reset_state.prev_distance, stepped.prev_distance
            ),
        )

        chaser_obs = jnp.where(done, reset_chaser_obs, chaser_obs)
        evader_obs = jnp.where(done, reset_evader_obs, evader_obs)
        return state, chaser_obs, evader_obs, chaser_reward, evader_reward, done, info

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (rl_config.num_minibatches * rl_config.update_epochs))
            / num_updates
        )
        return rl_config.lr * frac

    def init(rng):
        rng, chaser_rng, evader_rng = jax.random.split(rng, 3)
        init_x = (
            jnp.zeros((1, rl_config.num_envs, obs_size)),
            jnp.zeros((1, rl_config.num_envs)),
        )
        init_hstate = ScannedRNN.initialize_carry(
            rl_config.num_envs, rl_config.hidden_size
        )

        chaser_params = chaser_network.init(chaser_rng, init_hstate, init_x)
        evader_params = evader_network.init(evader_rng, init_hstate, init_x)

        if rl_config.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(rl_config.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(rl_config.max_grad_norm),
                optax.adam(rl_config.lr, eps=1e-5),
            )

        chaser_ts = TrainState.create(
            apply_fn=chaser_network.apply, params=chaser_params, tx=tx
        )
        evader_ts = TrainState.create(
            apply_fn=evader_network.apply, params=evader_params, tx=tx
        )

        # Init env
        rng, _rng = jax.random.split(rng)
        reset_rngs = jax.random.split(_rng, rl_config.num_envs)
        env_states, chaser_obs, evader_obs = jax.vmap(env.reset)(reset_rngs)

        chaser_hstate = ScannedRNN.initialize_carry(
            rl_config.num_envs, rl_config.hidden_size
        )
        evader_hstate = ScannedRNN.initialize_carry(
            rl_config.num_envs, rl_config.hidden_size
        )

        rng, _rng = jax.random.split(rng)
        runner_state = (
            chaser_ts,
            evader_ts,
            env_states,
            chaser_obs,
            evader_obs,
            jnp.zeros(rl_config.num_envs, dtype=bool),
            chaser_hstate,
            evader_hstate,
            _rng,
        )
        return runner_state

    def _calculate_gae(rewards, values, dones, last_val, last_done):
        def _get_advantages(carry, transition):
            gae, next_value, next_done = carry
            done, value, reward = transition
            delta = reward + rl_config.gamma * next_value * (1 - next_done) - value
            gae = delta + rl_config.gamma * rl_config.gae_lambda * (1 - next_done) * gae
            return (gae, value, done), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val, last_done),
            (dones, values, rewards),
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + values

    def _update_agent(
        network, train_state, init_hstate, agent_traj, advantages, targets, rng
    ):
        def _update_epoch(update_state, _):
            def _update_minibatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                    _, pi, value = network.apply(
                        params,
                        init_hstate[0],
                        (traj_batch.obs, traj_batch.done),
                    )
                    log_prob = pi.log_prob(traj_batch.action)

                    # Value loss
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-rl_config.clip_eps, rl_config.clip_eps)
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # Actor loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - rl_config.clip_eps,
                            1.0 + rl_config.clip_eps,
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                        loss_actor
                        + rl_config.vf_coef * value_loss
                        - rl_config.ent_coef * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, init_hstate, traj_batch, advantages, targets, rng = (
                update_state
            )

            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, rl_config.num_envs)
            batch = (init_hstate, traj_batch, advantages, targets)

            shuffled_batch = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )

            minibatches = jax.tree.map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], rl_config.num_minibatches, -1] + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled_batch,
            )

            train_state, total_loss = jax.lax.scan(
                _update_minibatch, train_state, minibatches
            )
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            return update_state, total_loss

        update_state = (
            train_state,
            init_hstate,
            agent_traj,
            advantages,
            targets,
            rng,
        )
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, rl_config.update_epochs
        )
        return update_state[0], loss_info

    def step(runner_state):
        def _env_step(carry, _):
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

            # Forward pass for chaser
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

            # Forward pass for evader
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

            # Step env
            step_rngs = jax.random.split(step_rng, rl_config.num_envs)
            (
                env_state,
                next_chaser_obs,
                next_evader_obs,
                chaser_reward,
                evader_reward,
                done,
                info,
            ) = jax.vmap(_auto_reset_step)(
                env_state, chaser_action, evader_action, step_rngs
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
            carry = (
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

        # Save initial hidden states for PPO update
        initial_chaser_hstate = runner_state[6]
        initial_evader_hstate = runner_state[7]

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, rl_config.num_steps
        )

        # Unpack runner state
        (
            chaser_ts,
            evader_ts,
            env_state,
            last_chaser_obs,
            last_evader_obs,
            last_done,
            chaser_hstate,
            evader_hstate,
            rng,
        ) = runner_state

        # Bootstrap values for both agents
        chaser_ac_in = (
            last_chaser_obs[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        _, _, chaser_last_val = chaser_network.apply(  # type: ignore[reportAssignmentType]
            chaser_ts.params, chaser_hstate, chaser_ac_in
        )
        chaser_last_val = chaser_last_val.squeeze(0)

        evader_ac_in = (
            last_evader_obs[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        _, _, evader_last_val = evader_network.apply(  # type: ignore[reportAssignmentType]
            evader_ts.params, evader_hstate, evader_ac_in
        )
        evader_last_val = evader_last_val.squeeze(0)

        # GAE for both agents
        chaser_advantages, chaser_targets = _calculate_gae(
            traj_batch.chaser_reward,
            traj_batch.chaser_value,
            traj_batch.done,
            chaser_last_val,
            last_done,
        )
        evader_advantages, evader_targets = _calculate_gae(
            traj_batch.evader_reward,
            traj_batch.evader_value,
            traj_batch.done,
            evader_last_val,
            last_done,
        )

        # Extract AgentTransition for each agent
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

        # Update both agents
        rng, chaser_update_rng, evader_update_rng = jax.random.split(rng, 3)

        chaser_ts, chaser_loss = _update_agent(
            chaser_network,
            chaser_ts,
            initial_chaser_hstate[None, :],
            chaser_traj,
            chaser_advantages,
            chaser_targets,
            chaser_update_rng,
        )
        evader_ts, evader_loss = _update_agent(
            evader_network,
            evader_ts,
            initial_evader_hstate[None, :],
            evader_traj,
            evader_advantages,
            evader_targets,
            evader_update_rng,
        )

        # Compute metrics
        finished = traj_batch.info.tagged | traj_batch.info.time_up
        n_finished = finished.astype(jnp.float32).sum()
        mean_ep_length = jnp.where(
            n_finished > 0,
            (traj_batch.info.step_count * finished.astype(jnp.float32)).sum()
            / n_finished,
            jnp.float32(0),
        )
        tag_rate = jnp.where(
            n_finished > 0,
            traj_batch.info.tagged.astype(jnp.float32).sum() / n_finished,
            jnp.float32(0),
        )

        metrics = {
            # Losses (mean over update_epochs x num_minibatches)
            "chaser/total_loss": chaser_loss[0].mean(),
            "chaser/value_loss": chaser_loss[1][0].mean(),
            "chaser/actor_loss": chaser_loss[1][1].mean(),
            "chaser/entropy": chaser_loss[1][2].mean(),
            "evader/total_loss": evader_loss[0].mean(),
            "evader/value_loss": evader_loss[1][0].mean(),
            "evader/actor_loss": evader_loss[1][1].mean(),
            "evader/entropy": evader_loss[1][2].mean(),
            # Environment
            "env/tag_rate": tag_rate,
            "env/mean_distance": traj_batch.info.distance.mean(),
            "env/mean_episode_length": mean_ep_length,
            # Rewards
            "chaser/mean_reward": traj_batch.chaser_reward.mean(),
            "evader/mean_reward": traj_batch.evader_reward.mean(),
            "chaser/mean_action_magnitude": jnp.abs(traj_batch.chaser_action).mean(),
            "evader/mean_action_magnitude": jnp.abs(traj_batch.evader_action).mean(),
        }

        runner_state = (
            chaser_ts,
            evader_ts,
            env_state,
            last_chaser_obs,
            last_evader_obs,
            last_done,
            chaser_hstate,
            evader_hstate,
            rng,
        )
        return runner_state, metrics

    return init, step, num_updates, env, chaser_network, evader_network


def make_video_rollout(video_env, chaser_network, evader_network, rl_config):
    """Create a JIT'd rollout function for video recording (compiles once)."""

    @jax.jit
    def rollout(chaser_params, evader_params, rng):
        state, chaser_obs, evader_obs = video_env.reset(rng)
        chaser_hstate = ScannedRNN.initialize_carry(1, rl_config.hidden_size)
        evader_hstate = ScannedRNN.initialize_carry(1, rl_config.hidden_size)
        done = jnp.bool_(False)

        def _scan_step(carry, _):
            state, chaser_obs, evader_obs, chaser_hstate, evader_hstate, done = carry
            qpos = state.mjx_data.qpos
            qvel = state.mjx_data.qvel

            chaser_ac_in = (
                chaser_obs[jnp.newaxis, jnp.newaxis, :],
                done[jnp.newaxis, jnp.newaxis],
            )
            chaser_hstate, chaser_pi, _ = chaser_network.apply(
                chaser_params, chaser_hstate, chaser_ac_in
            )
            chaser_action = jnp.clip(chaser_pi.loc[0, 0], -1.0, 1.0)

            evader_ac_in = (
                evader_obs[jnp.newaxis, jnp.newaxis, :],
                done[jnp.newaxis, jnp.newaxis],
            )
            evader_hstate, evader_pi, _ = evader_network.apply(
                evader_params, evader_hstate, evader_ac_in
            )
            evader_action = jnp.clip(evader_pi.loc[0, 0], -1.0, 1.0)

            # Zero actions after episode ends so agents stay still
            chaser_action = jnp.where(done, jnp.zeros(2), chaser_action)
            evader_action = jnp.where(done, jnp.zeros(2), evader_action)

            state, chaser_obs, evader_obs, _, _, step_done, _ = video_env.step(
                state, chaser_action, evader_action
            )
            done = done | step_done

            carry = (state, chaser_obs, evader_obs, chaser_hstate, evader_hstate, done)
            return carry, (qpos, qvel, step_done)

        init_carry = (state, chaser_obs, evader_obs, chaser_hstate, evader_hstate, done)
        _, (all_qpos, all_qvel, all_done) = jax.lax.scan(
            _scan_step, init_carry, None, length=video_env.max_steps
        )
        return all_qpos, all_qvel, all_done

    return rollout


def record_video(video_env, rollout_fn, runner_state, global_step, rng, fps, writer):
    import mujoco

    chaser_params = runner_state[0].params
    evader_params = runner_state[1].params
    all_qpos, all_qvel, all_done = rollout_fn(chaser_params, evader_params, rng)

    # Find episode end for trimming
    all_done_np = np.array(all_done)
    done_indices = np.where(all_done_np)[0]
    n_frames = int(done_indices[0]) + 1 if len(done_indices) > 0 else len(all_done_np)

    # Transfer to CPU once
    all_qpos_np = np.array(all_qpos[:n_frames])
    all_qvel_np = np.array(all_qvel[:n_frames])

    # Render on CPU
    renderer = mujoco.Renderer(video_env.mj_model, height=480, width=640)
    mj_data = mujoco.MjData(video_env.mj_model)

    frames = []
    for i in range(n_frames):
        mj_data.qpos[:] = all_qpos_np[i]
        mj_data.qvel[:] = all_qvel_np[i]
        mujoco.mj_forward(video_env.mj_model, mj_data)
        renderer.update_scene(mj_data)
        frames.append(renderer.render())

    renderer.close()

    # (T, H, W, C) -> (1, T, C, H, W) for tensorboard
    video_array = np.stack(frames)
    video_array = video_array.transpose(0, 3, 1, 2)[np.newaxis, ...]
    writer.add_video("video/policy", video_array, global_step, fps=fps)
    print(f"Recorded video at step {global_step} ({n_frames} frames)")


if __name__ == "__main__":
    from tensorboardX import SummaryWriter
    from tqdm import tqdm

    rl_config = RLConfig()
    env_config = EnvironmentConfig()
    rng = jax.random.PRNGKey(rl_config.seed)

    init_fn, step_fn, num_updates, env, chaser_network, evader_network = make_train(
        rl_config, env_config
    )
    runner_state = jax.jit(init_fn)(rng)
    step_jit = jax.jit(step_fn)

    # Create video env and JIT'd rollout once (compiles on first call)
    video_env = TagEnvironment(env_config, 1)
    video_rollout_fn = make_video_rollout(
        video_env, chaser_network, evader_network, rl_config
    )

    writer = SummaryWriter()
    timesteps_per_update = rl_config.num_steps * rl_config.num_envs

    for update in tqdm(range(num_updates), desc="Training"):
        runner_state, metrics = step_jit(runner_state)
        global_step = (update + 1) * timesteps_per_update

        for key, value in metrics.items():
            writer.add_scalar(key, value.item(), global_step)
        writer.flush()

        # Periodic video recording
        if rl_config.video_interval > 0 and (
            ((update + 1) % rl_config.video_interval == 0) or update == 0
        ):
            rng, video_rng = jax.random.split(rng)
            record_video(
                video_env,
                video_rollout_fn,
                runner_state,
                global_step,
                video_rng,
                env_config.action_frequency,
                writer,
            )

    writer.close()
