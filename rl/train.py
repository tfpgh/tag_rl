from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from environment.config import EnvironmentConfig
from environment.environment import TagEnvironment, observation_size
from rl.checkpoints import save_checkpoint
from rl.config import RLConfig
from rl.metrics import compute_training_metrics
from rl.models import ActorCriticCNNRNN, ScannedRNN
from rl.ppo import calculate_gae, update_agent
from rl.rollout import (
    RunnerState,
    bootstrap_values,
    collect_trajectories,
    split_agent_trajectories,
)

TrainMetrics = dict[str, jax.Array]
TrainInitFn = Callable[[jax.Array], RunnerState]
TrainStepFn = Callable[[RunnerState], tuple[RunnerState, TrainMetrics]]


def make_train(
    rl_config: RLConfig, env_config: EnvironmentConfig
) -> tuple[
    TrainInitFn,
    TrainStepFn,
    int,
    TagEnvironment,
    ActorCriticCNNRNN,
    ActorCriticCNNRNN,
]:
    num_updates = rl_config.total_timesteps // rl_config.num_steps // rl_config.num_envs

    env = TagEnvironment(env_config, rl_config.num_envs)
    obs_size = observation_size(env_config)
    action_dim = (2,)

    chaser_network = ActorCriticCNNRNN(
        action_dim=action_dim,
        hidden_size=rl_config.hidden_size,
        n_rays=env_config.n_rays,
    )
    evader_network = ActorCriticCNNRNN(
        action_dim=action_dim,
        hidden_size=rl_config.hidden_size,
        n_rays=env_config.n_rays,
    )

    def linear_schedule(count: int | float | jax.Array) -> int | float | jax.Array:
        frac = (
            1.0
            - (count // (rl_config.num_minibatches * rl_config.update_epochs))
            / num_updates
        )
        return rl_config.lr * frac

    def init(rng: jax.Array) -> RunnerState:
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
                optax.adam(
                    learning_rate=cast(optax.Schedule, linear_schedule), eps=1e-5
                ),
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
        runner_state = RunnerState(
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

    def step(runner_state: RunnerState) -> tuple[RunnerState, TrainMetrics]:
        initial_chaser_hstate = runner_state[6]
        initial_evader_hstate = runner_state[7]

        runner_state, traj_batch = collect_trajectories(
            runner_state,
            env,
            env_config,
            rl_config,
            chaser_network,
            evader_network,
        )

        chaser_last_val, evader_last_val, last_done = bootstrap_values(
            runner_state,
            chaser_network,
            evader_network,
        )

        chaser_advantages, chaser_targets = calculate_gae(
            rl_config,
            traj_batch.chaser_reward,
            traj_batch.chaser_value,
            traj_batch.done,
            chaser_last_val,
            last_done,
        )
        evader_advantages, evader_targets = calculate_gae(
            rl_config,
            traj_batch.evader_reward,
            traj_batch.evader_value,
            traj_batch.done,
            evader_last_val,
            last_done,
        )

        chaser_traj, evader_traj = split_agent_trajectories(traj_batch)

        rng = runner_state[-1]
        rng, chaser_update_rng, evader_update_rng = jax.random.split(rng, 3)

        chaser_ts, chaser_loss = update_agent(
            rl_config,
            chaser_network,
            runner_state[0],
            initial_chaser_hstate[None, :],
            chaser_traj,
            chaser_advantages,
            chaser_targets,
            chaser_update_rng,
        )
        evader_ts, evader_loss = update_agent(
            rl_config,
            evader_network,
            runner_state[1],
            initial_evader_hstate[None, :],
            evader_traj,
            evader_advantages,
            evader_targets,
            evader_update_rng,
        )

        metrics = compute_training_metrics(traj_batch, chaser_loss, evader_loss)

        runner_state = RunnerState(
            chaser_ts,
            evader_ts,
            runner_state[2],
            runner_state[3],
            runner_state[4],
            runner_state[5],
            runner_state[6],
            runner_state[7],
            rng,
        )
        return runner_state, metrics

    return init, step, num_updates, env, chaser_network, evader_network


if __name__ == "__main__":
    import os

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

    writer = SummaryWriter()
    checkpoint_dir = os.path.join(writer.logdir, "checkpoints")
    timesteps_per_update = rl_config.num_steps * rl_config.num_envs

    for update in tqdm(range(num_updates), desc="Training"):
        runner_state, metrics = step_jit(runner_state)
        global_step = (update + 1) * timesteps_per_update

        for key, value in metrics.items():
            writer.add_scalar(key, value.item(), global_step)
        writer.flush()

        # Periodic checkpoint saving
        if rl_config.checkpoint_interval > 0 and (
            (update + 1) % rl_config.checkpoint_interval == 0
        ):
            save_checkpoint(runner_state, global_step, checkpoint_dir)

    # Save final checkpoint
    save_checkpoint(runner_state, num_updates * timesteps_per_update, checkpoint_dir)
    writer.close()
