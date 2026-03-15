import argparse
import os
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter
from tqdm import tqdm

from environment.config import EnvironmentConfig
from environment.environment import TagEnvironment, observation_size
from rl.checkpoints import (
    load_checkpoint,
    load_checkpoint_configs,
    restore_runner_state,
    save_checkpoint,
)
from rl.config import RLConfig
from rl.metrics import compute_training_metrics
from rl.models import ScannedRNN
from rl.policy import build_policies
from rl.ppo import calculate_gae, update_agent
from rl.rollout import (
    RunnerState,
    bootstrap_values,
    collect_trajectories,
    split_agent_trajectories,
)
from rl.sharding import make_shardings, shard_runner_state

TrainMetrics = dict[str, jax.Array]
TrainInitFn = Callable[[jax.Array], RunnerState]
TrainStepFn = Callable[[RunnerState], tuple[RunnerState, TrainMetrics]]


def make_train(
    rl_config: RLConfig, env_config: EnvironmentConfig
) -> tuple[TrainInitFn, TrainStepFn, int]:
    num_updates = rl_config.num_updates
    num_envs = rl_config.num_envs

    env = TagEnvironment(env_config)
    obs_size = observation_size(env_config)
    chaser_network, evader_network = build_policies(rl_config.hidden_size, env_config)

    def linear_schedule(count: int | float | jax.Array) -> int | float | jax.Array:
        updates_completed = count // (
            rl_config.num_minibatches * rl_config.update_epochs
        )
        frac = 1.0 - updates_completed / max(num_updates, 1)
        return rl_config.lr * frac

    if rl_config.anneal_lr:
        tx = optax.chain(
            optax.clip_by_global_norm(rl_config.max_grad_norm),
            optax.adam(learning_rate=cast(optax.Schedule, linear_schedule), eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(rl_config.max_grad_norm),
            optax.adam(rl_config.lr, eps=1e-5),
        )

    def init(base_rng: jax.Array) -> RunnerState:
        param_rng, env_rng, runner_rng = jax.random.split(base_rng, 3)
        chaser_rng, evader_rng = jax.random.split(param_rng)

        init_x = (
            jnp.zeros((1, num_envs, obs_size)),
            jnp.zeros((1, num_envs)),
        )
        init_hstate = ScannedRNN.initialize_carry(num_envs, rl_config.hidden_size)

        chaser_params = chaser_network.init(chaser_rng, init_hstate, init_x)
        evader_params = evader_network.init(evader_rng, init_hstate, init_x)

        chaser_ts = TrainState.create(
            apply_fn=chaser_network.apply, params=chaser_params, tx=tx
        )
        evader_ts = TrainState.create(
            apply_fn=evader_network.apply, params=evader_params, tx=tx
        )

        reset_rngs = jax.random.split(env_rng, num_envs)
        env_states, chaser_obs, evader_obs = jax.vmap(env.reset)(reset_rngs)

        chaser_hstate = ScannedRNN.initialize_carry(num_envs, rl_config.hidden_size)
        evader_hstate = ScannedRNN.initialize_carry(num_envs, rl_config.hidden_size)

        return RunnerState(
            chaser_ts=chaser_ts,
            evader_ts=evader_ts,
            env_state=env_states,
            chaser_obs=chaser_obs,
            evader_obs=evader_obs,
            prev_done=jnp.zeros(num_envs, dtype=bool),
            chaser_hstate=chaser_hstate,
            evader_hstate=evader_hstate,
            rng=runner_rng,
        )

    def step(runner_state: RunnerState) -> tuple[RunnerState, TrainMetrics]:
        initial_chaser_hstate = runner_state.chaser_hstate
        initial_evader_hstate = runner_state.evader_hstate

        runner_state, traj_batch = collect_trajectories(
            runner_state,
            env,
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

        rng, chaser_update_rng, evader_update_rng = jax.random.split(
            runner_state.rng, 3
        )

        chaser_ts, chaser_loss = update_agent(
            rl_config,
            chaser_network,
            runner_state.chaser_ts,
            initial_chaser_hstate[None, :],
            chaser_traj,
            chaser_advantages,
            chaser_targets,
            chaser_update_rng,
        )
        evader_ts, evader_loss = update_agent(
            rl_config,
            evader_network,
            runner_state.evader_ts,
            initial_evader_hstate[None, :],
            evader_traj,
            evader_advantages,
            evader_targets,
            evader_update_rng,
        )

        metrics = compute_training_metrics(traj_batch, chaser_loss, evader_loss)

        return (
            RunnerState(
                chaser_ts=chaser_ts,
                evader_ts=evader_ts,
                env_state=runner_state.env_state,
                chaser_obs=runner_state.chaser_obs,
                evader_obs=runner_state.evader_obs,
                prev_done=runner_state.prev_done,
                chaser_hstate=runner_state.chaser_hstate,
                evader_hstate=runner_state.evader_hstate,
                rng=rng,
            ),
            metrics,
        )

    return init, step, num_updates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--num-devices", type=int, default=None)
    return parser.parse_args()


def resolve_logdir(logdir: str | None, resume: str | None) -> str | None:
    if logdir is not None:
        return logdir
    if resume is None:
        return None
    checkpoint_dir = os.path.dirname(resume)
    if os.path.basename(checkpoint_dir) != "checkpoints":
        raise ValueError(
            "Resumed checkpoints must live under a 'checkpoints' directory when logdir is omitted"
        )
    return os.path.dirname(checkpoint_dir)


def unreplicate_metrics(metrics: TrainMetrics) -> dict[str, float]:
    host_metrics = jax.device_get(metrics)
    return {key: float(value) for key, value in host_metrics.items()}


def load_training_configs(
    resume_path: str | None,
) -> tuple[RLConfig, EnvironmentConfig, dict[str, object] | None]:
    if resume_path is None:
        rl_config = RLConfig()
        env_config = EnvironmentConfig()
        checkpoint = None
    else:
        checkpoint = load_checkpoint(resume_path)
        rl_config, env_config = load_checkpoint_configs(checkpoint)

    return rl_config, env_config, checkpoint


def resolve_devices(num_devices_override: int | None) -> list[Any]:
    devices = list(jax.local_devices())
    if num_devices_override is None:
        return devices
    if num_devices_override <= 0:
        raise ValueError("--num-devices must be positive")
    if num_devices_override > len(devices):
        raise ValueError(
            f"Requested {num_devices_override} devices but only {len(devices)} are available"
        )
    return devices[:num_devices_override]


def main() -> None:
    args = parse_args()
    rl_config, env_config, checkpoint = load_training_configs(args.resume)

    devices = resolve_devices(args.num_devices)
    env_config.validate()
    rl_config.validate(len(devices))
    shardings = make_shardings(devices)
    print(
        "Training configuration: "
        f"devices={len(devices)}, "
        f"global_envs={rl_config.num_envs}, "
        f"envs_per_device={rl_config.num_envs // len(devices)}, "
        f"num_steps={rl_config.num_steps}, "
        f"num_minibatches={rl_config.num_minibatches}, "
        f"timesteps_per_update={rl_config.timesteps_per_update}"
    )

    init_fn, step_fn, num_updates = make_train(rl_config, env_config)
    init_jit = jax.jit(init_fn, device=devices[0])
    step_jit = jax.jit(step_fn, donate_argnums=(0,))

    writer = SummaryWriter(logdir=resolve_logdir(args.logdir, args.resume))
    checkpoint_dir = os.path.join(writer.logdir, "checkpoints")
    flush_interval = 10

    if checkpoint is None:
        rng = jax.random.PRNGKey(rl_config.seed)
        runner_state = shard_runner_state(init_jit(rng), shardings)
        global_step = 0
        start_update = 0
    else:
        rng = jax.random.PRNGKey(rl_config.seed)
        template_runner_state = init_jit(rng)
        runner_state = restore_runner_state(
            checkpoint, template_runner_state, shardings
        )
        global_step = int(cast(int, checkpoint["global_step"]))
        if global_step % rl_config.timesteps_per_update != 0:
            raise ValueError(
                "Checkpoint global_step is not aligned with the configured timesteps_per_update"
            )
        start_update = global_step // rl_config.timesteps_per_update
        print(f"Resumed training from {args.resume} at global_step={global_step}")

    for update in tqdm(
        range(start_update, num_updates),
        initial=start_update,
        total=num_updates,
        desc="Training",
    ):
        runner_state, metrics = step_jit(runner_state)
        global_step = (update + 1) * rl_config.timesteps_per_update

        for key, value in unreplicate_metrics(metrics).items():
            writer.add_scalar(key, value, global_step)
        if (update + 1) % flush_interval == 0:
            writer.flush()

        if rl_config.checkpoint_interval > 0 and (
            (update + 1) % rl_config.checkpoint_interval == 0
        ):
            save_checkpoint(
                runner_state,
                global_step,
                checkpoint_dir,
                rl_config,
                env_config,
            )
            writer.flush()

    save_checkpoint(
        runner_state,
        global_step,
        checkpoint_dir,
        rl_config,
        env_config,
    )
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
