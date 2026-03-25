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
from environment.curriculum import curriculum_progress as compute_curriculum_progress
from environment.environment import TagEnvironment, observation_size
from environment.model_build import (
    build_mjx_models_parallel,
    nominal_domain_params,
    stack_mjx_models,
)
from environment.randomization import domain_params_distance, sample_domain_params
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

AXIS_NAME = "devices"
TrainMetrics = dict[str, jax.Array]
TrainInitFn = Callable[[jax.Array], RunnerState]
TrainStepFn = Callable[[RunnerState, jax.Array], tuple[RunnerState, TrainMetrics]]


def build_model_pool(env: TagEnvironment, rl_config: RLConfig):
    nominal = nominal_domain_params()
    rng = jax.random.PRNGKey(rl_config.seed)
    sampled: list[tuple[float, Any]] = []
    for index in range(max(rl_config.model_pool_size - 1, 0)):
        rng, sample_rng = jax.random.split(rng)
        params = sample_domain_params(
            sample_rng, env.config, jnp.asarray(1.0, dtype=jnp.float32)
        )
        distance = float(jax.device_get(domain_params_distance(params, env.config)))
        sampled.append((distance, params))
    sampled.sort(key=lambda item: item[0])
    domain_params_list = [nominal, *[params for _, params in sampled]]
    model_list = build_mjx_models_parallel(
        env.mj_model,
        env.model_indices,
        domain_params_list,
        max_workers=rl_config.model_pool_build_workers,
    )
    model_pool = stack_mjx_models(model_list)
    domain_param_pool = jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0), *domain_params_list
    )
    return model_pool, domain_param_pool


def learning_rate_for_update(
    rl_config: RLConfig, updates_completed: int | float | jax.Array
) -> int | float | jax.Array:
    if not rl_config.anneal_lr:
        return rl_config.lr

    start_fraction = rl_config.lr_decay_start_fraction
    total_updates = max(rl_config.num_updates, 1)
    start_update = start_fraction * total_updates
    updates_completed = jnp.asarray(updates_completed, dtype=jnp.float32)

    if start_fraction >= 1.0:
        scale = jnp.asarray(1.0, dtype=jnp.float32)
    else:
        decay_progress = jnp.clip(
            (updates_completed - start_update)
            / jnp.maximum(total_updates - start_update, 1.0),
            0.0,
            1.0,
        )
        start_scale = jnp.asarray(1.0, dtype=jnp.float32)
        end_scale = jnp.asarray(rl_config.final_lr_scale, dtype=jnp.float32)
        scale = start_scale + (end_scale - start_scale) * decay_progress
    return rl_config.lr * scale


def make_train(
    rl_config: RLConfig, env_config: EnvironmentConfig, num_devices: int
) -> tuple[TrainInitFn, TrainStepFn, int]:
    num_updates = rl_config.num_updates
    num_envs = rl_config.num_envs
    local_num_envs = num_envs // num_devices

    env = TagEnvironment(env_config)
    model_pool, domain_param_pool = build_model_pool(env, rl_config)
    obs_size = observation_size(env_config)
    chaser_network, evader_network = build_policies(rl_config.hidden_size, env_config)

    def linear_schedule(count: int | float | jax.Array) -> int | float | jax.Array:
        updates_completed = count // (
            rl_config.num_minibatches * rl_config.update_epochs
        )
        return learning_rate_for_update(rl_config, updates_completed)

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
        axis_index = jax.lax.axis_index(AXIS_NAME)
        param_rng, env_rng, runner_rng = jax.random.split(base_rng, 3)
        chaser_rng, evader_rng = jax.random.split(param_rng)

        init_x = (
            jnp.zeros((1, local_num_envs, obs_size)),
            jnp.zeros((1, local_num_envs)),
        )
        init_hstate = ScannedRNN.initialize_carry(local_num_envs, rl_config.hidden_size)

        chaser_params = chaser_network.init(chaser_rng, init_hstate, init_x)
        evader_params = evader_network.init(evader_rng, init_hstate, init_x)

        chaser_ts = TrainState.create(
            apply_fn=chaser_network.apply, params=chaser_params, tx=tx
        )
        evader_ts = TrainState.create(
            apply_fn=evader_network.apply, params=evader_params, tx=tx
        )

        device_env_rng = jax.random.fold_in(env_rng, axis_index)
        reset_rngs = jax.random.split(device_env_rng, local_num_envs)
        initial_model_indices = jnp.zeros(local_num_envs, dtype=jnp.int32)
        initial_models = jax.tree.map(lambda x: x[initial_model_indices], model_pool)
        initial_domain_params = jax.tree.map(
            lambda x: x[initial_model_indices], domain_param_pool
        )
        env_states, chaser_obs, evader_obs = jax.vmap(
            env.reset, in_axes=(0, None, 0, 0, 0)
        )(
            reset_rngs,
            jnp.asarray(0.0, dtype=jnp.float32),
            initial_models,
            initial_domain_params,
            initial_model_indices,
        )

        chaser_hstate = ScannedRNN.initialize_carry(
            local_num_envs, rl_config.hidden_size
        )
        evader_hstate = ScannedRNN.initialize_carry(
            local_num_envs, rl_config.hidden_size
        )

        return RunnerState(
            chaser_ts=chaser_ts,
            evader_ts=evader_ts,
            env_state=env_states,
            chaser_obs=chaser_obs,
            evader_obs=evader_obs,
            prev_done=jnp.zeros(local_num_envs, dtype=bool),
            chaser_hstate=chaser_hstate,
            evader_hstate=evader_hstate,
            rng=jax.random.fold_in(runner_rng, axis_index),
        )

    def step(
        runner_state: RunnerState, curriculum_progress: jax.Array
    ) -> tuple[RunnerState, TrainMetrics]:
        initial_chaser_hstate = runner_state.chaser_hstate
        initial_evader_hstate = runner_state.evader_hstate

        runner_state, traj_batch = collect_trajectories(
            runner_state,
            env,
            model_pool,
            domain_param_pool,
            rl_config,
            chaser_network,
            evader_network,
            curriculum_progress,
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
            axis_name=AXIS_NAME,
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
            axis_name=AXIS_NAME,
        )

        metrics = compute_training_metrics(
            traj_batch, chaser_loss, evader_loss, axis_name=AXIS_NAME
        )
        metrics["curriculum/progress"] = curriculum_progress

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
    host_metrics = jax.tree.map(lambda x: jax.device_get(x[0]), metrics)
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
    print(
        "Training configuration: "
        f"devices={len(devices)}, "
        f"global_envs={rl_config.num_envs}, "
        f"local_envs={rl_config.num_envs // len(devices)}, "
        f"model_pool_size={rl_config.model_pool_size}, "
        f"model_pool_build_workers={rl_config.model_pool_build_workers}, "
        f"num_steps={rl_config.num_steps}, "
        f"num_minibatches={rl_config.num_minibatches}, "
        f"timesteps_per_update={rl_config.timesteps_per_update}"
    )

    init_fn, step_fn, num_updates = make_train(rl_config, env_config, len(devices))
    init_pmap = jax.pmap(init_fn, axis_name=AXIS_NAME, devices=devices)
    step_pmap = jax.pmap(step_fn, axis_name=AXIS_NAME, devices=devices)

    writer = SummaryWriter(logdir=resolve_logdir(args.logdir, args.resume))
    checkpoint_dir = os.path.join(writer.logdir, "checkpoints")
    flush_interval = 10

    if checkpoint is None:
        rng = jax.random.PRNGKey(rl_config.seed)
        base_rngs = jnp.broadcast_to(rng, (len(devices),) + rng.shape)
        runner_state = init_pmap(base_rngs)
        global_step = 0
        start_update = 0
    else:
        rng = jax.random.PRNGKey(rl_config.seed)
        base_rngs = jnp.broadcast_to(rng, (len(devices),) + rng.shape)
        template_runner_state = init_pmap(base_rngs)
        runner_state = restore_runner_state(checkpoint, template_runner_state, devices)
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
        progress = compute_curriculum_progress(
            update, num_updates, env_config.curriculum
        )
        progress_array = jnp.full((len(devices),), progress, dtype=jnp.float32)
        runner_state, metrics = step_pmap(runner_state, progress_array)
        global_step = (update + 1) * rl_config.timesteps_per_update
        current_lr = float(learning_rate_for_update(rl_config, update))

        for key, value in unreplicate_metrics(metrics).items():
            writer.add_scalar(key, value, global_step)
        writer.add_scalar("optimizer/learning_rate", current_lr, global_step)
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
