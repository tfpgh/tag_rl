import os
import pickle
from dataclasses import asdict
from typing import Any, Sequence

import jax

from environment.config import EnvironmentConfig
from rl.config import RLConfig
from rl.rollout import RunnerState

CHECKPOINT_VERSION = 2


def _unreplicate_tree(tree: Any) -> Any:
    return jax.tree.map(lambda x: jax.device_get(x[0]), tree)


def _host_to_sharded_tree(tree: Any, devices: Sequence) -> Any:
    n_devices = len(devices)
    return jax.tree.map(
        lambda x: jax.device_put_sharded([x[i] for i in range(n_devices)], devices),
        tree,
    )


def _extract_policy_params(checkpoint: dict[str, Any]) -> tuple[Any, Any]:
    if "chaser_params" in checkpoint and "evader_params" in checkpoint:
        return checkpoint["chaser_params"], checkpoint["evader_params"]
    runner_state = checkpoint.get("runner_state")
    if runner_state is None:
        raise ValueError("Checkpoint does not contain policy parameters")
    return (
        runner_state["chaser_train_state"]["params"],
        runner_state["evader_train_state"]["params"],
    )


def _serialize_train_state(train_state: Any) -> dict[str, Any]:
    return {
        "step": jax.device_get(train_state.step),
        "params": jax.device_get(train_state.params),
        "opt_state": jax.device_get(train_state.opt_state),
    }


def _restore_train_state(template: Any, payload: dict[str, Any]) -> Any:
    return template.replace(
        step=payload["step"],
        params=payload["params"],
        opt_state=payload["opt_state"],
    )


def save_checkpoint(
    runner_state: RunnerState,
    global_step: int,
    checkpoint_dir: str,
    rl_config: RLConfig,
    env_config: EnvironmentConfig,
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"step_{global_step}.pkl")

    chaser_train_state = _serialize_train_state(
        _unreplicate_tree(runner_state.chaser_ts)
    )
    evader_train_state = _serialize_train_state(
        _unreplicate_tree(runner_state.evader_ts)
    )
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "global_step": global_step,
        "num_devices": rl_config.num_devices,
        "model_type": rl_config.model_type,
        "rl_config": asdict(rl_config),
        "env_config": asdict(env_config),
        "chaser_params": chaser_train_state["params"],
        "evader_params": evader_train_state["params"],
        "runner_state": {
            "chaser_train_state": chaser_train_state,
            "evader_train_state": evader_train_state,
            "env_state": jax.device_get(runner_state.env_state),
            "chaser_obs": jax.device_get(runner_state.chaser_obs),
            "evader_obs": jax.device_get(runner_state.evader_obs),
            "done": jax.device_get(runner_state.done),
            "chaser_hstate": jax.device_get(runner_state.chaser_hstate),
            "evader_hstate": jax.device_get(runner_state.evader_hstate),
            "rng": jax.device_get(runner_state.rng),
        },
    }

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Saved checkpoint at step {global_step} to {path}")
    return path


def load_checkpoint(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {path} is not a dictionary payload")
    return checkpoint


def restore_runner_state(
    checkpoint: dict[str, Any],
    template_runner_state: RunnerState,
    devices: Sequence,
) -> RunnerState:
    runner_state = checkpoint.get("runner_state")
    if runner_state is None:
        raise ValueError(
            "Checkpoint does not contain full runner state; it cannot be resumed for training"
        )

    checkpoint_num_devices = checkpoint.get("num_devices")
    actual_num_devices = len(devices)
    if checkpoint_num_devices != actual_num_devices:
        raise ValueError(
            "Checkpoint was saved with "
            f"{checkpoint_num_devices} devices but the current runtime has "
            f"{actual_num_devices}; exact resume requires the same device count"
        )

    template_chaser_train_state = _unreplicate_tree(template_runner_state.chaser_ts)
    template_evader_train_state = _unreplicate_tree(template_runner_state.evader_ts)

    return RunnerState(
        chaser_ts=jax.device_put_replicated(
            _restore_train_state(
                template_chaser_train_state, runner_state["chaser_train_state"]
            ),
            devices,
        ),
        evader_ts=jax.device_put_replicated(
            _restore_train_state(
                template_evader_train_state, runner_state["evader_train_state"]
            ),
            devices,
        ),
        env_state=_host_to_sharded_tree(runner_state["env_state"], devices),
        chaser_obs=_host_to_sharded_tree(runner_state["chaser_obs"], devices),
        evader_obs=_host_to_sharded_tree(runner_state["evader_obs"], devices),
        done=_host_to_sharded_tree(runner_state["done"], devices),
        chaser_hstate=_host_to_sharded_tree(runner_state["chaser_hstate"], devices),
        evader_hstate=_host_to_sharded_tree(runner_state["evader_hstate"], devices),
        rng=_host_to_sharded_tree(runner_state["rng"], devices),
    )


def load_policy_params(path: str) -> tuple[Any, Any, dict[str, Any]]:
    checkpoint = load_checkpoint(path)
    chaser_params, evader_params = _extract_policy_params(checkpoint)
    return chaser_params, evader_params, checkpoint


def load_checkpoint_configs(
    checkpoint: dict[str, Any],
) -> tuple[RLConfig | None, EnvironmentConfig | None]:
    rl_config_data = checkpoint.get("rl_config")
    env_config_data = checkpoint.get("env_config")

    rl_config = RLConfig(**rl_config_data) if rl_config_data is not None else None
    env_config = (
        EnvironmentConfig(**env_config_data) if env_config_data is not None else None
    )
    return rl_config, env_config
