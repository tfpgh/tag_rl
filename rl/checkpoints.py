import os
import pickle
from dataclasses import asdict
from typing import Any

import jax

from environment.config import EnvironmentConfig
from rl.config import RLConfig
from rl.rollout import RunnerState
from rl.sharding import TrainingShardings, shard_runner_state


def _extract_policy_params(checkpoint: dict[str, Any]) -> tuple[Any, Any]:
    try:
        runner_state = checkpoint["runner_state"]
        return (
            runner_state["chaser_train_state"]["params"],
            runner_state["evader_train_state"]["params"],
        )
    except KeyError as exc:
        raise ValueError("Checkpoint does not contain policy parameters") from exc


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

    chaser_train_state = _serialize_train_state(runner_state.chaser_ts)
    evader_train_state = _serialize_train_state(runner_state.evader_ts)
    checkpoint = {
        "global_step": global_step,
        "rl_config": asdict(rl_config),
        "env_config": asdict(env_config),
        "runner_state": {
            "chaser_train_state": chaser_train_state,
            "evader_train_state": evader_train_state,
            "env_state": jax.device_get(runner_state.env_state),
            "chaser_obs": jax.device_get(runner_state.chaser_obs),
            "evader_obs": jax.device_get(runner_state.evader_obs),
            "prev_done": jax.device_get(runner_state.prev_done),
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
    shardings: TrainingShardings,
) -> RunnerState:
    runner_state = checkpoint.get("runner_state")
    if not isinstance(runner_state, dict):
        raise ValueError(
            "Checkpoint does not contain full runner state; it cannot be resumed for training"
        )

    try:
        restored = RunnerState(
            chaser_ts=_restore_train_state(
                template_runner_state.chaser_ts, runner_state["chaser_train_state"]
            ),
            evader_ts=_restore_train_state(
                template_runner_state.evader_ts, runner_state["evader_train_state"]
            ),
            env_state=runner_state["env_state"],
            chaser_obs=runner_state["chaser_obs"],
            evader_obs=runner_state["evader_obs"],
            prev_done=runner_state["prev_done"],
            chaser_hstate=runner_state["chaser_hstate"],
            evader_hstate=runner_state["evader_hstate"],
            rng=runner_state["rng"],
        )
    except KeyError as exc:
        raise ValueError("Checkpoint runner_state is missing required fields") from exc
    return shard_runner_state(restored, shardings)


def load_policy_params(path: str) -> tuple[Any, Any, dict[str, Any]]:
    checkpoint = load_checkpoint(path)
    chaser_params, evader_params = _extract_policy_params(checkpoint)
    return chaser_params, evader_params, checkpoint


def load_checkpoint_configs(
    checkpoint: dict[str, Any],
) -> tuple[RLConfig, EnvironmentConfig]:
    rl_config_data = checkpoint.get("rl_config")
    env_config_data = checkpoint.get("env_config")
    if not isinstance(rl_config_data, dict) or not isinstance(env_config_data, dict):
        raise ValueError(
            "Checkpoint does not contain serialized RL/environment configs"
        )
    return RLConfig(**rl_config_data), EnvironmentConfig(**env_config_data)
