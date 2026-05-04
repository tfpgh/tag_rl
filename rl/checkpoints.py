import os
import pickle
from dataclasses import asdict
from typing import Any, Sequence

import jax

from environment.config import EnvironmentConfig
from rl.config import RLConfig
from rl.rollout import RunnerState


class _LegacyOpaqueTuple(tuple):
    """Stand-in for environment.types NamedTuples whose schema has drifted.

    Stores values positionally; field names are lost. Only safe for fields the
    caller does not introspect — used to load old checkpoints for inference,
    where env_state is never read.
    """

    def __new__(cls, *args: Any) -> "_LegacyOpaqueTuple":
        return tuple.__new__(cls, args)


_LEGACY_PERMISSIVE_TYPES = {
    "AgentDynamicsParams",
    "AgentKinematics",
    "DomainParams",
    "ObstacleState",
    "PipelineParams",
    "TagEnvironmentState",
    "TagEnvironmentStepInfo",
}


class _PermissiveCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "environment.types" and name in _LEGACY_PERMISSIVE_TYPES:
            return _LegacyOpaqueTuple
        return super().find_class(module, name)


def _unreplicate_tree(tree: Any) -> Any:
    return jax.tree.map(lambda x: jax.device_get(x[0]), tree)


def _host_to_sharded_tree(tree: Any, devices: Sequence[Any]) -> Any:
    n_devices = len(devices)
    return jax.tree.map(
        lambda x: jax.device_put_sharded([x[i] for i in range(n_devices)], devices),
        tree,
    )


def _require_device_count(runner_state: dict[str, Any], num_devices: int) -> None:
    sharded_keys = (
        "env_state",
        "chaser_obs",
        "evader_obs",
        "prev_done",
        "chaser_hstate",
        "evader_hstate",
        "rng",
    )
    for key in sharded_keys:
        value = runner_state[key]
        if getattr(value, "shape", (None,))[0] != num_devices:
            raise ValueError(
                f"Checkpoint {key} expects {getattr(value, 'shape', None)} but active device count is {num_devices}"
            )


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

    chaser_train_state = _serialize_train_state(
        _unreplicate_tree(runner_state.chaser_ts)
    )
    evader_train_state = _serialize_train_state(
        _unreplicate_tree(runner_state.evader_ts)
    )
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
    try:
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
    except TypeError as exc:
        # NamedTuple arity mismatch — checkpoint was saved with an older
        # environment.types schema. Retry with a permissive unpickler so policy
        # params still load. env_state becomes opaque and must not be used.
        if "positional argument" not in str(exc):
            raise
        print(
            f"WARNING: {path} contains stale environment.types NamedTuples "
            f"({exc}). Loading with permissive unpickler — env_state will be "
            f"opaque tuples and is not safe to resume training from."
        )
        with open(path, "rb") as f:
            checkpoint = _PermissiveCheckpointUnpickler(f).load()
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {path} is not a dictionary payload")
    return checkpoint


def restore_runner_state(
    checkpoint: dict[str, Any],
    template_runner_state: RunnerState,
    devices: Sequence[Any],
) -> RunnerState:
    runner_state = checkpoint.get("runner_state")
    if not isinstance(runner_state, dict):
        raise ValueError(
            "Checkpoint does not contain full runner state; it cannot be resumed for training"
        )
    _require_device_count(runner_state, len(devices))

    try:
        restored = RunnerState(
            chaser_ts=_restore_train_state(
                _unreplicate_tree(template_runner_state.chaser_ts),
                runner_state["chaser_train_state"],
            ),
            evader_ts=_restore_train_state(
                _unreplicate_tree(template_runner_state.evader_ts),
                runner_state["evader_train_state"],
            ),
            chaser_obs=_host_to_sharded_tree(runner_state["chaser_obs"], devices),
            evader_obs=_host_to_sharded_tree(runner_state["evader_obs"], devices),
            env_state=_host_to_sharded_tree(runner_state["env_state"], devices),
            prev_done=_host_to_sharded_tree(runner_state["prev_done"], devices),
            chaser_hstate=_host_to_sharded_tree(runner_state["chaser_hstate"], devices),
            evader_hstate=_host_to_sharded_tree(runner_state["evader_hstate"], devices),
            rng=_host_to_sharded_tree(runner_state["rng"], devices),
        )
    except KeyError as exc:
        raise ValueError("Checkpoint runner_state is missing required fields") from exc
    return RunnerState(
        chaser_ts=jax.device_put_replicated(restored.chaser_ts, devices),
        evader_ts=jax.device_put_replicated(restored.evader_ts, devices),
        env_state=restored.env_state,
        chaser_obs=restored.chaser_obs,
        evader_obs=restored.evader_obs,
        prev_done=restored.prev_done,
        chaser_hstate=restored.chaser_hstate,
        evader_hstate=restored.evader_hstate,
        rng=restored.rng,
    )


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
    return RLConfig(**rl_config_data), EnvironmentConfig.from_dict(env_config_data)
