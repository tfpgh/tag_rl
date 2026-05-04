from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from runtime import jax_setup as _jax_setup  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from environment.config import EnvironmentConfig
from environment.environment import observation_size
from rl.checkpoints import load_checkpoint_configs, load_policy_params
from rl.models import ScannedRNN
from rl.policy import build_policies


@dataclass(slots=True)
class DualPolicyOutputs:
    chaser_action: np.ndarray
    evader_action: np.ndarray
    chaser_value: float
    evader_value: float
    chaser_action_mean: np.ndarray
    evader_action_mean: np.ndarray
    chaser_action_stddev: np.ndarray
    evader_action_stddev: np.ndarray
    chaser_hidden_state: np.ndarray
    evader_hidden_state: np.ndarray


class DualPolicyRunner:
    def __init__(self, checkpoint_path: str) -> None:
        chaser_params, evader_params, checkpoint = load_policy_params(checkpoint_path)
        rl_config, env_config = load_checkpoint_configs(checkpoint)
        self.env_config = env_config
        self.hidden_size = rl_config.hidden_size
        self.chaser_policy, self.evader_policy = build_policies(
            rl_config.hidden_size, env_config
        )
        self.chaser_params = jax.device_put(chaser_params)
        self.evader_params = jax.device_put(evader_params)
        self._step_fn = jax.jit(self._step_impl)
        self.reset()
        self.warmup()

    def reset(self) -> None:
        self.chaser_hstate = ScannedRNN.initialize_carry(1, self.hidden_size)
        self.evader_hstate = ScannedRNN.initialize_carry(1, self.hidden_size)

    def warmup(self) -> None:
        obs_size = observation_size(self.env_config)
        zero_obs = jnp.zeros((obs_size,), dtype=jnp.float32)
        chaser_hstate, evader_hstate, *_ = self._step_fn(
            self.chaser_params,
            self.evader_params,
            self.chaser_hstate,
            self.evader_hstate,
            zero_obs,
            zero_obs,
        )
        self.chaser_hstate = chaser_hstate
        self.evader_hstate = evader_hstate
        self.reset()

    def _step_impl(
        self,
        chaser_params: Any,
        evader_params: Any,
        chaser_hstate: jax.Array,
        evader_hstate: jax.Array,
        chaser_obs: jax.Array,
        evader_obs: jax.Array,
    ) -> tuple[
        jax.Array, jax.Array,
        jax.Array, jax.Array,
        jax.Array, jax.Array,
        jax.Array, jax.Array,
        jax.Array, jax.Array,
    ]:
        chaser_input = (
            jnp.asarray(chaser_obs, dtype=jnp.float32)[None, None, :],
            jnp.zeros((1, 1), dtype=bool),
        )
        evader_input = (
            jnp.asarray(evader_obs, dtype=jnp.float32)[None, None, :],
            jnp.zeros((1, 1), dtype=bool),
        )
        next_chaser_hstate, chaser_pi, chaser_value = self.chaser_policy.apply(
            chaser_params, chaser_hstate, chaser_input
        )
        next_evader_hstate, evader_pi, evader_value = self.evader_policy.apply(
            evader_params, evader_hstate, evader_input
        )
        chaser_mean = chaser_pi.loc[0, 0]
        evader_mean = evader_pi.loc[0, 0]
        # scale_diag may be stored without batch dims; broadcast then index to (action_dim,).
        chaser_stddev = jnp.broadcast_to(chaser_pi.scale_diag, chaser_pi.loc.shape)[0, 0]
        evader_stddev = jnp.broadcast_to(evader_pi.scale_diag, evader_pi.loc.shape)[0, 0]
        return (
            next_chaser_hstate,
            next_evader_hstate,
            jnp.clip(chaser_mean, -1.0, 1.0),
            jnp.clip(evader_mean, -1.0, 1.0),
            chaser_value[0, 0],
            evader_value[0, 0],
            chaser_mean,
            evader_mean,
            chaser_stddev,
            evader_stddev,
        )

    def step(self, chaser_obs: jax.Array, evader_obs: jax.Array) -> DualPolicyOutputs:
        (
            self.chaser_hstate,
            self.evader_hstate,
            chaser_action,
            evader_action,
            chaser_value,
            evader_value,
            chaser_mean,
            evader_mean,
            chaser_stddev,
            evader_stddev,
        ) = self._step_fn(
            self.chaser_params,
            self.evader_params,
            self.chaser_hstate,
            self.evader_hstate,
            jnp.asarray(chaser_obs, dtype=jnp.float32),
            jnp.asarray(evader_obs, dtype=jnp.float32),
        )
        chaser_action_host, evader_action_host = jax.device_get(
            (chaser_action, evader_action)
        )
        chaser_value_host, evader_value_host = jax.device_get(
            (chaser_value, evader_value)
        )
        chaser_mean_host, evader_mean_host = jax.device_get(
            (chaser_mean, evader_mean)
        )
        chaser_stddev_host, evader_stddev_host = jax.device_get(
            (chaser_stddev, evader_stddev)
        )
        chaser_hstate_host, evader_hstate_host = jax.device_get(
            (self.chaser_hstate, self.evader_hstate)
        )
        return DualPolicyOutputs(
            chaser_action=np.asarray(chaser_action_host, dtype=np.float32),
            evader_action=np.asarray(evader_action_host, dtype=np.float32),
            chaser_value=float(chaser_value_host),
            evader_value=float(evader_value_host),
            chaser_action_mean=np.asarray(chaser_mean_host, dtype=np.float32),
            evader_action_mean=np.asarray(evader_mean_host, dtype=np.float32),
            chaser_action_stddev=np.asarray(chaser_stddev_host, dtype=np.float32),
            evader_action_stddev=np.asarray(evader_stddev_host, dtype=np.float32),
            chaser_hidden_state=np.asarray(chaser_hstate_host[0], dtype=np.float32),
            evader_hidden_state=np.asarray(evader_hstate_host[0], dtype=np.float32),
        )


def sync_live_env_config(
    env_config: EnvironmentConfig,
    board_width_m: float,
    board_height_m: float,
    obstacle_size_m: float,
) -> EnvironmentConfig:
    live_env_config = EnvironmentConfig.from_dict(asdict(env_config))
    live_env_config.arena.arena_width = board_width_m
    live_env_config.arena.arena_height = board_height_m
    live_env_config.layout_randomization.obstacle_width = obstacle_size_m
    return live_env_config
