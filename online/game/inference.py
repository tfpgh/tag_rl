from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from runtime import jax_setup as _jax_setup  # noqa: F401

import jax
import jax.numpy as jnp

from environment.config import EnvironmentConfig
from rl.checkpoints import load_checkpoint_configs, load_policy_params
from rl.models import ScannedRNN
from rl.policy import build_policies


@dataclass(slots=True)
class DualPolicyOutputs:
    chaser_action: jax.Array
    evader_action: jax.Array


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
        obs_size = 2 * self.env_config.n_rays + 1
        zero_obs = jnp.zeros((obs_size,), dtype=jnp.float32)
        chaser_hstate, evader_hstate, _, _ = self._step_fn(
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
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        chaser_input = (
            jnp.asarray(chaser_obs, dtype=jnp.float32)[None, None, :],
            jnp.zeros((1, 1), dtype=bool),
        )
        evader_input = (
            jnp.asarray(evader_obs, dtype=jnp.float32)[None, None, :],
            jnp.zeros((1, 1), dtype=bool),
        )
        next_chaser_hstate, chaser_pi, _ = self.chaser_policy.apply(
            chaser_params, chaser_hstate, chaser_input
        )
        next_evader_hstate, evader_pi, _ = self.evader_policy.apply(
            evader_params, evader_hstate, evader_input
        )
        return (
            next_chaser_hstate,
            next_evader_hstate,
            jnp.clip(chaser_pi.loc[0, 0], -1.0, 1.0),
            jnp.clip(evader_pi.loc[0, 0], -1.0, 1.0),
        )

    def step(self, chaser_obs: jax.Array, evader_obs: jax.Array) -> DualPolicyOutputs:
        (
            self.chaser_hstate,
            self.evader_hstate,
            chaser_action,
            evader_action,
        ) = self._step_fn(
            self.chaser_params,
            self.evader_params,
            self.chaser_hstate,
            self.evader_hstate,
            jnp.asarray(chaser_obs, dtype=jnp.float32),
            jnp.asarray(evader_obs, dtype=jnp.float32),
        )
        return DualPolicyOutputs(
            chaser_action=chaser_action,
            evader_action=evader_action,
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
