from __future__ import annotations

from dataclasses import dataclass
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
        self.reset()

    def reset(self) -> None:
        self.chaser_hstate = ScannedRNN.initialize_carry(1, self.hidden_size)
        self.evader_hstate = ScannedRNN.initialize_carry(1, self.hidden_size)

    def step(self, chaser_obs: jax.Array, evader_obs: jax.Array) -> DualPolicyOutputs:
        chaser_input = (
            jnp.asarray(chaser_obs, dtype=jnp.float32)[None, None, :],
            jnp.zeros((1, 1), dtype=bool),
        )
        evader_input = (
            jnp.asarray(evader_obs, dtype=jnp.float32)[None, None, :],
            jnp.zeros((1, 1), dtype=bool),
        )
        self.chaser_hstate, chaser_pi, _ = self.chaser_policy.apply(
            self.chaser_params, self.chaser_hstate, chaser_input
        )
        self.evader_hstate, evader_pi, _ = self.evader_policy.apply(
            self.evader_params, self.evader_hstate, evader_input
        )
        return DualPolicyOutputs(
            chaser_action=jnp.clip(chaser_pi.loc[0, 0], -1.0, 1.0),
            evader_action=jnp.clip(evader_pi.loc[0, 0], -1.0, 1.0),
        )


def sync_live_env_config(
    env_config: EnvironmentConfig,
    board_width_m: float,
    board_height_m: float,
    obstacle_size_m: float,
) -> EnvironmentConfig:
    env_config.arena.arena_width = board_width_m
    env_config.arena.arena_height = board_height_m
    env_config.layout_randomization.obstacle_width = obstacle_size_m
    return env_config
