from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from online.config import DemoConfig
from rl.checkpoints import load_checkpoint_configs, load_policy_params
from rl.models import ScannedRNN
from rl.policy import build_policies


@dataclass(slots=True)
class PolicyOutput:
    chaser_action: np.ndarray
    evader_action: np.ndarray
    observation_size: int


class PolicyRunner:
    def __init__(self, config: DemoConfig) -> None:
        chaser_params, evader_params, checkpoint = load_policy_params(
            str(config.checkpoint_path)
        )
        rl_config, env_config = load_checkpoint_configs(checkpoint)
        self.rl_config = rl_config
        self.env_config = env_config
        self.chaser_network, self.evader_network = build_policies(
            rl_config.hidden_size, env_config
        )
        self.chaser_params = chaser_params
        self.evader_params = evader_params
        self.hidden_size = rl_config.hidden_size
        self.reset()
        self._infer = jax.jit(self._infer_impl)

    def reset(self) -> None:
        self._chaser_h = ScannedRNN.initialize_carry(1, self.hidden_size)
        self._evader_h = ScannedRNN.initialize_carry(1, self.hidden_size)
        self._prev_done = jnp.zeros((1, 1), dtype=bool)

    def _infer_impl(
        self,
        chaser_params: Any,
        evader_params: Any,
        chaser_h: jax.Array,
        evader_h: jax.Array,
        prev_done: jax.Array,
        chaser_obs: jax.Array,
        evader_obs: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        chaser_in = (chaser_obs[None, None, :], prev_done)
        next_chaser_h, chaser_pi, _ = self.chaser_network.apply(
            chaser_params, chaser_h, chaser_in
        )
        evader_in = (evader_obs[None, None, :], prev_done)
        next_evader_h, evader_pi, _ = self.evader_network.apply(
            evader_params, evader_h, evader_in
        )
        return next_chaser_h, next_evader_h, chaser_pi.loc[0, 0], evader_pi.loc[0, 0]

    def step(self, chaser_obs: np.ndarray, evader_obs: np.ndarray) -> PolicyOutput:
        (
            self._chaser_h,
            self._evader_h,
            chaser_action,
            evader_action,
        ) = self._infer(
            self.chaser_params,
            self.evader_params,
            self._chaser_h,
            self._evader_h,
            self._prev_done,
            jnp.asarray(chaser_obs),
            jnp.asarray(evader_obs),
        )
        self._prev_done = jnp.zeros_like(self._prev_done)
        return PolicyOutput(
            chaser_action=np.asarray(
                jnp.clip(chaser_action, -1.0, 1.0), dtype=np.float32
            ),
            evader_action=np.asarray(
                jnp.clip(evader_action, -1.0, 1.0), dtype=np.float32
            ),
            observation_size=int(chaser_obs.shape[0]),
        )
