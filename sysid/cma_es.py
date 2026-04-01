from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms.distribution_based.cma_es import CMA_ES


@dataclass(slots=True)
class CMAESState:
    key: jax.Array
    params: object
    strategy: CMA_ES
    state: object
    generation: int


class CMAES:
    def __init__(
        self, dim: int, population_size: int, seed: int, sigma: float = 1.0
    ) -> None:
        if population_size < 4:
            raise ValueError("population_size must be at least 4")
        strategy = CMA_ES(population_size=population_size, solution=jnp.zeros((dim,)))
        params = replace(
            strategy.default_params, std_init=jnp.asarray(sigma, dtype=jnp.float32)
        )
        key = jax.random.PRNGKey(seed)
        key, init_key = jax.random.split(key)
        state = strategy.init(init_key, jnp.zeros((dim,), dtype=jnp.float32), params)
        self.state = CMAESState(
            key=key,
            params=params,
            strategy=strategy,
            state=state,
            generation=0,
        )

    def ask(self) -> np.ndarray:
        key, ask_key = jax.random.split(self.state.key)
        population, state = self.state.strategy.ask(
            ask_key, self.state.state, self.state.params
        )
        self.state = CMAESState(
            key=key,
            params=self.state.params,
            strategy=self.state.strategy,
            state=state,
            generation=self.state.generation,
        )
        return np.asarray(jax.device_get(population), dtype=np.float32)

    def tell(self, population: np.ndarray, fitness: np.ndarray) -> None:
        key, tell_key = jax.random.split(self.state.key)
        state, _ = self.state.strategy.tell(
            tell_key,
            jnp.asarray(population, dtype=jnp.float32),
            jnp.asarray(fitness, dtype=jnp.float32),
            self.state.state,
            self.state.params,
        )
        self.state = CMAESState(
            key=key,
            params=self.state.params,
            strategy=self.state.strategy,
            state=state,
            generation=self.state.generation + 1,
        )
