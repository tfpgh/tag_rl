from collections.abc import Sequence
from typing import Any, NamedTuple

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from rl.rollout import RunnerState

AXIS_NAME = "env"


class TrainingShardings(NamedTuple):
    mesh: Mesh
    replicated: NamedSharding
    data: NamedSharding


def make_shardings(devices: Sequence[Any]) -> TrainingShardings:
    mesh = Mesh(np.asarray(devices, dtype=object), (AXIS_NAME,))
    return TrainingShardings(
        mesh=mesh,
        replicated=NamedSharding(mesh, P()),
        data=NamedSharding(mesh, P(AXIS_NAME)),
    )


def shard_env_state(env_state: Any, shardings: TrainingShardings) -> Any:
    return jax.tree.map(lambda x: jax.device_put(x, shardings.data), env_state)


def shard_runner_state(
    runner_state: RunnerState, shardings: TrainingShardings
) -> RunnerState:
    return RunnerState(
        chaser_ts=jax.device_put(runner_state.chaser_ts, shardings.replicated),
        evader_ts=jax.device_put(runner_state.evader_ts, shardings.replicated),
        env_state=shard_env_state(runner_state.env_state, shardings),
        chaser_obs=jax.device_put(runner_state.chaser_obs, shardings.data),
        evader_obs=jax.device_put(runner_state.evader_obs, shardings.data),
        prev_done=jax.device_put(runner_state.prev_done, shardings.data),
        chaser_hstate=jax.device_put(runner_state.chaser_hstate, shardings.data),
        evader_hstate=jax.device_put(runner_state.evader_hstate, shardings.data),
        rng=jax.device_put(runner_state.rng, shardings.replicated),
    )
