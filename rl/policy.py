from typing import Any

from environment.config import EnvironmentConfig
from rl.models import ActorCriticPolicy

PolicyModule = Any


def build_policies(
    hidden_size: int, env_config: EnvironmentConfig
) -> tuple[ActorCriticPolicy, ActorCriticPolicy]:
    return (
        ActorCriticPolicy(
            action_dim=(2,),
            hidden_size=hidden_size,
            n_rays=env_config.n_rays,
        ),
        ActorCriticPolicy(
            action_dim=(2,),
            hidden_size=hidden_size,
            n_rays=env_config.n_rays,
        ),
    )
