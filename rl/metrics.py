import jax
import jax.numpy as jnp
from typing import cast

from rl.ppo import LossInfo
from rl.rollout import Transition


def compute_training_metrics(
    traj_batch: Transition, chaser_loss: LossInfo, evader_loss: LossInfo
) -> dict[str, jax.Array]:
    finished = traj_batch.info.tagged | traj_batch.info.time_up
    n_finished = finished.astype(jnp.float32).sum()
    mean_ep_length = jnp.where(
        n_finished > 0,
        (traj_batch.info.step_count * finished.astype(jnp.float32)).sum() / n_finished,
        jnp.float32(0),
    )
    tag_rate = jnp.where(
        n_finished > 0,
        traj_batch.info.tagged.astype(jnp.float32).sum() / n_finished,
        jnp.float32(0),
    )

    metrics = cast(
        dict[str, jax.Array],
        {
            "chaser/total_loss": jnp.asarray(chaser_loss[0].mean()),
            "chaser/value_loss": jnp.asarray(chaser_loss[1][0].mean()),
            "chaser/actor_loss": jnp.asarray(chaser_loss[1][1].mean()),
            "chaser/entropy": jnp.asarray(chaser_loss[1][2].mean()),
            "evader/total_loss": jnp.asarray(evader_loss[0].mean()),
            "evader/value_loss": jnp.asarray(evader_loss[1][0].mean()),
            "evader/actor_loss": jnp.asarray(evader_loss[1][1].mean()),
            "evader/entropy": jnp.asarray(evader_loss[1][2].mean()),
            "env/tag_rate": tag_rate,
            "env/mean_distance": traj_batch.info.distance.mean(),
            "env/mean_episode_length": mean_ep_length,
            "chaser/mean_reward": traj_batch.chaser_reward.mean(),
            "evader/mean_reward": traj_batch.evader_reward.mean(),
            "chaser/mean_action_magnitude": jnp.abs(traj_batch.chaser_action).mean(),
            "evader/mean_action_magnitude": jnp.abs(traj_batch.evader_action).mean(),
            "chaser/collision_rate": traj_batch.info.chaser_collision.astype(
                jnp.float32
            ).mean(),
            "evader/collision_rate": traj_batch.info.evader_collision.astype(
                jnp.float32
            ).mean(),
        },
    )
    return metrics
