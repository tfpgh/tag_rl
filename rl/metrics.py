import jax
import jax.numpy as jnp

from rl.ppo import LossInfo
from rl.rollout import Transition


def safe_rate(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    return jnp.asarray(
        jnp.where(denominator > 0, numerator / denominator, jnp.float32(0))
    )


def compute_training_metrics(
    traj_batch: Transition, chaser_loss: LossInfo, evader_loss: LossInfo
) -> dict[str, jax.Array]:
    tagged = traj_batch.info.tagged.astype(jnp.float32)
    time_up = traj_batch.info.time_up.astype(jnp.float32)
    chaser_collision = traj_batch.info.chaser_collision.astype(jnp.float32)
    evader_collision = traj_batch.info.evader_collision.astype(jnp.float32)
    finished = (
        traj_batch.info.tagged
        | traj_batch.info.time_up
        | traj_batch.info.chaser_collision
        | traj_batch.info.evader_collision
    )
    n_finished = finished.astype(jnp.float32).sum()
    finished_f32 = finished.astype(jnp.float32)

    mean_episode_length = safe_rate(
        (traj_batch.info.step_count * finished_f32).sum(),
        n_finished,
    )
    tag_rate = safe_rate(tagged.sum(), n_finished)
    time_up_rate = safe_rate(time_up.sum(), n_finished)
    chaser_collision_termination_rate = safe_rate(chaser_collision.sum(), n_finished)
    evader_collision_termination_rate = safe_rate(evader_collision.sum(), n_finished)

    return {
        "chaser/total_loss": jnp.asarray(chaser_loss[0].mean()),
        "chaser/value_loss": jnp.asarray(chaser_loss[1][0].mean()),
        "chaser/actor_loss": jnp.asarray(chaser_loss[1][1].mean()),
        "chaser/entropy": jnp.asarray(chaser_loss[1][2].mean()),
        "evader/total_loss": jnp.asarray(evader_loss[0].mean()),
        "evader/value_loss": jnp.asarray(evader_loss[1][0].mean()),
        "evader/actor_loss": jnp.asarray(evader_loss[1][1].mean()),
        "evader/entropy": jnp.asarray(evader_loss[1][2].mean()),
        "env/tag_rate": tag_rate,
        "env/time_up_rate": time_up_rate,
        "env/mean_distance": traj_batch.info.distance.mean(),
        "env/mean_episode_length": mean_episode_length,
        "chaser/mean_reward": traj_batch.chaser_reward.mean(),
        "evader/mean_reward": traj_batch.evader_reward.mean(),
        "chaser/mean_action_magnitude": jnp.abs(traj_batch.chaser_action).mean(),
        "evader/mean_action_magnitude": jnp.abs(traj_batch.evader_action).mean(),
        "chaser/collision_rate": chaser_collision.mean(),
        "evader/collision_rate": evader_collision.mean(),
        "env/termination_by_chaser_collision_rate": chaser_collision_termination_rate,
        "env/termination_by_evader_collision_rate": evader_collision_termination_rate,
    }
