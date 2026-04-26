import jax
import jax.numpy as jnp

from rl.ppo import LossInfo
from rl.rollout import Transition


def safe_rate(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    return jnp.asarray(
        jnp.where(denominator > 0, numerator / denominator, jnp.float32(0))
    )


def safe_std(sum_: jax.Array, sum_squares: jax.Array, count: jax.Array) -> jax.Array:
    mean = safe_rate(sum_, count)
    second_moment = safe_rate(sum_squares, count)
    variance = jnp.maximum(second_moment - jnp.square(mean), jnp.float32(0))
    return jnp.sqrt(variance)


def compute_training_metrics(
    traj_batch: Transition,
    chaser_loss: LossInfo,
    evader_loss: LossInfo,
    axis_name: str | None = None,
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
    chaser_action_clipped = jnp.not_equal(
        traj_batch.chaser_action, traj_batch.chaser_action_raw
    ).astype(jnp.float32)
    evader_action_clipped = jnp.not_equal(
        traj_batch.evader_action, traj_batch.evader_action_raw
    ).astype(jnp.float32)
    chaser_action_any_clipped = jnp.any(chaser_action_clipped > 0, axis=-1).astype(
        jnp.float32
    )
    evader_action_any_clipped = jnp.any(evader_action_clipped > 0, axis=-1).astype(
        jnp.float32
    )
    chaser_forward_velocity = traj_batch.info.chaser_forward_velocity.astype(jnp.float32)
    chaser_angular_velocity = traj_batch.info.chaser_angular_velocity.astype(jnp.float32)
    evader_forward_velocity = traj_batch.info.evader_forward_velocity.astype(jnp.float32)
    evader_angular_velocity = traj_batch.info.evader_angular_velocity.astype(jnp.float32)
    action_delay_substeps = traj_batch.info.action_delay_substeps.astype(jnp.float32)
    observation_delay_substeps = traj_batch.info.observation_delay_substeps.astype(
        jnp.float32
    )
    obstacle_count = traj_batch.info.obstacle_count.astype(jnp.float32)
    action_drop_probability = traj_batch.info.action_drop_probability.astype(
        jnp.float32
    )
    frame_drop_probability = traj_batch.info.frame_drop_probability.astype(jnp.float32)
    stale_observation_probability = (
        traj_batch.info.stale_observation_probability.astype(jnp.float32)
    )
    position_noise_std = traj_batch.info.position_noise_std.astype(jnp.float32)
    yaw_noise_std = traj_batch.info.yaw_noise_std.astype(jnp.float32)
    chaser_motor_balance = traj_batch.info.chaser_motor_balance.astype(jnp.float32)
    evader_motor_balance = traj_batch.info.evader_motor_balance.astype(jnp.float32)
    chaser_motor_strength_scale = traj_batch.info.chaser_motor_strength_scale.astype(
        jnp.float32
    )
    evader_motor_strength_scale = traj_batch.info.evader_motor_strength_scale.astype(
        jnp.float32
    )
    randomization_count = jnp.asarray(traj_batch.info.distance.size, dtype=jnp.float32)

    metric_sums = {
        "chaser_total_loss": chaser_loss[0].sum(),
        "chaser_total_loss_count": jnp.asarray(chaser_loss[0].size, dtype=jnp.float32),
        "chaser_value_loss": chaser_loss[1][0].sum(),
        "chaser_value_loss_count": jnp.asarray(
            chaser_loss[1][0].size, dtype=jnp.float32
        ),
        "chaser_actor_loss": chaser_loss[1][1].sum(),
        "chaser_actor_loss_count": jnp.asarray(
            chaser_loss[1][1].size, dtype=jnp.float32
        ),
        "chaser_entropy": chaser_loss[1][2].sum(),
        "chaser_entropy_count": jnp.asarray(chaser_loss[1][2].size, dtype=jnp.float32),
        "evader_total_loss": evader_loss[0].sum(),
        "evader_total_loss_count": jnp.asarray(evader_loss[0].size, dtype=jnp.float32),
        "evader_value_loss": evader_loss[1][0].sum(),
        "evader_value_loss_count": jnp.asarray(
            evader_loss[1][0].size, dtype=jnp.float32
        ),
        "evader_actor_loss": evader_loss[1][1].sum(),
        "evader_actor_loss_count": jnp.asarray(
            evader_loss[1][1].size, dtype=jnp.float32
        ),
        "evader_entropy": evader_loss[1][2].sum(),
        "evader_entropy_count": jnp.asarray(evader_loss[1][2].size, dtype=jnp.float32),
        "tagged_sum": tagged.sum(),
        "time_up_sum": time_up.sum(),
        "chaser_collision_sum": chaser_collision.sum(),
        "evader_collision_sum": evader_collision.sum(),
        "n_finished": n_finished,
        "episode_length_sum": (traj_batch.info.step_count * finished_f32).sum(),
        "distance_sum": traj_batch.info.distance.sum(),
        "distance_count": randomization_count,
        "chaser_reward_sum": traj_batch.chaser_reward.sum(),
        "chaser_reward_count": jnp.asarray(
            traj_batch.chaser_reward.size, dtype=jnp.float32
        ),
        "evader_reward_sum": traj_batch.evader_reward.sum(),
        "evader_reward_count": jnp.asarray(
            traj_batch.evader_reward.size, dtype=jnp.float32
        ),
        "chaser_action_mag_sum": jnp.abs(traj_batch.chaser_action).sum(),
        "chaser_action_mag_count": jnp.asarray(
            traj_batch.chaser_action.size, dtype=jnp.float32
        ),
        "chaser_action_left_sum": traj_batch.chaser_action[..., 0].sum(),
        "chaser_action_left_count": jnp.asarray(
            traj_batch.chaser_action[..., 0].size, dtype=jnp.float32
        ),
        "chaser_action_right_sum": traj_batch.chaser_action[..., 1].sum(),
        "chaser_action_right_count": jnp.asarray(
            traj_batch.chaser_action[..., 1].size, dtype=jnp.float32
        ),
        "chaser_action_clipped_sum": chaser_action_clipped.sum(),
        "chaser_action_clipped_count": jnp.asarray(
            chaser_action_clipped.size, dtype=jnp.float32
        ),
        "chaser_action_any_clipped_sum": chaser_action_any_clipped.sum(),
        "chaser_action_any_clipped_count": jnp.asarray(
            chaser_action_any_clipped.size, dtype=jnp.float32
        ),
        "evader_action_mag_sum": jnp.abs(traj_batch.evader_action).sum(),
        "evader_action_mag_count": jnp.asarray(
            traj_batch.evader_action.size, dtype=jnp.float32
        ),
        "evader_action_left_sum": traj_batch.evader_action[..., 0].sum(),
        "evader_action_left_count": jnp.asarray(
            traj_batch.evader_action[..., 0].size, dtype=jnp.float32
        ),
        "evader_action_right_sum": traj_batch.evader_action[..., 1].sum(),
        "evader_action_right_count": jnp.asarray(
            traj_batch.evader_action[..., 1].size, dtype=jnp.float32
        ),
        "evader_action_clipped_sum": evader_action_clipped.sum(),
        "evader_action_clipped_count": jnp.asarray(
            evader_action_clipped.size, dtype=jnp.float32
        ),
        "evader_action_any_clipped_sum": evader_action_any_clipped.sum(),
        "evader_action_any_clipped_count": jnp.asarray(
            evader_action_any_clipped.size, dtype=jnp.float32
        ),
        "chaser_forward_velocity_sum": chaser_forward_velocity.sum(),
        "chaser_forward_velocity_count": jnp.asarray(
            chaser_forward_velocity.size, dtype=jnp.float32
        ),
        "chaser_angular_velocity_sum": chaser_angular_velocity.sum(),
        "chaser_angular_velocity_count": jnp.asarray(
            chaser_angular_velocity.size, dtype=jnp.float32
        ),
        "evader_forward_velocity_sum": evader_forward_velocity.sum(),
        "evader_forward_velocity_count": jnp.asarray(
            evader_forward_velocity.size, dtype=jnp.float32
        ),
        "evader_angular_velocity_sum": evader_angular_velocity.sum(),
        "evader_angular_velocity_count": jnp.asarray(
            evader_angular_velocity.size, dtype=jnp.float32
        ),
        "chaser_collision_term_sum": chaser_collision.sum(),
        "evader_collision_term_sum": evader_collision.sum(),
        "action_delay_substeps_sum": action_delay_substeps.sum(),
        "action_delay_substeps_sq_sum": jnp.square(action_delay_substeps).sum(),
        "observation_delay_substeps_sum": observation_delay_substeps.sum(),
        "observation_delay_substeps_sq_sum": jnp.square(
            observation_delay_substeps
        ).sum(),
        "obstacle_count_sum": obstacle_count.sum(),
        "obstacle_count_sq_sum": jnp.square(obstacle_count).sum(),
        "action_drop_probability_sum": action_drop_probability.sum(),
        "action_drop_probability_sq_sum": jnp.square(action_drop_probability).sum(),
        "frame_drop_probability_sum": frame_drop_probability.sum(),
        "frame_drop_probability_sq_sum": jnp.square(frame_drop_probability).sum(),
        "stale_observation_probability_sum": stale_observation_probability.sum(),
        "stale_observation_probability_sq_sum": jnp.square(
            stale_observation_probability
        ).sum(),
        "position_noise_std_sum": position_noise_std.sum(),
        "position_noise_std_sq_sum": jnp.square(position_noise_std).sum(),
        "yaw_noise_std_sum": yaw_noise_std.sum(),
        "yaw_noise_std_sq_sum": jnp.square(yaw_noise_std).sum(),
        "chaser_motor_balance_sum": chaser_motor_balance.sum(),
        "chaser_motor_balance_sq_sum": jnp.square(chaser_motor_balance).sum(),
        "chaser_motor_balance_abs_sum": jnp.abs(chaser_motor_balance).sum(),
        "evader_motor_balance_sum": evader_motor_balance.sum(),
        "evader_motor_balance_sq_sum": jnp.square(evader_motor_balance).sum(),
        "evader_motor_balance_abs_sum": jnp.abs(evader_motor_balance).sum(),
        "chaser_motor_strength_scale_sum": chaser_motor_strength_scale.sum(),
        "chaser_motor_strength_scale_sq_sum": jnp.square(
            chaser_motor_strength_scale
        ).sum(),
        "chaser_motor_strength_scale_abs_dev_sum": jnp.abs(
            chaser_motor_strength_scale - 1.0
        ).sum(),
        "evader_motor_strength_scale_sum": evader_motor_strength_scale.sum(),
        "evader_motor_strength_scale_sq_sum": jnp.square(
            evader_motor_strength_scale
        ).sum(),
        "evader_motor_strength_scale_abs_dev_sum": jnp.abs(
            evader_motor_strength_scale - 1.0
        ).sum(),
    }

    if axis_name is not None:
        metric_sums = jax.lax.psum(metric_sums, axis_name=axis_name)

    mean_episode_length = safe_rate(
        metric_sums["episode_length_sum"],
        metric_sums["n_finished"],
    )
    tag_rate = safe_rate(metric_sums["tagged_sum"], metric_sums["n_finished"])
    time_up_rate = safe_rate(metric_sums["time_up_sum"], metric_sums["n_finished"])
    chaser_collision_termination_rate = safe_rate(
        metric_sums["chaser_collision_term_sum"], metric_sums["n_finished"]
    )
    evader_collision_termination_rate = safe_rate(
        metric_sums["evader_collision_term_sum"], metric_sums["n_finished"]
    )

    return {
        "chaser/total_loss": safe_rate(
            metric_sums["chaser_total_loss"], metric_sums["chaser_total_loss_count"]
        ),
        "chaser/value_loss": safe_rate(
            metric_sums["chaser_value_loss"], metric_sums["chaser_value_loss_count"]
        ),
        "chaser/actor_loss": safe_rate(
            metric_sums["chaser_actor_loss"], metric_sums["chaser_actor_loss_count"]
        ),
        "chaser/entropy": safe_rate(
            metric_sums["chaser_entropy"], metric_sums["chaser_entropy_count"]
        ),
        "evader/total_loss": safe_rate(
            metric_sums["evader_total_loss"], metric_sums["evader_total_loss_count"]
        ),
        "evader/value_loss": safe_rate(
            metric_sums["evader_value_loss"], metric_sums["evader_value_loss_count"]
        ),
        "evader/actor_loss": safe_rate(
            metric_sums["evader_actor_loss"], metric_sums["evader_actor_loss_count"]
        ),
        "evader/entropy": safe_rate(
            metric_sums["evader_entropy"], metric_sums["evader_entropy_count"]
        ),
        "env/tag_rate": tag_rate,
        "env/time_up_rate": time_up_rate,
        "env/mean_distance": safe_rate(
            metric_sums["distance_sum"], metric_sums["distance_count"]
        ),
        "env/mean_episode_length": mean_episode_length,
        "chaser/mean_reward": safe_rate(
            metric_sums["chaser_reward_sum"], metric_sums["chaser_reward_count"]
        ),
        "evader/mean_reward": safe_rate(
            metric_sums["evader_reward_sum"], metric_sums["evader_reward_count"]
        ),
        "chaser/mean_action_magnitude": safe_rate(
            metric_sums["chaser_action_mag_sum"], metric_sums["chaser_action_mag_count"]
        ),
        "chaser/mean_left_action": safe_rate(
            metric_sums["chaser_action_left_sum"],
            metric_sums["chaser_action_left_count"],
        ),
        "chaser/mean_right_action": safe_rate(
            metric_sums["chaser_action_right_sum"],
            metric_sums["chaser_action_right_count"],
        ),
        "chaser/mean_forward_velocity": safe_rate(
            metric_sums["chaser_forward_velocity_sum"],
            metric_sums["chaser_forward_velocity_count"],
        ),
        "chaser/mean_angular_velocity": safe_rate(
            metric_sums["chaser_angular_velocity_sum"],
            metric_sums["chaser_angular_velocity_count"],
        ),
        "chaser/action_clipped_fraction": safe_rate(
            metric_sums["chaser_action_clipped_sum"],
            metric_sums["chaser_action_clipped_count"],
        ),
        "chaser/action_any_clipped_fraction": safe_rate(
            metric_sums["chaser_action_any_clipped_sum"],
            metric_sums["chaser_action_any_clipped_count"],
        ),
        "evader/mean_action_magnitude": safe_rate(
            metric_sums["evader_action_mag_sum"], metric_sums["evader_action_mag_count"]
        ),
        "evader/mean_left_action": safe_rate(
            metric_sums["evader_action_left_sum"],
            metric_sums["evader_action_left_count"],
        ),
        "evader/mean_right_action": safe_rate(
            metric_sums["evader_action_right_sum"],
            metric_sums["evader_action_right_count"],
        ),
        "evader/mean_forward_velocity": safe_rate(
            metric_sums["evader_forward_velocity_sum"],
            metric_sums["evader_forward_velocity_count"],
        ),
        "evader/mean_angular_velocity": safe_rate(
            metric_sums["evader_angular_velocity_sum"],
            metric_sums["evader_angular_velocity_count"],
        ),
        "evader/action_clipped_fraction": safe_rate(
            metric_sums["evader_action_clipped_sum"],
            metric_sums["evader_action_clipped_count"],
        ),
        "evader/action_any_clipped_fraction": safe_rate(
            metric_sums["evader_action_any_clipped_sum"],
            metric_sums["evader_action_any_clipped_count"],
        ),
        "chaser/collision_rate": safe_rate(
            metric_sums["chaser_collision_sum"], metric_sums["distance_count"]
        ),
        "evader/collision_rate": safe_rate(
            metric_sums["evader_collision_sum"], metric_sums["distance_count"]
        ),
        "env/termination_by_chaser_collision_rate": chaser_collision_termination_rate,
        "env/termination_by_evader_collision_rate": evader_collision_termination_rate,
        "randomization/action_delay_substeps": safe_rate(
            metric_sums["action_delay_substeps_sum"], metric_sums["distance_count"]
        ),
        "randomization/action_delay_substeps_std": safe_std(
            metric_sums["action_delay_substeps_sum"],
            metric_sums["action_delay_substeps_sq_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/observation_delay_substeps": safe_rate(
            metric_sums["observation_delay_substeps_sum"], metric_sums["distance_count"]
        ),
        "randomization/observation_delay_substeps_std": safe_std(
            metric_sums["observation_delay_substeps_sum"],
            metric_sums["observation_delay_substeps_sq_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/obstacle_count": safe_rate(
            metric_sums["obstacle_count_sum"], metric_sums["distance_count"]
        ),
        "randomization/obstacle_count_std": safe_std(
            metric_sums["obstacle_count_sum"],
            metric_sums["obstacle_count_sq_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/action_drop_probability": safe_rate(
            metric_sums["action_drop_probability_sum"], metric_sums["distance_count"]
        ),
        "randomization/action_drop_probability_std": safe_std(
            metric_sums["action_drop_probability_sum"],
            metric_sums["action_drop_probability_sq_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/frame_drop_probability": safe_rate(
            metric_sums["frame_drop_probability_sum"], metric_sums["distance_count"]
        ),
        "randomization/frame_drop_probability_std": safe_std(
            metric_sums["frame_drop_probability_sum"],
            metric_sums["frame_drop_probability_sq_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/stale_observation_probability": safe_rate(
            metric_sums["stale_observation_probability_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/stale_observation_probability_std": safe_std(
            metric_sums["stale_observation_probability_sum"],
            metric_sums["stale_observation_probability_sq_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/position_noise_std": safe_rate(
            metric_sums["position_noise_std_sum"], metric_sums["distance_count"]
        ),
        "randomization/position_noise_std_std": safe_std(
            metric_sums["position_noise_std_sum"],
            metric_sums["position_noise_std_sq_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/yaw_noise_std": safe_rate(
            metric_sums["yaw_noise_std_sum"], metric_sums["distance_count"]
        ),
        "randomization/yaw_noise_std_std": safe_std(
            metric_sums["yaw_noise_std_sum"],
            metric_sums["yaw_noise_std_sq_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/chaser_motor_balance": safe_rate(
            metric_sums["chaser_motor_balance_sum"], metric_sums["distance_count"]
        ),
        "randomization/chaser_motor_balance_std": safe_std(
            metric_sums["chaser_motor_balance_sum"],
            metric_sums["chaser_motor_balance_sq_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/chaser_motor_balance_abs": safe_rate(
            metric_sums["chaser_motor_balance_abs_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/evader_motor_balance": safe_rate(
            metric_sums["evader_motor_balance_sum"], metric_sums["distance_count"]
        ),
        "randomization/evader_motor_balance_std": safe_std(
            metric_sums["evader_motor_balance_sum"],
            metric_sums["evader_motor_balance_sq_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/evader_motor_balance_abs": safe_rate(
            metric_sums["evader_motor_balance_abs_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/chaser_motor_strength_scale": safe_rate(
            metric_sums["chaser_motor_strength_scale_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/chaser_motor_strength_scale_std": safe_std(
            metric_sums["chaser_motor_strength_scale_sum"],
            metric_sums["chaser_motor_strength_scale_sq_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/chaser_motor_strength_scale_abs_deviation": safe_rate(
            metric_sums["chaser_motor_strength_scale_abs_dev_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/evader_motor_strength_scale": safe_rate(
            metric_sums["evader_motor_strength_scale_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/evader_motor_strength_scale_std": safe_std(
            metric_sums["evader_motor_strength_scale_sum"],
            metric_sums["evader_motor_strength_scale_sq_sum"],
            metric_sums["distance_count"],
        ),
        "randomization/evader_motor_strength_scale_abs_deviation": safe_rate(
            metric_sums["evader_motor_strength_scale_abs_dev_sum"],
            metric_sums["distance_count"],
        ),
    }
