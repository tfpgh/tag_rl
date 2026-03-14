from typing import Any

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from rl.config import RLConfig
from rl.rollout import AgentTransition

ActorCriticModule = Any
LossBreakdown = tuple[jax.Array, jax.Array, jax.Array]
LossInfo = tuple[jax.Array, LossBreakdown]
AdvantageCarry = tuple[jax.Array, jax.Array, jax.Array]
UpdateState = tuple[
    TrainState,
    jax.Array,
    AgentTransition,
    jax.Array,
    jax.Array,
    jax.Array,
]


def calculate_gae(
    rl_config: RLConfig,
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    last_val: jax.Array,
    last_done: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    def _get_advantages(
        carry: AdvantageCarry, transition: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[AdvantageCarry, jax.Array]:
        gae, next_value, next_done = carry
        done, value, reward = transition
        delta = reward + rl_config.gamma * next_value * (1 - next_done) - value
        gae = delta + rl_config.gamma * rl_config.gae_lambda * (1 - next_done) * gae
        return (gae, value, done), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val, last_done),
        (dones, values, rewards),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values


def update_agent(
    rl_config: RLConfig,
    network: ActorCriticModule,
    train_state: TrainState,
    init_hstate: jax.Array,
    agent_traj: AgentTransition,
    advantages: jax.Array,
    targets: jax.Array,
    rng: jax.Array,
    axis_name: str | None = None,
) -> tuple[TrainState, LossInfo]:
    def _update_epoch(
        update_state: UpdateState, _: None
    ) -> tuple[UpdateState, LossInfo]:
        def _update_minibatch(
            train_state: TrainState,
            batch_info: tuple[jax.Array, AgentTransition, jax.Array, jax.Array],
        ) -> tuple[TrainState, LossInfo]:
            init_hstate, traj_batch, advantages, targets = batch_info

            def _loss_fn(
                params: Any,
                init_hstate: jax.Array,
                traj_batch: AgentTransition,
                gae: jax.Array,
                targets: jax.Array,
            ) -> tuple[jax.Array, LossBreakdown]:
                _, pi, value = network.apply(
                    params,
                    init_hstate[0],
                    (traj_batch.obs, traj_batch.done),
                )
                log_prob = pi.log_prob(traj_batch.action)

                value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                    -rl_config.clip_eps, rl_config.clip_eps
                )
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss = (
                    0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                )

                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                ratio = jnp.exp(log_prob - traj_batch.log_prob)
                loss_actor1 = ratio * gae
                loss_actor2 = (
                    jnp.clip(
                        ratio,
                        1.0 - rl_config.clip_eps,
                        1.0 + rl_config.clip_eps,
                    )
                    * gae
                )
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                entropy = pi.entropy().mean()

                total_loss = (
                    loss_actor
                    + rl_config.vf_coef * value_loss
                    - rl_config.ent_coef * entropy
                )
                return total_loss, (value_loss, loss_actor, entropy)

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params,
                init_hstate,
                traj_batch,
                advantages,
                targets,
            )
            if axis_name is not None:
                grads = jax.lax.pmean(grads, axis_name=axis_name)
                total_loss = jax.tree.map(
                    lambda x: jax.lax.pmean(x, axis_name=axis_name), total_loss
                )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, total_loss

        train_state, init_hstate, traj_batch, advantages, targets, rng = update_state

        rng, _rng = jax.random.split(rng)
        permutation = jax.random.permutation(_rng, traj_batch.done.shape[1])
        batch = (init_hstate, traj_batch, advantages, targets)

        shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)

        minibatches = jax.tree.map(
            lambda x: jnp.swapaxes(
                jnp.reshape(
                    x,
                    [x.shape[0], rl_config.num_minibatches, -1] + list(x.shape[2:]),
                ),
                1,
                0,
            ),
            shuffled_batch,
        )

        train_state, total_loss = jax.lax.scan(
            _update_minibatch, train_state, minibatches
        )
        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        return update_state, total_loss

    update_state = (
        train_state,
        init_hstate,
        agent_traj,
        advantages,
        targets,
        rng,
    )
    update_state, loss_info = jax.lax.scan(
        _update_epoch, update_state, None, rl_config.update_epochs
    )
    return update_state[0], loss_info
