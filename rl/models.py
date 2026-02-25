import functools
from typing import Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(
        self, carry: jax.Array, x: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=rnn_state.shape[-1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int) -> jax.Array:
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.key(0), (batch_size, hidden_size)
        )


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    hidden_size: int = 128

    @nn.compact
    def __call__(
        self, hidden: jax.Array, x: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, distrax.MultivariateNormalDiag, jax.Array]:
        obs, dones = x
        embedding = nn.Dense(
            features=self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            features=self.hidden_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            features=self.action_dim[0],
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_log_std = self.param("log_std", constant(0.0), self.action_dim)

        pi = distrax.MultivariateNormalDiag(
            loc=actor_mean, scale_diag=jnp.exp(actor_log_std)
        )

        critic = nn.Dense(
            features=self.hidden_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            features=1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
