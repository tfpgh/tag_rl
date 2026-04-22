import functools
from typing import Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, normal, variance_scaling


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
            resets[:, None],
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


class RayEncoder(nn.Module):
    """1D CNN for ray sensor data with circular padding."""

    features: int = 32

    @nn.compact
    def __call__(self, rays: jax.Array) -> jax.Array:
        # rays: (..., n_rays, 2) — leading dims are batch
        x = nn.Conv(features=16, kernel_size=(7,), strides=(4,), padding="CIRCULAR")(
            rays
        )
        x = nn.relu(x)  # (..., 64, 16)
        x = nn.Conv(
            features=self.features, kernel_size=(5,), strides=(4,), padding="CIRCULAR"
        )(x)
        x = nn.relu(x)  # (..., 16, 32)
        return x.reshape(*x.shape[:-2], -1)  # flatten → 512


class _ActorCriticBase(nn.Module):
    """Shared actor-critic head. Subclasses implement _embed(obs)."""

    action_dim: Sequence[int]
    hidden_size: int = 128

    def _embed(self, obs: jax.Array) -> jax.Array:
        raise NotImplementedError

    @nn.compact
    def __call__(
        self, hidden: jax.Array, x: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, distrax.MultivariateNormalDiag, jax.Array]:
        obs, dones = x

        embedding = self._embed(obs)
        embedding = nn.Dense(
            features=self.hidden_size,
            kernel_init=variance_scaling(2.0, "fan_in", "truncated_normal"),
            bias_init=constant(0.0),
        )(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            features=self.hidden_size,
            kernel_init=variance_scaling(2.0, "fan_in", "truncated_normal"),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            features=self.action_dim[0],
            kernel_init=normal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_log_std = self.param("log_std", constant(-1.0), self.action_dim)
        actor_log_std = jnp.clip(actor_log_std, -5.0, 0.5)

        pi = distrax.MultivariateNormalDiag(
            loc=actor_mean, scale_diag=jnp.exp(actor_log_std)
        )

        critic = nn.Dense(
            features=self.hidden_size,
            kernel_init=variance_scaling(2.0, "fan_in", "truncated_normal"),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            features=1,
            kernel_init=variance_scaling(1.0, "fan_in", "truncated_normal"),
            bias_init=constant(0.0),
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class ActorCriticPolicy(_ActorCriticBase):
    """1D CNN ray encoder + scalar bypass."""

    n_rays: int = 256

    def _embed(self, obs: jax.Array) -> jax.Array:
        ray_dist = obs[..., : self.n_rays]
        ray_type = obs[..., self.n_rays : 2 * self.n_rays]
        scalars = obs[..., 2 * self.n_rays :]
        rays = jnp.stack([ray_dist, ray_type], axis=-1)
        ray_features = RayEncoder()(rays)
        return jnp.concatenate([scalars, ray_features], axis=-1)
