import jax
import jax.numpy as jnp


def apply_deadzone(command: jax.Array, deadzone: jax.Array) -> jax.Array:
    deadzone = jnp.clip(deadzone, 0.0, 0.999)
    magnitude = jnp.abs(command)
    scaled = jnp.maximum(magnitude - deadzone, 0.0) / jnp.maximum(1.0 - deadzone, 1e-6)
    return jnp.sign(command) * jnp.clip(scaled, 0.0, 1.0)
