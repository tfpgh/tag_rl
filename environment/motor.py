import jax
import jax.numpy as jnp


def apply_deadzone(command: jax.Array, deadzone: jax.Array) -> jax.Array:
    deadzone = jnp.clip(deadzone, 0.0, 0.999)
    magnitude = jnp.abs(command)
    scaled = jnp.maximum(magnitude - deadzone, 0.0) / jnp.maximum(1.0 - deadzone, 1e-6)
    return jnp.sign(command) * jnp.clip(scaled, 0.0, 1.0)


def motor_lag_alpha(time_constant_seconds: jax.Array, dt_seconds: float) -> jax.Array:
    dt = jnp.asarray(dt_seconds, dtype=jnp.float32)
    return jnp.where(
        time_constant_seconds <= 1e-6,
        jnp.asarray(1.0, dtype=jnp.float32),
        1.0 - jnp.exp(-dt / time_constant_seconds),
    )


def shape_motor_command(
    previous_command: jax.Array,
    target_command: jax.Array,
    deadzone: jax.Array,
    time_constant_seconds: jax.Array,
    dt_seconds: float,
) -> jax.Array:
    deadzoned_target = apply_deadzone(target_command, deadzone)
    alpha = motor_lag_alpha(time_constant_seconds, dt_seconds)
    return previous_command + alpha * (deadzoned_target - previous_command)
