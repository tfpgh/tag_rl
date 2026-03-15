import jax.numpy as jnp

from environment.config import CurriculumConfig


def curriculum_progress(
    update_idx: int | float, num_updates: int, config: CurriculumConfig
) -> float:
    if not config.enabled or num_updates <= 1:
        return 1.0
    return float(jnp.clip(update_idx / float(num_updates - 1), 0.0, 1.0))


def curriculum_scale(
    progress: jnp.ndarray, start: float, full: float, exponent: float
) -> jnp.ndarray:
    if full <= start:
        return jnp.where(progress >= full, 1.0, 0.0)
    normalized = jnp.clip((progress - start) / (full - start), 0.0, 1.0)
    return normalized**exponent
