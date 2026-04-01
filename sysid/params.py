from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from environment.types import AgentDynamicsParams, DomainParams, PipelineParams


def _sigmoid(x: jax.Array) -> jax.Array:
    return 0.5 * (jnp.tanh(x / 2.0) + 1.0)


@dataclass(frozen=True, slots=True)
class ParamBounds:
    low: float
    high: float


PARAM_SPECS: tuple[tuple[str, ParamBounds], ...] = (
    ("track_width_scale", ParamBounds(0.92, 1.08)),
    ("motor_strength_scale", ParamBounds(0.75, 1.25)),
    ("back_emf_scale", ParamBounds(0.75, 1.25)),
    ("motor_balance", ParamBounds(-0.12, 0.12)),
    ("wheel_friction_scale", ParamBounds(0.75, 1.25)),
    ("wheel_scrub_scale", ParamBounds(0.50, 1.50)),
    ("wheel_frictionloss_scale", ParamBounds(0.75, 1.25)),
    ("caster_friction_scale", ParamBounds(0.75, 1.25)),
    ("mass_scale", ParamBounds(0.85, 1.15)),
    ("com_offset_x", ParamBounds(-0.005, 0.005)),
    ("com_offset_y", ParamBounds(-0.005, 0.005)),
    ("command_delay_substeps", ParamBounds(0.0, 8.0)),
    ("observation_delay_substeps", ParamBounds(0.0, 8.0)),
)

PARAM_NAMES = tuple(name for name, _ in PARAM_SPECS)
LATENT_DIM = len(PARAM_SPECS)


def latent_to_physical(latent: jax.Array) -> dict[str, jax.Array]:
    params: dict[str, jax.Array] = {}
    for index, (name, bounds) in enumerate(PARAM_SPECS):
        scale = _sigmoid(latent[index])
        params[name] = bounds.low + (bounds.high - bounds.low) * scale
    params["command_delay_substeps"] = jnp.rint(params["command_delay_substeps"])
    params["observation_delay_substeps"] = jnp.rint(
        params["observation_delay_substeps"]
    )
    return params


def build_domain_and_pipeline(latent: jax.Array) -> tuple[DomainParams, PipelineParams]:
    params = latent_to_physical(latent)
    agent = AgentDynamicsParams(
        mass_scale=params["mass_scale"],
        com_offset_xy=jnp.array(
            [params["com_offset_x"], params["com_offset_y"]], dtype=jnp.float32
        ),
        track_width_scale=params["track_width_scale"],
        wheel_friction_scale=params["wheel_friction_scale"],
        wheel_scrub_scale=params["wheel_scrub_scale"],
        caster_friction_scale=params["caster_friction_scale"],
        wheel_frictionloss_scale=params["wheel_frictionloss_scale"],
        motor_strength_scale=params["motor_strength_scale"],
        back_emf_scale=params["back_emf_scale"],
        motor_balance=params["motor_balance"],
    )
    domain = DomainParams(chaser=agent, evader=agent)
    pipeline = PipelineParams(
        action_delay_substeps=jnp.int32(params["command_delay_substeps"]),
        observation_delay_substeps=jnp.int32(params["observation_delay_substeps"]),
        action_drop_probability=jnp.asarray(0.0, dtype=jnp.float32),
        frame_drop_probability=jnp.asarray(0.0, dtype=jnp.float32),
        stale_observation_probability=jnp.asarray(0.0, dtype=jnp.float32),
        position_noise_std=jnp.asarray(0.0, dtype=jnp.float32),
        yaw_noise_std=jnp.asarray(0.0, dtype=jnp.float32),
        max_stale_steps=jnp.int32(0),
    )
    return domain, pipeline


def default_latent_center() -> jax.Array:
    return jnp.zeros((LATENT_DIM,), dtype=jnp.float32)
