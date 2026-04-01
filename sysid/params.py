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
    ("track_width_scale", ParamBounds(0.88, 1.12)),
    ("motor_strength_scale", ParamBounds(0.75, 1.40)),
    ("back_emf_scale", ParamBounds(0.55, 1.25)),
    ("motor_balance", ParamBounds(-0.12, 0.12)),
    ("wheel_friction_scale", ParamBounds(0.75, 1.25)),
    ("wheel_friction_balance", ParamBounds(-0.12, 0.12)),
    ("wheel_scrub_scale", ParamBounds(0.50, 1.50)),
    ("wheel_frictionloss_scale", ParamBounds(0.75, 1.25)),
    ("wheel_frictionloss_balance", ParamBounds(-0.12, 0.12)),
    ("caster_friction_scale", ParamBounds(0.75, 1.25)),
    ("mass_scale", ParamBounds(0.98, 1.02)),
    ("com_offset_x", ParamBounds(-0.005, 0.005)),
    ("com_offset_y", ParamBounds(-0.005, 0.005)),
    ("command_delay_substeps", ParamBounds(0.0, 3.0)),
    ("observation_delay_substeps", ParamBounds(0.0, 12.0)),
)

PARAM_NAMES = tuple(name for name, _ in PARAM_SPECS)
LATENT_DIM = len(PARAM_SPECS)
PARAM_BOUNDS = {name: bounds for name, bounds in PARAM_SPECS}
MAX_COMMAND_DELAY_SUBSTEPS = int(PARAM_BOUNDS["command_delay_substeps"].high)
MAX_OBSERVATION_DELAY_SUBSTEPS = int(PARAM_BOUNDS["observation_delay_substeps"].high)


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


def build_domain_and_pipeline_from_physical(
    params: dict[str, jax.Array | float | int],
) -> tuple[DomainParams, PipelineParams]:
    agent = AgentDynamicsParams(
        mass_scale=jnp.asarray(params["mass_scale"], dtype=jnp.float32),
        com_offset_xy=jnp.array(
            [params["com_offset_x"], params["com_offset_y"]], dtype=jnp.float32
        ),
        track_width_scale=jnp.asarray(params["track_width_scale"], dtype=jnp.float32),
        wheel_friction_scale=jnp.asarray(
            params["wheel_friction_scale"], dtype=jnp.float32
        ),
        wheel_friction_balance=jnp.asarray(
            params["wheel_friction_balance"], dtype=jnp.float32
        ),
        wheel_scrub_scale=jnp.asarray(params["wheel_scrub_scale"], dtype=jnp.float32),
        caster_friction_scale=jnp.asarray(
            params["caster_friction_scale"], dtype=jnp.float32
        ),
        wheel_frictionloss_scale=jnp.asarray(
            params["wheel_frictionloss_scale"], dtype=jnp.float32
        ),
        wheel_frictionloss_balance=jnp.asarray(
            params["wheel_frictionloss_balance"], dtype=jnp.float32
        ),
        motor_strength_scale=jnp.asarray(
            params["motor_strength_scale"], dtype=jnp.float32
        ),
        back_emf_scale=jnp.asarray(params["back_emf_scale"], dtype=jnp.float32),
        motor_balance=jnp.asarray(params["motor_balance"], dtype=jnp.float32),
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


def build_domain_and_pipeline(latent: jax.Array) -> tuple[DomainParams, PipelineParams]:
    return build_domain_and_pipeline_from_physical(latent_to_physical(latent))


def default_latent_center() -> jax.Array:
    return jnp.zeros((LATENT_DIM,), dtype=jnp.float32)
