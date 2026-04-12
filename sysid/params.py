from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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
    ("track_width_scale", ParamBounds(0.75, 1.25)),  # mid 1.00
    ("wheel_radius_scale", ParamBounds(0.70, 1.30)),  # mid 1.00
    ("caster_radius_scale", ParamBounds(0.70, 1.30)),  # mid 1.00
    ("caster_offset_x", ParamBounds(-0.020, 0.020)),  # mid 0.0
    ("com_offset_x", ParamBounds(-0.020, 0.040)),  # mid 0.01
    ("wheel_slide_friction_scale", ParamBounds(0.50, 3.50)),  # mid 2.00
    ("wheel_torsional_friction_scale", ParamBounds(0.50, 3.50)),  # mid 2.00
    ("wheel_rolling_friction_scale", ParamBounds(0.50, 3.50)),  # mid 2.00
    ("caster_slide_friction_scale", ParamBounds(0.50, 3.50)),  # mid 2.00
    ("caster_torsional_friction_scale", ParamBounds(0.50, 3.50)),  # mid 2.00
    ("wheel_joint_damping_scale", ParamBounds(0.50, 4.50)),  # mid 2.50
    ("wheel_joint_frictionloss_scale", ParamBounds(0.05, 1.00)),  # mid 0.525
    ("wheel_armature_scale", ParamBounds(0.50, 4.50)),  # mid 2.50
    ("motor_strength_scale", ParamBounds(0.50, 4.50)),  # mid 2.50
    ("back_emf_scale", ParamBounds(0.50, 4.50)),  # mid 2.50
    ("motor_balance", ParamBounds(-0.20, 0.20)),  # mid 0.0
    ("motor_deadzone", ParamBounds(0.0, 0.010)),  # mid 0.005
    ("motor_time_constant_seconds", ParamBounds(0.010, 0.060)),  # mid 0.035
    ("command_delay_substeps", ParamBounds(0.0, 8.0)),  # mid 4
    ("observation_delay_substeps", ParamBounds(0.0, 8.0)),  # mid 4
)

FROZEN_PARAMS: dict[str, float] = {}

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


def physical_to_latent(params: Mapping[str, jax.Array | float | int]) -> jax.Array:
    latent: list[jax.Array] = []
    for name, bounds in PARAM_SPECS:
        value = jnp.asarray(params[name], dtype=jnp.float32)
        span = bounds.high - bounds.low
        fraction = (value - bounds.low) / span
        fraction = jnp.clip(fraction, 1e-6, 1.0 - 1e-6)
        latent.append(jnp.log(fraction / (1.0 - fraction)))
    return jnp.stack(latent, axis=0)


def build_domain_and_pipeline_from_physical(
    params: Mapping[str, jax.Array | float | int],
) -> tuple[DomainParams, PipelineParams]:
    resolved = {**FROZEN_PARAMS, **params}
    agent = AgentDynamicsParams(
        mass_scale=jnp.asarray(1.0, dtype=jnp.float32),
        com_offset_xy=jnp.array([resolved["com_offset_x"], 0.0], dtype=jnp.float32),
        track_width_scale=jnp.asarray(resolved["track_width_scale"], dtype=jnp.float32),
        wheel_radius_scale=jnp.asarray(
            resolved["wheel_radius_scale"], dtype=jnp.float32
        ),
        wheel_slide_friction_scale=jnp.asarray(
            resolved["wheel_slide_friction_scale"], dtype=jnp.float32
        ),
        wheel_torsional_friction_scale=jnp.asarray(
            resolved["wheel_torsional_friction_scale"], dtype=jnp.float32
        ),
        wheel_rolling_friction_scale=jnp.asarray(
            resolved["wheel_rolling_friction_scale"], dtype=jnp.float32
        ),
        caster_radius_scale=jnp.asarray(
            resolved["caster_radius_scale"], dtype=jnp.float32
        ),
        caster_offset_x=jnp.asarray(resolved["caster_offset_x"], dtype=jnp.float32),
        caster_slide_friction_scale=jnp.asarray(
            resolved["caster_slide_friction_scale"], dtype=jnp.float32
        ),
        caster_torsional_friction_scale=jnp.asarray(
            resolved["caster_torsional_friction_scale"], dtype=jnp.float32
        ),
        wheel_joint_damping_scale=jnp.asarray(
            resolved["wheel_joint_damping_scale"], dtype=jnp.float32
        ),
        wheel_joint_frictionloss_scale=jnp.asarray(
            resolved["wheel_joint_frictionloss_scale"], dtype=jnp.float32
        ),
        wheel_armature_scale=jnp.asarray(
            resolved["wheel_armature_scale"], dtype=jnp.float32
        ),
        motor_strength_scale=jnp.asarray(
            resolved["motor_strength_scale"], dtype=jnp.float32
        ),
        back_emf_scale=jnp.asarray(resolved["back_emf_scale"], dtype=jnp.float32),
        motor_balance=jnp.asarray(resolved["motor_balance"], dtype=jnp.float32),
        motor_deadzone=jnp.asarray(resolved["motor_deadzone"], dtype=jnp.float32),
        motor_time_constant_seconds=jnp.asarray(
            resolved["motor_time_constant_seconds"], dtype=jnp.float32
        ),
    )
    domain = DomainParams(chaser=agent, evader=agent)
    pipeline = PipelineParams(
        action_delay_substeps=jnp.int32(resolved["command_delay_substeps"]),
        observation_delay_substeps=jnp.int32(resolved["observation_delay_substeps"]),
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
