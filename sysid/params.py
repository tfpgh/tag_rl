from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import jax
import jax.numpy as jnp

from environment.types import AgentDynamicsParams, DomainParams, PipelineParams

PARAM_TRANSFORM_LINEAR = "linear"
PARAM_TRANSFORM_LOG = "log"
PARAM_TRANSFORM_INTEGER = "integer"


@dataclass(frozen=True, slots=True)
class ParamBounds:
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class ParamSpec:
    name: str
    bounds: ParamBounds
    center: float
    transform: Literal["linear", "log", "integer"]


PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec(
        "track_width_scale",
        ParamBounds(0.75, 1.25),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "wheel_radius_scale",
        ParamBounds(0.85, 1.15),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "traction_scale",
        ParamBounds(0.1, 10.0),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "turn_scrub_scale",
        ParamBounds(0.25, 1.75),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "motor_strength_scale",
        ParamBounds(0.2, 3.0),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "back_emf_scale",
        ParamBounds(0.2, 3.0),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "motor_balance",
        ParamBounds(-0.15, 0.15),
        0.0,
        PARAM_TRANSFORM_LINEAR,
    ),
    ParamSpec(
        "motor_time_constant_seconds",
        ParamBounds(0.001, 0.080),
        0.01,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "command_delay_substeps",
        ParamBounds(0.0, 10.0),
        0.0,
        PARAM_TRANSFORM_INTEGER,
    ),
    ParamSpec(
        "observation_delay_substeps",
        ParamBounds(0.0, 22.0),
        0.0,
        PARAM_TRANSFORM_INTEGER,
    ),
)

PARAM_NAMES = tuple(spec.name for spec in PARAM_SPECS)
LATENT_DIM = len(PARAM_SPECS)
PARAM_BOUNDS = {spec.name: spec.bounds for spec in PARAM_SPECS}
PARAM_SPEC_BY_NAME = {spec.name: spec for spec in PARAM_SPECS}
MAX_COMMAND_DELAY_SUBSTEPS = int(PARAM_BOUNDS["command_delay_substeps"].high)
MAX_OBSERVATION_DELAY_SUBSTEPS = int(PARAM_BOUNDS["observation_delay_substeps"].high)
DEFAULT_PHYSICAL_PARAMS: dict[str, float] = {
    spec.name: float(spec.center)
    for spec in PARAM_SPECS
    if spec.transform != PARAM_TRANSFORM_INTEGER
}
DEFAULT_PHYSICAL_PARAMS["command_delay_substeps"] = 0.0
DEFAULT_PHYSICAL_PARAMS["observation_delay_substeps"] = 0.0


def _signed_sigmoid(x: jax.Array) -> jax.Array:
    return jnp.tanh(x / 2.0)


def _atanh(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.log((1.0 + x) / (1.0 - x))


def _validate_spec(spec: ParamSpec) -> None:
    if not (spec.bounds.low <= spec.center <= spec.bounds.high):
        raise ValueError(f"Center for {spec.name} must lie within bounds")
    if spec.transform == PARAM_TRANSFORM_LOG and spec.bounds.low <= 0:
        raise ValueError(f"Log-space param {spec.name} must have positive bounds")


for _spec in PARAM_SPECS:
    _validate_spec(_spec)


def _unit_to_physical(unit: jax.Array, spec: ParamSpec) -> jax.Array:
    low = jnp.asarray(spec.bounds.low, dtype=jnp.float32)
    high = jnp.asarray(spec.bounds.high, dtype=jnp.float32)
    center = jnp.asarray(spec.center, dtype=jnp.float32)
    if spec.transform == PARAM_TRANSFORM_LOG:
        log_low = jnp.log(low)
        log_high = jnp.log(high)
        log_center = jnp.log(center)
        return jnp.where(
            unit >= 0.0,
            jnp.exp(log_center + unit * (log_high - log_center)),
            jnp.exp(log_center + unit * (log_center - log_low)),
        )
    return jnp.where(
        unit >= 0.0,
        center + unit * (high - center),
        center + unit * (center - low),
    )


def _physical_to_unit(value: jax.Array, spec: ParamSpec) -> jax.Array:
    low = jnp.asarray(spec.bounds.low, dtype=jnp.float32)
    high = jnp.asarray(spec.bounds.high, dtype=jnp.float32)
    center = jnp.asarray(spec.center, dtype=jnp.float32)
    value = jnp.clip(value, low, high)
    if spec.transform == PARAM_TRANSFORM_LOG:
        log_value = jnp.log(value)
        log_low = jnp.log(low)
        log_high = jnp.log(high)
        log_center = jnp.log(center)
        unit = jnp.where(
            value >= center,
            (log_value - log_center) / jnp.maximum(log_high - log_center, 1e-6),
            (log_value - log_center) / jnp.maximum(log_center - log_low, 1e-6),
        )
    else:
        unit = jnp.where(
            value >= center,
            (value - center) / jnp.maximum(high - center, 1e-6),
            (value - center) / jnp.maximum(center - low, 1e-6),
        )
    return jnp.clip(unit, -1.0 + 1e-6, 1.0 - 1e-6)


def latent_to_physical(latent: jax.Array) -> dict[str, jax.Array]:
    params: dict[str, jax.Array] = {}
    for index, spec in enumerate(PARAM_SPECS):
        params[spec.name] = _unit_to_physical(_signed_sigmoid(latent[index]), spec)
    params["command_delay_substeps"] = jnp.rint(params["command_delay_substeps"])
    params["observation_delay_substeps"] = jnp.rint(
        params["observation_delay_substeps"]
    )
    return params


def physical_to_latent(params: Mapping[str, jax.Array | float | int]) -> jax.Array:
    latent: list[jax.Array] = []
    for spec in PARAM_SPECS:
        value = jnp.asarray(params[spec.name], dtype=jnp.float32)
        latent.append(2.0 * _atanh(_physical_to_unit(value, spec)))
    return jnp.stack(latent, axis=0)


def resolve_physical_params(
    params: Mapping[str, jax.Array | float | int],
) -> dict[str, jax.Array]:
    """Fill the keep-set physical params from a payload, defaulting missing
    entries and silently dropping unknown keys (so legacy sysid outputs with
    removed parameters still load)."""
    resolved: dict[str, jax.Array] = {
        name: jnp.asarray(value, dtype=jnp.float32)
        for name, value in DEFAULT_PHYSICAL_PARAMS.items()
    }
    for name, value in params.items():
        if name in resolved:
            resolved[name] = jnp.asarray(value, dtype=jnp.float32)
    return resolved


def build_domain_and_pipeline_from_physical(
    params: Mapping[str, jax.Array | float | int],
) -> tuple[DomainParams, PipelineParams]:
    resolved = resolve_physical_params(params)
    agent = AgentDynamicsParams(
        track_width_scale=resolved["track_width_scale"],
        wheel_radius_scale=resolved["wheel_radius_scale"],
        traction_scale=resolved["traction_scale"],
        turn_scrub_scale=resolved["turn_scrub_scale"],
        motor_strength_scale=resolved["motor_strength_scale"],
        back_emf_scale=resolved["back_emf_scale"],
        motor_balance=resolved["motor_balance"],
        motor_time_constant_seconds=resolved["motor_time_constant_seconds"],
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
