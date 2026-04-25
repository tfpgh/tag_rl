from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import jax
import jax.numpy as jnp

from environment.config import EnvironmentConfig
from environment.types import AgentDynamicsParams, DomainParams, PipelineParams

PARAM_TRANSFORM_LINEAR = "linear"
PARAM_TRANSFORM_LOG = "log"
PARAM_TRANSFORM_INTEGER = "integer"

_ENV_CONFIG = EnvironmentConfig()
_DYNAMICS = _ENV_CONFIG.dynamics_randomization
_PIPELINE = _ENV_CONFIG.pipeline_randomization


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
        ParamBounds(0.25, 1.75),
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
        ParamBounds(0.25, 1.75),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "back_emf_scale",
        ParamBounds(0.25, 1.75),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "body_contact_friction_scale",
        ParamBounds(0.35, 1.75),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "caster_radius_scale",
        ParamBounds(0.5, 1.5),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "caster_offset_x",
        ParamBounds(-0.02, 0.02),
        0.0,
        PARAM_TRANSFORM_LINEAR,
    ),
    ParamSpec(
        "com_offset_x",
        ParamBounds(-0.02, 0.02),
        0.0,
        PARAM_TRANSFORM_LINEAR,
    ),
    ParamSpec(
        "com_offset_z",
        ParamBounds(-0.02, 0.02),
        0.0,
        PARAM_TRANSFORM_LINEAR,
    ),
    ParamSpec(
        "wheel_slide_friction_scale",
        ParamBounds(0.25, 2.0),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "wheel_torsional_friction_scale",
        ParamBounds(0.25, 2.0),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "wheel_rolling_friction_scale",
        ParamBounds(0.25, 2.0),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "caster_slide_friction_scale",
        ParamBounds(0.25, 2.5),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "caster_torsional_friction_scale",
        ParamBounds(0.25, 2.5),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "wheel_joint_damping_scale",
        ParamBounds(0.25, 4.0),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "wheel_joint_frictionloss_scale",
        ParamBounds(0.1, 4.0),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "wheel_armature_scale",
        ParamBounds(0.25, 4.0),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "motor_curve_sharpness",
        ParamBounds(0.0, 4.0),
        0.0,
        PARAM_TRANSFORM_LINEAR,
    ),
    ParamSpec(
        "traction_slip_threshold",
        ParamBounds(0.05, 2.0),
        1.0,
        PARAM_TRANSFORM_LOG,
    ),
    ParamSpec(
        "traction_loss_strength",
        ParamBounds(0.0, 4.0),
        0.0,
        PARAM_TRANSFORM_LINEAR,
    ),
    ParamSpec(
        "traction_min_scale",
        ParamBounds(0.25, 1.0),
        1.0,
        PARAM_TRANSFORM_LINEAR,
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

FROZEN_PARAMS: dict[str, float] = {}

PARAM_NAMES = tuple(spec.name for spec in PARAM_SPECS)
LATENT_DIM = len(PARAM_SPECS)
PARAM_BOUNDS = {spec.name: spec.bounds for spec in PARAM_SPECS}
PARAM_SPEC_BY_NAME = {spec.name: spec for spec in PARAM_SPECS}
MAX_COMMAND_DELAY_SUBSTEPS = int(PARAM_BOUNDS["command_delay_substeps"].high)
MAX_OBSERVATION_DELAY_SUBSTEPS = int(PARAM_BOUNDS["observation_delay_substeps"].high)
DEFAULT_PHYSICAL_PARAMS = {
    spec.name: float(spec.center)
    for spec in PARAM_SPECS
    if spec.transform != PARAM_TRANSFORM_INTEGER
}
DEFAULT_PHYSICAL_PARAMS.update(FROZEN_PARAMS)
DEFAULT_PHYSICAL_PARAMS["command_delay_substeps"] = float(0.0)
DEFAULT_PHYSICAL_PARAMS["observation_delay_substeps"] = float(0.0)


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
) -> dict[str, jax.Array | float]:
    resolved: dict[str, jax.Array | float] = {
        name: jnp.asarray(value, dtype=jnp.float32)
        for name, value in DEFAULT_PHYSICAL_PARAMS.items()
    }
    for name, value in params.items():
        resolved[name] = jnp.asarray(value, dtype=jnp.float32)
    if "traction_scale" not in params and "wheel_slide_friction_scale" in params:
        resolved["traction_scale"] = jnp.asarray(
            params["wheel_slide_friction_scale"], dtype=jnp.float32
        )
    if "turn_scrub_scale" not in params:
        torsion = params.get("wheel_torsional_friction_scale")
        rolling = params.get("wheel_rolling_friction_scale")
        if torsion is not None and rolling is not None:
            resolved["turn_scrub_scale"] = 0.5 * (
                jnp.asarray(torsion, dtype=jnp.float32)
                + jnp.asarray(rolling, dtype=jnp.float32)
            )
        elif torsion is not None:
            resolved["turn_scrub_scale"] = jnp.asarray(torsion, dtype=jnp.float32)
        elif rolling is not None:
            resolved["turn_scrub_scale"] = jnp.asarray(rolling, dtype=jnp.float32)
    return resolved


def build_domain_and_pipeline_from_physical(
    params: Mapping[str, jax.Array | float | int],
) -> tuple[DomainParams, PipelineParams]:
    resolved = resolve_physical_params(params)
    agent = AgentDynamicsParams(
        mass_scale=jnp.asarray(1.0, dtype=jnp.float32),
        com_offset_xyz=jnp.array(
            [resolved["com_offset_x"], 0.0, resolved["com_offset_z"]],
            dtype=jnp.float32,
        ),
        track_width_scale=jnp.asarray(resolved["track_width_scale"], dtype=jnp.float32),
        wheel_radius_scale=jnp.asarray(
            resolved["wheel_radius_scale"], dtype=jnp.float32
        ),
        traction_scale=jnp.asarray(resolved["traction_scale"], dtype=jnp.float32),
        turn_scrub_scale=jnp.asarray(resolved["turn_scrub_scale"], dtype=jnp.float32),
        wheel_slide_friction_scale=jnp.asarray(
            resolved["traction_scale"], dtype=jnp.float32
        ),
        wheel_torsional_friction_scale=jnp.asarray(
            resolved["turn_scrub_scale"], dtype=jnp.float32
        ),
        wheel_rolling_friction_scale=jnp.asarray(
            resolved["turn_scrub_scale"], dtype=jnp.float32
        ),
        body_contact_friction_scale=jnp.asarray(
            resolved["body_contact_friction_scale"], dtype=jnp.float32
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
        motor_deadzone=jnp.asarray(0.0, dtype=jnp.float32),
        motor_time_constant_seconds=jnp.asarray(
            resolved["motor_time_constant_seconds"], dtype=jnp.float32
        ),
        motor_curve_sharpness=jnp.asarray(
            resolved["motor_curve_sharpness"], dtype=jnp.float32
        ),
        traction_slip_threshold=jnp.asarray(
            resolved["traction_slip_threshold"], dtype=jnp.float32
        ),
        traction_loss_strength=jnp.asarray(
            resolved["traction_loss_strength"], dtype=jnp.float32
        ),
        traction_min_scale=jnp.asarray(
            resolved["traction_min_scale"], dtype=jnp.float32
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
