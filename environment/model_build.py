from __future__ import annotations

import copy
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from environment.types import AgentDynamicsParams, DomainParams, ModelIndices


def nominal_agent_dynamics_params() -> AgentDynamicsParams:
    one = jnp.asarray(1.0, dtype=jnp.float32)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return AgentDynamicsParams(
        mass_scale=one,
        com_offset_xy=jnp.zeros(2, dtype=jnp.float32),
        track_width_scale=one,
        wheel_friction_scale=one,
        wheel_scrub_scale=one,
        caster_friction_scale=one,
        wheel_frictionloss_scale=one,
        motor_strength_scale=one,
        back_emf_scale=one,
        motor_balance=zero,
    )


def nominal_domain_params() -> DomainParams:
    nominal = nominal_agent_dynamics_params()
    return DomainParams(chaser=nominal, evader=nominal)


def _apply_agent_dynamics(
    model: mujoco.MjModel,
    agent_indices,
    agent_params: AgentDynamicsParams,
) -> None:
    mass_scale = float(agent_params.mass_scale)
    model.body_mass[agent_indices.body_id] *= mass_scale
    model.body_inertia[agent_indices.body_id] *= mass_scale
    model.body_ipos[agent_indices.body_id, :2] += np.asarray(
        jax.device_get(agent_params.com_offset_xy), dtype=np.float64
    )

    track_width_scale = float(agent_params.track_width_scale)
    model.body_pos[agent_indices.left_wheel_body_id, 1] *= track_width_scale
    model.body_pos[agent_indices.right_wheel_body_id, 1] *= track_width_scale

    for wheel_geom_id in (
        agent_indices.left_wheel_geom_id,
        agent_indices.right_wheel_geom_id,
    ):
        model.geom_friction[wheel_geom_id, 0] *= float(
            agent_params.wheel_friction_scale
        )
        model.geom_friction[wheel_geom_id, 1] *= float(agent_params.wheel_scrub_scale)
        model.geom_friction[wheel_geom_id, 2] *= float(agent_params.wheel_scrub_scale)

    model.geom_friction[agent_indices.caster_geom_id] *= float(
        agent_params.caster_friction_scale
    )

    for dof_id in (agent_indices.left_wheel_dof_id, agent_indices.right_wheel_dof_id):
        model.dof_frictionloss[dof_id] *= float(agent_params.wheel_frictionloss_scale)

    strength_scale = float(agent_params.motor_strength_scale)
    back_emf_scale = float(agent_params.back_emf_scale)
    motor_balance = float(agent_params.motor_balance)
    left_scale = strength_scale * (1.0 - motor_balance)
    right_scale = strength_scale * (1.0 + motor_balance)
    left_back_emf = back_emf_scale * (1.0 - motor_balance)
    right_back_emf = back_emf_scale * (1.0 + motor_balance)
    model.actuator_gainprm[agent_indices.left_actuator_id, 0] *= left_scale
    model.actuator_gainprm[agent_indices.right_actuator_id, 0] *= right_scale
    model.actuator_biasprm[agent_indices.left_actuator_id, 2] *= left_back_emf
    model.actuator_biasprm[agent_indices.right_actuator_id, 2] *= right_back_emf


def build_cpu_model(
    base_model: mujoco.MjModel,
    domain_params: DomainParams,
    model_indices: ModelIndices,
) -> mujoco.MjModel:
    model = copy.deepcopy(base_model)
    _apply_agent_dynamics(model, model_indices.chaser, domain_params.chaser)
    _apply_agent_dynamics(model, model_indices.evader, domain_params.evader)
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    return model


def build_mjx_model(
    base_model: mujoco.MjModel,
    domain_params: DomainParams,
    model_indices: ModelIndices,
) -> mjx.Model:
    return mjx.put_model(build_cpu_model(base_model, domain_params, model_indices))


def stack_mjx_models(models: Sequence[mjx.Model]) -> mjx.Model:
    if not models:
        raise ValueError("models must be non-empty")

    def stack_leaf(first, *rest):
        if isinstance(first, np.ndarray):
            return first
        if hasattr(first, "dtype") and hasattr(first, "shape"):
            return jnp.stack((first, *rest), axis=0)
        return first

    return jax.tree.map(stack_leaf, models[0], *models[1:])


def gather_batched_mjx_model(model_pool: mjx.Model, indices: jax.Array) -> mjx.Model:
    def gather_leaf(leaf):
        if isinstance(leaf, np.ndarray):
            return leaf
        if hasattr(leaf, "shape") and leaf.ndim > 0:
            return leaf[indices]
        return leaf

    return jax.tree.map(gather_leaf, model_pool)
