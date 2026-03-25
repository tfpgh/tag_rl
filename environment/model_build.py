from __future__ import annotations

import copy
import multiprocessing as mp
import os
import threading
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from environment.types import AgentDynamicsParams, DomainParams, ModelIndices


@dataclass(frozen=True, slots=True)
class HostAgentDynamicsParams:
    mass_scale: float
    com_offset_x: float
    com_offset_y: float
    track_width_scale: float
    wheel_friction_scale: float
    wheel_scrub_scale: float
    caster_friction_scale: float
    wheel_frictionloss_scale: float
    motor_strength_scale: float
    back_emf_scale: float
    motor_balance: float


@dataclass(frozen=True, slots=True)
class HostDomainParams:
    chaser: HostAgentDynamicsParams
    evader: HostAgentDynamicsParams


_PROCESS_LOCAL_BUILDER = None


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


def nominal_host_agent_dynamics_params() -> HostAgentDynamicsParams:
    return HostAgentDynamicsParams(
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0
    )


def nominal_host_domain_params() -> HostDomainParams:
    nominal = nominal_host_agent_dynamics_params()
    return HostDomainParams(chaser=nominal, evader=nominal)


def to_host_domain_params(domain_params: DomainParams) -> HostDomainParams:
    def convert(agent: AgentDynamicsParams) -> HostAgentDynamicsParams:
        com = np.asarray(agent.com_offset_xy, dtype=np.float64)
        return HostAgentDynamicsParams(
            mass_scale=float(agent.mass_scale),
            com_offset_x=float(com[0]),
            com_offset_y=float(com[1]),
            track_width_scale=float(agent.track_width_scale),
            wheel_friction_scale=float(agent.wheel_friction_scale),
            wheel_scrub_scale=float(agent.wheel_scrub_scale),
            caster_friction_scale=float(agent.caster_friction_scale),
            wheel_frictionloss_scale=float(agent.wheel_frictionloss_scale),
            motor_strength_scale=float(agent.motor_strength_scale),
            back_emf_scale=float(agent.back_emf_scale),
            motor_balance=float(agent.motor_balance),
        )

    return HostDomainParams(
        chaser=convert(domain_params.chaser), evader=convert(domain_params.evader)
    )


@dataclass(frozen=True, slots=True)
class _BaseArrays:
    body_mass: np.ndarray
    body_inertia: np.ndarray
    body_ipos: np.ndarray
    body_pos: np.ndarray
    geom_friction: np.ndarray
    dof_frictionloss: np.ndarray
    actuator_gainprm: np.ndarray
    actuator_biasprm: np.ndarray


class ModelBuilder:
    def __init__(self, base_model: mujoco.MjModel, model_indices: ModelIndices) -> None:
        self.model = copy.deepcopy(base_model)
        self.data = mujoco.MjData(self.model)
        self.model_indices = model_indices
        self.base = _BaseArrays(
            body_mass=np.array(base_model.body_mass, copy=True),
            body_inertia=np.array(base_model.body_inertia, copy=True),
            body_ipos=np.array(base_model.body_ipos, copy=True),
            body_pos=np.array(base_model.body_pos, copy=True),
            geom_friction=np.array(base_model.geom_friction, copy=True),
            dof_frictionloss=np.array(base_model.dof_frictionloss, copy=True),
            actuator_gainprm=np.array(base_model.actuator_gainprm, copy=True),
            actuator_biasprm=np.array(base_model.actuator_biasprm, copy=True),
        )

    def restore(self) -> None:
        self.model.body_mass[:] = self.base.body_mass
        self.model.body_inertia[:] = self.base.body_inertia
        self.model.body_ipos[:] = self.base.body_ipos
        self.model.body_pos[:] = self.base.body_pos
        self.model.geom_friction[:] = self.base.geom_friction
        self.model.dof_frictionloss[:] = self.base.dof_frictionloss
        self.model.actuator_gainprm[:] = self.base.actuator_gainprm
        self.model.actuator_biasprm[:] = self.base.actuator_biasprm

    def _apply_agent_dynamics(
        self, agent_indices, agent_params: HostAgentDynamicsParams
    ) -> None:
        mass_scale = agent_params.mass_scale
        self.model.body_mass[agent_indices.body_id] *= mass_scale
        self.model.body_inertia[agent_indices.body_id] *= mass_scale
        self.model.body_ipos[agent_indices.body_id, 0] += agent_params.com_offset_x
        self.model.body_ipos[agent_indices.body_id, 1] += agent_params.com_offset_y

        track_width_scale = agent_params.track_width_scale
        self.model.body_pos[agent_indices.left_wheel_body_id, 1] *= track_width_scale
        self.model.body_pos[agent_indices.right_wheel_body_id, 1] *= track_width_scale

        for wheel_geom_id in (
            agent_indices.left_wheel_geom_id,
            agent_indices.right_wheel_geom_id,
        ):
            self.model.geom_friction[wheel_geom_id, 0] *= (
                agent_params.wheel_friction_scale
            )
            self.model.geom_friction[wheel_geom_id, 1] *= agent_params.wheel_scrub_scale
            self.model.geom_friction[wheel_geom_id, 2] *= agent_params.wheel_scrub_scale

        self.model.geom_friction[agent_indices.caster_geom_id] *= (
            agent_params.caster_friction_scale
        )

        for dof_id in (
            agent_indices.left_wheel_dof_id,
            agent_indices.right_wheel_dof_id,
        ):
            self.model.dof_frictionloss[dof_id] *= agent_params.wheel_frictionloss_scale

        strength_scale = agent_params.motor_strength_scale
        back_emf_scale = agent_params.back_emf_scale
        motor_balance = agent_params.motor_balance
        left_scale = strength_scale * (1.0 - motor_balance)
        right_scale = strength_scale * (1.0 + motor_balance)
        left_back_emf = back_emf_scale * (1.0 - motor_balance)
        right_back_emf = back_emf_scale * (1.0 + motor_balance)
        self.model.actuator_gainprm[agent_indices.left_actuator_id, 0] *= left_scale
        self.model.actuator_gainprm[agent_indices.right_actuator_id, 0] *= right_scale
        self.model.actuator_biasprm[agent_indices.left_actuator_id, 2] *= left_back_emf
        self.model.actuator_biasprm[agent_indices.right_actuator_id, 2] *= (
            right_back_emf
        )

    def build_cpu_model(self, domain_params: HostDomainParams) -> mujoco.MjModel:
        self.restore()
        self._apply_agent_dynamics(self.model_indices.chaser, domain_params.chaser)
        self._apply_agent_dynamics(self.model_indices.evader, domain_params.evader)
        mujoco.mj_setConst(self.model, self.data)
        return self.model

    def build_mjx_model(self, domain_params: HostDomainParams) -> mjx.Model:
        return mjx.put_model(self.build_cpu_model(domain_params))


def _build_agent_model_indices(model: mujoco.MjModel, prefix: str):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix)
    left_wheel_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_left_wheel"
    )
    right_wheel_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_right_wheel"
    )
    left_wheel_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_left_wheel_geom"
    )
    right_wheel_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_right_wheel_geom"
    )
    caster_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_caster_ball_geom"
    )
    left_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_left_wheel_joint"
    )
    right_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_right_wheel_joint"
    )
    left_actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}_left_motor"
    )
    right_actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}_right_motor"
    )
    from environment.types import AgentModelIndices

    return AgentModelIndices(
        body_id=body_id,
        left_wheel_body_id=left_wheel_body_id,
        right_wheel_body_id=right_wheel_body_id,
        left_wheel_geom_id=left_wheel_geom_id,
        right_wheel_geom_id=right_wheel_geom_id,
        caster_geom_id=caster_geom_id,
        left_wheel_dof_id=int(model.jnt_dofadr[left_joint_id]),
        right_wheel_dof_id=int(model.jnt_dofadr[right_joint_id]),
        left_actuator_id=left_actuator_id,
        right_actuator_id=right_actuator_id,
    )


def build_model_indices(model: mujoco.MjModel) -> ModelIndices:
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    return ModelIndices(
        floor_geom_id=floor_geom_id,
        chaser=_build_agent_model_indices(model, "chaser"),
        evader=_build_agent_model_indices(model, "evader"),
    )


def _init_process_builder(model_xml: str) -> None:
    global _PROCESS_LOCAL_BUILDER
    model = mujoco.MjModel.from_xml_string(model_xml)
    _PROCESS_LOCAL_BUILDER = ModelBuilder(model, build_model_indices(model))


def _process_build_one(params: HostDomainParams) -> mjx.Model:
    if _PROCESS_LOCAL_BUILDER is None:
        raise RuntimeError("process-local model builder not initialized")
    return _PROCESS_LOCAL_BUILDER.build_mjx_model(params)


def create_process_model_pool(model_xml: str, max_workers: int | None = None):
    worker_count = max_workers or _default_build_workers()
    if "spawn" in mp.get_all_start_methods():
        ctx = mp.get_context("spawn")
    else:
        ctx = None
    return ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=ctx,
        initializer=_init_process_builder,
        initargs=(model_xml,),
    )


def build_cpu_model(
    base_model: mujoco.MjModel,
    domain_params: HostDomainParams,
    model_indices: ModelIndices,
) -> mujoco.MjModel:
    return ModelBuilder(base_model, model_indices).build_cpu_model(domain_params)


def build_mjx_model(
    base_model: mujoco.MjModel,
    domain_params: HostDomainParams,
    model_indices: ModelIndices,
) -> mjx.Model:
    return ModelBuilder(base_model, model_indices).build_mjx_model(domain_params)


def _default_build_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 16))


def build_mjx_models_parallel(
    base_model: mujoco.MjModel,
    model_indices: ModelIndices,
    domain_params_list: Sequence[HostDomainParams],
    max_workers: int | None = None,
) -> list[mjx.Model]:
    if not domain_params_list:
        return []
    if len(domain_params_list) == 1:
        return [build_mjx_model(base_model, domain_params_list[0], model_indices)]

    worker_count = max_workers or _default_build_workers()
    worker_count = max(1, min(worker_count, len(domain_params_list)))
    if worker_count == 1:
        return [
            build_mjx_model(base_model, params, model_indices)
            for params in domain_params_list
        ]

    local = threading.local()

    def build_one(params: HostDomainParams) -> mjx.Model:
        builder = getattr(local, "builder", None)
        if builder is None:
            builder = ModelBuilder(base_model, model_indices)
            local.builder = builder
        return builder.build_mjx_model(params)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(build_one, domain_params_list))


def build_mjx_models_process_parallel(
    model_xml: str,
    domain_params_list: Sequence[HostDomainParams],
    max_workers: int | None = None,
    executor: ProcessPoolExecutor | None = None,
) -> list[mjx.Model]:
    if not domain_params_list:
        return []
    if executor is not None:
        return list(executor.map(_process_build_one, domain_params_list))
    worker_count = max_workers or _default_build_workers()
    if worker_count <= 1 or len(domain_params_list) == 1:
        local_model = mujoco.MjModel.from_xml_string(model_xml)
        local_builder = ModelBuilder(local_model, build_model_indices(local_model))
        return [local_builder.build_mjx_model(params) for params in domain_params_list]
    with create_process_model_pool(model_xml, worker_count) as pool:
        return list(pool.map(_process_build_one, domain_params_list))


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


def pad_stacked_mjx_models(model_batch: mjx.Model, pad: int) -> mjx.Model:
    if pad <= 0:
        return model_batch

    def pad_leaf(leaf):
        if isinstance(leaf, np.ndarray):
            return leaf
        if hasattr(leaf, "shape") and leaf.ndim > 0:
            pad_width = [(0, pad)] + [(0, 0)] * (leaf.ndim - 1)
            return jnp.pad(leaf, pad_width)
        return leaf

    return jax.tree.map(pad_leaf, model_batch)


def reshape_stacked_mjx_models_for_devices(
    model_batch: mjx.Model, device_count: int
) -> mjx.Model:
    def reshape_leaf(leaf):
        if isinstance(leaf, np.ndarray):
            return leaf
        if hasattr(leaf, "shape") and leaf.ndim > 0:
            shard_size = leaf.shape[0] // device_count
            return leaf.reshape((device_count, shard_size) + leaf.shape[1:])
        return leaf

    return jax.tree.map(reshape_leaf, model_batch)


def device_put_sharded_mjx_models(model_batch: mjx.Model, devices) -> mjx.Model:
    device_count = len(devices)

    def put_leaf(leaf):
        if isinstance(leaf, np.ndarray):
            return leaf
        if hasattr(leaf, "shape") and leaf.ndim > 0 and leaf.shape[0] == device_count:
            shards = [leaf[i] for i in range(device_count)]
            return jax.device_put_sharded(shards, devices)
        return leaf

    return jax.tree.map(put_leaf, model_batch)


def mjx_model_in_axes(model_tree: mjx.Model):
    def in_axis_leaf(leaf):
        if isinstance(leaf, np.ndarray):
            return None
        if hasattr(leaf, "shape") and leaf.ndim > 0:
            return 0
        return None

    return jax.tree.map(in_axis_leaf, model_tree)


def gather_batched_mjx_model(model_pool: mjx.Model, indices: jax.Array) -> mjx.Model:
    def gather_leaf(leaf):
        if isinstance(leaf, np.ndarray):
            return leaf
        if hasattr(leaf, "shape") and leaf.ndim > 0:
            return leaf[indices]
        return leaf

    return jax.tree.map(gather_leaf, model_pool)
