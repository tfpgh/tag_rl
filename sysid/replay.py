from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import mujoco
import numpy as np

from environment.config import EnvironmentConfig
from environment.mjcf import generate_mjcf
from environment.mujoco_data import JointQposSlices
from sysid.align import align_run
from sysid.types import AlignedTrajectory, ReplayMetrics, RunData

RobotRole = Literal["chaser", "evader"]


@dataclass(frozen=True, slots=True)
class NominalParameters:
    action_delay_steps: int = 0
    floor_friction_scale: float = 1.0
    mass_scale: float = 1.0
    inertia_scale: float = 1.0
    wheel_friction_scale: float = 1.0
    caster_friction_scale: float = 1.0
    wheel_damping_scale: float = 1.0
    wheel_frictionloss_scale: float = 1.0
    motor_strength_scale: float = 1.0
    motor_balance: float = 0.0


@dataclass(frozen=True, slots=True)
class ModelIndices:
    floor_geom_id: int
    chaser_body_id: int
    evader_body_id: int
    chaser_left_wheel_geom_id: int
    chaser_right_wheel_geom_id: int
    chaser_caster_geom_id: int
    evader_left_wheel_geom_id: int
    evader_right_wheel_geom_id: int
    evader_caster_geom_id: int
    chaser_left_wheel_dof_id: int
    chaser_right_wheel_dof_id: int
    evader_left_wheel_dof_id: int
    evader_right_wheel_dof_id: int
    chaser_left_actuator_id: int
    chaser_right_actuator_id: int
    evader_left_actuator_id: int
    evader_right_actuator_id: int


def infer_target_role(run: RunData) -> RobotRole:
    if run.target_robot_tag_id == 5:
        return "evader"
    return "chaser"


def _yaw_to_quaternion(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)


def _quaternion_to_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


@lru_cache(maxsize=1)
def _base_model(
    config_key: tuple[float, ...],
) -> tuple[mujoco.MjModel, ModelIndices, JointQposSlices, int]:
    config = EnvironmentConfig()
    xml = generate_mjcf(config, mode="training")
    model = mujoco.MjModel.from_xml_string(xml)
    slices = JointQposSlices(model)
    substeps_per_action = round(1.0 / (config.action_frequency * model.opt.timestep))

    def geom_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def body_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    def joint_dof_id(name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(model.jnt_dofadr[joint_id])

    def actuator_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    indices = ModelIndices(
        floor_geom_id=geom_id("floor"),
        chaser_body_id=body_id("chaser"),
        evader_body_id=body_id("evader"),
        chaser_left_wheel_geom_id=geom_id("chaser_left_wheel_geom"),
        chaser_right_wheel_geom_id=geom_id("chaser_right_wheel_geom"),
        chaser_caster_geom_id=geom_id("chaser_caster_ball_geom"),
        evader_left_wheel_geom_id=geom_id("evader_left_wheel_geom"),
        evader_right_wheel_geom_id=geom_id("evader_right_wheel_geom"),
        evader_caster_geom_id=geom_id("evader_caster_ball_geom"),
        chaser_left_wheel_dof_id=joint_dof_id("chaser_left_wheel_joint"),
        chaser_right_wheel_dof_id=joint_dof_id("chaser_right_wheel_joint"),
        evader_left_wheel_dof_id=joint_dof_id("evader_left_wheel_joint"),
        evader_right_wheel_dof_id=joint_dof_id("evader_right_wheel_joint"),
        chaser_left_actuator_id=actuator_id("chaser_left_motor"),
        chaser_right_actuator_id=actuator_id("chaser_right_motor"),
        evader_left_actuator_id=actuator_id("evader_left_motor"),
        evader_right_actuator_id=actuator_id("evader_right_motor"),
    )
    return model, indices, slices, substeps_per_action


def _model_cache_key(config: EnvironmentConfig) -> tuple[float, ...]:
    return (
        config.arena_width,
        config.arena_height,
        config.agent_radius,
        config.agent_z,
        float(config.action_frequency),
    )


def _clone_model(
    config: EnvironmentConfig,
) -> tuple[mujoco.MjModel, ModelIndices, JointQposSlices, int]:
    base_model, indices, slices, substeps = _base_model(_model_cache_key(config))
    model = mujoco.MjModel.from_xml_string(generate_mjcf(config, mode="training"))
    return model, indices, slices, substeps


def _agent_indices(
    indices: ModelIndices, role: RobotRole
) -> tuple[int, int, int, int, int, int, int]:
    if role == "chaser":
        return (
            indices.chaser_body_id,
            indices.chaser_left_wheel_geom_id,
            indices.chaser_right_wheel_geom_id,
            indices.chaser_caster_geom_id,
            indices.chaser_left_wheel_dof_id,
            indices.chaser_right_wheel_dof_id,
            indices.chaser_left_actuator_id,
        )
    return (
        indices.evader_body_id,
        indices.evader_left_wheel_geom_id,
        indices.evader_right_wheel_geom_id,
        indices.evader_caster_geom_id,
        indices.evader_left_wheel_dof_id,
        indices.evader_right_wheel_dof_id,
        indices.evader_left_actuator_id,
    )


def _apply_nominal_parameters(
    model: mujoco.MjModel,
    indices: ModelIndices,
    params: NominalParameters,
    role: RobotRole,
) -> None:
    model.geom_friction[indices.floor_geom_id, 0] *= params.floor_friction_scale

    if role == "chaser":
        body_id = indices.chaser_body_id
        left_geom_id = indices.chaser_left_wheel_geom_id
        right_geom_id = indices.chaser_right_wheel_geom_id
        caster_geom_id = indices.chaser_caster_geom_id
        left_dof_id = indices.chaser_left_wheel_dof_id
        right_dof_id = indices.chaser_right_wheel_dof_id
        left_actuator_id = indices.chaser_left_actuator_id
        right_actuator_id = indices.chaser_right_actuator_id
    else:
        body_id = indices.evader_body_id
        left_geom_id = indices.evader_left_wheel_geom_id
        right_geom_id = indices.evader_right_wheel_geom_id
        caster_geom_id = indices.evader_caster_geom_id
        left_dof_id = indices.evader_left_wheel_dof_id
        right_dof_id = indices.evader_right_wheel_dof_id
        left_actuator_id = indices.evader_left_actuator_id
        right_actuator_id = indices.evader_right_actuator_id

    wheel_scale_left = params.wheel_friction_scale * (1.0 + params.motor_balance)
    wheel_scale_right = params.wheel_friction_scale * (1.0 - params.motor_balance)
    motor_scale_left = params.motor_strength_scale * (1.0 + params.motor_balance)
    motor_scale_right = params.motor_strength_scale * (1.0 - params.motor_balance)

    model.geom_friction[left_geom_id, 0] *= wheel_scale_left
    model.geom_friction[right_geom_id, 0] *= wheel_scale_right
    model.geom_friction[caster_geom_id, 0] *= params.caster_friction_scale
    model.body_mass[body_id] *= params.mass_scale
    model.body_inertia[body_id] *= params.mass_scale * params.inertia_scale
    model.dof_damping[left_dof_id] *= params.wheel_damping_scale
    model.dof_damping[right_dof_id] *= params.wheel_damping_scale
    model.dof_frictionloss[left_dof_id] *= params.wheel_frictionloss_scale
    model.dof_frictionloss[right_dof_id] *= params.wheel_frictionloss_scale
    model.actuator_gear[left_actuator_id, 0] *= motor_scale_left
    model.actuator_gear[right_actuator_id, 0] *= motor_scale_right


def _parked_pose(config: EnvironmentConfig) -> tuple[np.ndarray, float]:
    return (
        np.array(
            [
                0.5 * config.arena_width - 2.0 * config.agent_radius,
                -(0.5 * config.arena_height - 2.0 * config.agent_radius),
            ],
            dtype=np.float64,
        ),
        0.0,
    )


def _set_initial_state(
    data: mujoco.MjData,
    slices: JointQposSlices,
    trajectory: AlignedTrajectory,
    role: RobotRole,
    config: EnvironmentConfig,
) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    target_xy = np.array([trajectory.x_m[0], trajectory.y_m[0]], dtype=np.float64)
    parked_xy, parked_yaw = _parked_pose(config)
    target_quat = _yaw_to_quaternion(trajectory.yaw_rad[0])
    parked_quat = _yaw_to_quaternion(parked_yaw)
    identity_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if role == "chaser":
        chaser_xy, chaser_quat = target_xy, target_quat
        evader_xy, evader_quat = parked_xy, parked_quat
    else:
        chaser_xy, chaser_quat = parked_xy, parked_quat
        evader_xy, evader_quat = target_xy, target_quat

    data.qpos[slices.chaser_root.start : slices.chaser_root.start + 3] = [
        chaser_xy[0],
        chaser_xy[1],
        config.agent_z,
    ]
    data.qpos[slices.chaser_root.start + 3 : slices.chaser_root.start + 7] = chaser_quat
    data.qpos[slices.chaser_caster_ball_joint] = identity_quaternion
    data.qpos[slices.evader_root.start : slices.evader_root.start + 3] = [
        evader_xy[0],
        evader_xy[1],
        config.agent_z,
    ]
    data.qpos[slices.evader_root.start + 3 : slices.evader_root.start + 7] = evader_quat
    data.qpos[slices.evader_caster_ball_joint] = identity_quaternion
    mujoco.mj_forward(data.model, data)


def _command_array(trajectory: AlignedTrajectory, role: RobotRole) -> np.ndarray:
    left = np.asarray(trajectory.left, dtype=np.float64)
    right = np.asarray(trajectory.right, dtype=np.float64)
    zeros = np.zeros_like(left)
    if role == "chaser":
        return np.stack([left, right, zeros, zeros], axis=-1)
    return np.stack([zeros, zeros, left, right], axis=-1)


def _apply_action_delay(ctrl: np.ndarray, action_delay_steps: int) -> np.ndarray:
    if action_delay_steps <= 0:
        return ctrl
    delayed = np.zeros_like(ctrl)
    delayed[action_delay_steps:] = ctrl[:-action_delay_steps]
    return delayed


def replay_aligned_trajectory(
    trajectory: AlignedTrajectory,
    params: NominalParameters,
    target_role: RobotRole,
    env_config: EnvironmentConfig | None = None,
) -> np.ndarray:
    if not trajectory.times_s:
        return np.zeros((0, 3), dtype=np.float64)

    config = env_config or EnvironmentConfig()
    model, indices, slices, substeps_per_action = _clone_model(config)
    data = mujoco.MjData(model)
    _apply_nominal_parameters(model, indices, params, target_role)
    _set_initial_state(data, slices, trajectory, target_role, config)
    controls = _apply_action_delay(
        _command_array(trajectory, target_role), params.action_delay_steps
    )

    poses = np.zeros((len(trajectory.times_s), 3), dtype=np.float64)
    for index, ctrl in enumerate(controls):
        data.ctrl[:] = ctrl
        for _ in range(substeps_per_action):
            mujoco.mj_step(model, data)
        root_slice = (
            slices.chaser_root if target_role == "chaser" else slices.evader_root
        )
        root_qpos = data.qpos[root_slice]
        poses[index, 0] = root_qpos[0]
        poses[index, 1] = root_qpos[1]
        poses[index, 2] = _quaternion_to_yaw(np.asarray(root_qpos[3:7]))
    return poses


def evaluate_run(
    run: RunData,
    params: NominalParameters,
    target_hz: float | None = None,
    env_config: EnvironmentConfig | None = None,
) -> ReplayMetrics:
    trajectory = align_run(run, target_hz=target_hz)
    if not trajectory.times_s:
        return ReplayMetrics(0.0, 0.0, 0.0, 0.0, float("inf"), 0)

    predicted = replay_aligned_trajectory(
        trajectory, params, infer_target_role(run), env_config=env_config
    )
    reference = np.stack([trajectory.x_m, trajectory.y_m, trajectory.yaw_rad], axis=-1)
    delta_xy = reference[:, :2] - predicted[:, :2]
    position_sq = np.sum(np.square(delta_xy), axis=-1)
    yaw_delta = np.arctan2(
        np.sin(reference[:, 2] - predicted[:, 2]),
        np.cos(reference[:, 2] - predicted[:, 2]),
    )
    position_rmse = float(np.sqrt(np.mean(position_sq)))
    yaw_rmse = float(np.sqrt(np.mean(np.square(yaw_delta))))
    endpoint_position = float(np.sqrt(position_sq[-1]))
    endpoint_yaw = float(abs(yaw_delta[-1]))
    score = (
        position_rmse + 0.25 * yaw_rmse + 0.5 * endpoint_position + 0.1 * endpoint_yaw
    )
    return ReplayMetrics(
        position_rmse_m=position_rmse,
        yaw_rmse_rad=yaw_rmse,
        endpoint_position_error_m=endpoint_position,
        endpoint_yaw_error_rad=endpoint_yaw,
        score=score,
        sample_count=int(reference.shape[0]),
    )


def summarize_metrics(metrics: ReplayMetrics) -> dict[str, float]:
    return {
        "position_rmse_m": metrics.position_rmse_m,
        "yaw_rmse_rad": metrics.yaw_rmse_rad,
        "endpoint_position_error_m": metrics.endpoint_position_error_m,
        "endpoint_yaw_error_rad": metrics.endpoint_yaw_error_rad,
        "score": metrics.score,
        "sample_count": float(metrics.sample_count),
    }
