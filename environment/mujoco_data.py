import jax
import jax.numpy as jnp
import mujoco

JOINT_DIMS = {
    mujoco.mjtJoint.mjJNT_FREE: 7,
    mujoco.mjtJoint.mjJNT_BALL: 4,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}


def quaternion_to_yaw(quaternion: jax.Array) -> jax.Array:
    w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quaternion(yaw: jax.Array) -> jax.Array:
    return jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])


class SensorSlices:
    chaser_position: slice
    chaser_quaternion: slice
    chaser_velocity: slice
    chaser_angular_velocity: slice

    evader_position: slice
    evader_quaternion: slice
    evader_velocity: slice
    evader_angular_velocity: slice

    def __init__(self, model: mujoco.MjModel) -> None:
        for i in range(model.nsensor):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            adr = model.sensor_adr[i]
            dim = model.sensor_dim[i]

            setattr(self, name, slice(adr, adr + dim))


class JointQposSlices:
    chaser_root: slice
    chaser_left_wheel_joint: slice
    chaser_right_wheel_joint: slice
    chaser_caster_ball_joint: slice

    evader_root: slice
    evader_left_wheel_joint: slice
    evader_right_wheel_joint: slice
    evader_caster_ball_joint: slice

    def __init__(self, model: mujoco.MjModel) -> None:
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            adr = model.jnt_qposadr[i]
            dim = JOINT_DIMS[model.jnt_type[i]]

            setattr(self, name, slice(adr, adr + dim))
