import jax.numpy as jnp
import mujoco


def quaternion_to_yaw(quaternion: jnp.ndarray) -> jnp.ndarray:
    w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quat(yaw: jnp.ndarray) -> jnp.ndarray:
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
