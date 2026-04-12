import os

if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"


from collections.abc import Callable
from typing import Any

from runtime import jax_setup as _jax_setup  # noqa: F401

import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import numpy.typing as npt

from environment.environment import TagEnvironment
from environment.mjcf import generate_mjcf
from environment.mujoco_data import yaw_to_quaternion
from rl.checkpoints import load_checkpoint_configs, load_policy_params
from rl.models import ScannedRNN
from rl.policy import PolicyModule, build_policies

RolloutResult = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
RenderFrames = list[npt.NDArray[np.uint8]]

# ── Configuration ──────────────────────────────────────────────────────
CHECKPOINT_PATH = "runs/Feb28_12-38-35_node002/checkpoints/step_419430400.pkl"
OUTPUT_PATH = "render.mp4"
NUM_EPISODES = 10
VIDEO_HEIGHT = 1080
VIDEO_WIDTH = 1920
PRNG_KEY = 0
CURRICULUM_PROGRESS = 0.0  # 0.0 = nominal params only, 1.0 = full DR


def make_rollout(
    env: TagEnvironment,
    chaser_network: PolicyModule,
    evader_network: PolicyModule,
    hidden_size: int,
) -> Callable[[Any, Any, jax.Array], RolloutResult]:
    """Create a JIT'd single-episode rollout function (compiles once)."""

    @jax.jit
    def rollout(
        chaser_params: Any, evader_params: Any, rng: jax.Array
    ) -> RolloutResult:
        rng, reset_rng = jax.random.split(rng)
        state, chaser_obs, evader_obs = env.reset(
            reset_rng, jnp.asarray(CURRICULUM_PROGRESS, dtype=jnp.float32)
        )
        chaser_hstate = ScannedRNN.initialize_carry(1, hidden_size)
        evader_hstate = ScannedRNN.initialize_carry(1, hidden_size)
        done = jnp.bool_(False)

        def _scan_step(
            carry: tuple[
                Any,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
            _: None,
        ) -> tuple[
            tuple[
                Any,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
            tuple[jax.Array, jax.Array, jax.Array],
        ]:
            (
                state,
                chaser_obs,
                evader_obs,
                chaser_hstate,
                evader_hstate,
                done,
                rng,
            ) = carry
            qpos = state.mjx_data.qpos
            qvel = state.mjx_data.qvel

            chaser_ac_in = (
                chaser_obs[jnp.newaxis, jnp.newaxis, :],
                done[jnp.newaxis, jnp.newaxis],
            )
            chaser_hstate, chaser_pi, _ = chaser_network.apply(
                chaser_params, chaser_hstate, chaser_ac_in
            )
            chaser_action = jnp.clip(chaser_pi.loc[0, 0], -1.0, 1.0)

            evader_ac_in = (
                evader_obs[jnp.newaxis, jnp.newaxis, :],
                done[jnp.newaxis, jnp.newaxis],
            )
            evader_hstate, evader_pi, _ = evader_network.apply(
                evader_params, evader_hstate, evader_ac_in
            )
            evader_action = jnp.clip(evader_pi.loc[0, 0], -1.0, 1.0)

            # Zero actions after episode ends so agents stay still
            chaser_action = jnp.where(done, jnp.zeros(2), chaser_action)
            evader_action = jnp.where(done, jnp.zeros(2), evader_action)

            step_rng, next_rng = jax.random.split(rng)
            state, chaser_obs, evader_obs, _, _, step_done, _ = env.step(
                state, chaser_action, evader_action, step_rng
            )
            done = done | step_done

            carry = (
                state,
                chaser_obs,
                evader_obs,
                chaser_hstate,
                evader_hstate,
                done,
                next_rng,
            )
            return carry, (qpos, qvel, step_done)

        init_carry = (
            state,
            chaser_obs,
            evader_obs,
            chaser_hstate,
            evader_hstate,
            done,
            rng,
        )
        _, (all_qpos, all_qvel, all_done) = jax.lax.scan(
            _scan_step, init_carry, None, length=env.max_steps
        )
        return (
            all_qpos,
            all_qvel,
            state.obstacle_positions_xy,
            state.obstacle_yaws,
            state.obstacle_active,
            all_done,
        )

    return rollout


def obstacle_mocap_buffers(
    mj_model: mujoco.MjModel,
    obstacle_positions_xy: npt.NDArray[np.float32 | np.float64],
    obstacle_yaws: npt.NDArray[np.float32 | np.float64],
    obstacle_active: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    mocap_pos = np.zeros((mj_model.nmocap, 3), dtype=np.float64)
    mocap_quat = np.zeros((mj_model.nmocap, 4), dtype=np.float64)
    mocap_quat[:, 0] = 1.0

    for i in range(len(obstacle_active)):
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}")
        mocap_id = int(mj_model.body_mocapid[body_id])
        if obstacle_active[i]:
            mocap_pos[mocap_id] = np.array(
                [obstacle_positions_xy[i, 0], obstacle_positions_xy[i, 1], 0.05],
                dtype=np.float64,
            )
            mocap_quat[mocap_id] = np.asarray(yaw_to_quaternion(obstacle_yaws[i]))
        else:
            mocap_pos[mocap_id] = np.array([0.0, 0.0, -10.0], dtype=np.float64)
    return mocap_pos, mocap_quat


def render_trajectory(
    mj_model: mujoco.MjModel,
    all_qpos: npt.NDArray[np.float64],
    all_qvel: npt.NDArray[np.float64],
    obstacle_positions_xy: npt.NDArray[np.float64],
    obstacle_yaws: npt.NDArray[np.float64],
    obstacle_active: npt.NDArray[np.bool_],
) -> RenderFrames:
    """Render a trajectory on CPU, returning a list of RGB frames."""
    renderer = mujoco.Renderer(mj_model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
    mj_data = mujoco.MjData(mj_model)
    mocap_pos, mocap_quat = obstacle_mocap_buffers(
        mj_model, obstacle_positions_xy, obstacle_yaws, obstacle_active
    )

    frames: RenderFrames = []
    n_frames = len(all_qpos)
    for i in range(n_frames):
        mj_data.qpos[:] = all_qpos[i]
        mj_data.qvel[:] = all_qvel[i]
        mj_data.mocap_pos[:] = mocap_pos
        mj_data.mocap_quat[:] = mocap_quat
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data)
        frames.append(renderer.render().copy())

    renderer.close()
    return frames


if __name__ == "__main__":
    chaser_params, evader_params, checkpoint = load_policy_params(CHECKPOINT_PATH)
    rl_config, env_config = load_checkpoint_configs(checkpoint)
    chaser_params = jax.device_put(chaser_params)
    evader_params = jax.device_put(evader_params)
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    # Create environment (single instance)
    env = TagEnvironment(env_config)
    render_model = mujoco.MjModel.from_xml_string(
        generate_mjcf(env_config, mode="render")
    )

    chaser_network, evader_network = build_policies(rl_config.hidden_size, env_config)

    # Build JIT'd rollout (compiles on first call)
    rollout_fn = make_rollout(
        env, chaser_network, evader_network, rl_config.hidden_size
    )

    rng = jax.random.PRNGKey(PRNG_KEY)
    all_frames = []

    for ep in range(NUM_EPISODES):
        rng, ep_rng = jax.random.split(rng)
        (
            all_qpos,
            all_qvel,
            obstacle_positions_xy,
            obstacle_yaws,
            obstacle_active,
            all_done,
        ) = rollout_fn(chaser_params, evader_params, ep_rng)

        (
            all_qpos_host,
            all_qvel_host,
            obstacle_positions_xy_host,
            obstacle_yaws_host,
            obstacle_active_host,
            all_done_host,
        ) = jax.device_get(
            (
                all_qpos,
                all_qvel,
                obstacle_positions_xy,
                obstacle_yaws,
                obstacle_active,
                all_done,
            )
        )

        # Trim to first done
        done_indices = np.where(all_done_host)[0]
        n_frames = (
            int(done_indices[0]) + 1 if len(done_indices) > 0 else len(all_done_host)
        )

        # Transfer to CPU and trim
        frames = render_trajectory(
            render_model,
            all_qpos_host[:n_frames],
            all_qvel_host[:n_frames],
            obstacle_positions_xy_host,
            obstacle_yaws_host,
            obstacle_active_host,
        )
        all_frames.extend(frames)
        print(f"Episode {ep + 1}/{NUM_EPISODES}: {n_frames} frames")

    # Write video
    imageio.mimwrite(
        OUTPUT_PATH,
        all_frames,
        fps=env_config.action_frequency,
        codec="h264",
    )
    print(f"Saved {len(all_frames)} frames to {OUTPUT_PATH}")
