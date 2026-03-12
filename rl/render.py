import os

if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"


import pickle

import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from environment.config import EnvironmentConfig
from environment.environment import TagEnvironment
from rl.config import RLConfig
from rl.models import ActorCriticCNNRNN, ActorCriticRNN, ScannedRNN

# ── Configuration ──────────────────────────────────────────────────────
CHECKPOINT_PATH = "runs/Feb28_12-38-35_node002/checkpoints/step_419430400.pkl"
OUTPUT_PATH = "render.mp4"
NUM_EPISODES = 10
MODEL_TYPE = "cnn_rnn"  # or "rnn"
VIDEO_HEIGHT = 1080
VIDEO_WIDTH = 1920
PRNG_KEY = 0


def make_rollout(env, chaser_network, evader_network, rl_config):
    """Create a JIT'd single-episode rollout function (compiles once)."""

    @jax.jit
    def rollout(chaser_params, evader_params, rng):
        state, chaser_obs, evader_obs = env.reset(rng)
        chaser_hstate = ScannedRNN.initialize_carry(1, rl_config.hidden_size)
        evader_hstate = ScannedRNN.initialize_carry(1, rl_config.hidden_size)
        done = jnp.bool_(False)

        def _scan_step(carry, _):
            state, chaser_obs, evader_obs, chaser_hstate, evader_hstate, done = carry
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

            state, chaser_obs, evader_obs, _, _, step_done, _ = env.step(
                state, chaser_action, evader_action
            )
            done = done | step_done

            carry = (
                state,
                chaser_obs,
                evader_obs,
                chaser_hstate,
                evader_hstate,
                done,
            )
            mocap_pos = state.mjx_data.mocap_pos
            mocap_quat = state.mjx_data.mocap_quat
            return carry, (qpos, qvel, mocap_pos, mocap_quat, step_done)

        init_carry = (
            state,
            chaser_obs,
            evader_obs,
            chaser_hstate,
            evader_hstate,
            done,
        )
        _, (all_qpos, all_qvel, all_mocap_pos, all_mocap_quat, all_done) = jax.lax.scan(
            _scan_step, init_carry, None, length=env.max_steps
        )
        return all_qpos, all_qvel, all_mocap_pos, all_mocap_quat, all_done

    return rollout


def render_trajectory(mj_model, all_qpos, all_qvel, all_mocap_pos, all_mocap_quat):
    """Render a trajectory on CPU, returning a list of RGB frames."""
    renderer = mujoco.Renderer(mj_model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
    mj_data = mujoco.MjData(mj_model)

    frames = []
    n_frames = len(all_qpos)
    for i in range(n_frames):
        mj_data.qpos[:] = all_qpos[i]
        mj_data.qvel[:] = all_qvel[i]
        mj_data.mocap_pos[:] = all_mocap_pos[i]
        mj_data.mocap_quat[:] = all_mocap_quat[i]
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data)
        frames.append(renderer.render().copy())

    renderer.close()
    return frames


if __name__ == "__main__":
    rl_config = RLConfig()
    env_config = EnvironmentConfig()

    # Load checkpoint
    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    chaser_params = jax.device_put(checkpoint["chaser_params"])
    evader_params = jax.device_put(checkpoint["evader_params"])
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    # Create environment (single instance)
    env = TagEnvironment(env_config, 1)

    # Instantiate networks
    action_dim = (2,)
    if MODEL_TYPE == "cnn_rnn":
        chaser_network = ActorCriticCNNRNN(
            action_dim=action_dim,
            hidden_size=rl_config.hidden_size,
            n_rays=env_config.n_rays,
        )
        evader_network = ActorCriticCNNRNN(
            action_dim=action_dim,
            hidden_size=rl_config.hidden_size,
            n_rays=env_config.n_rays,
        )
    elif MODEL_TYPE == "rnn":
        chaser_network = ActorCriticRNN(
            action_dim=action_dim,
            hidden_size=rl_config.hidden_size,
        )
        evader_network = ActorCriticRNN(
            action_dim=action_dim,
            hidden_size=rl_config.hidden_size,
        )
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    # Build JIT'd rollout (compiles on first call)
    rollout_fn = make_rollout(env, chaser_network, evader_network, rl_config)

    rng = jax.random.PRNGKey(PRNG_KEY)
    all_frames = []

    for ep in range(NUM_EPISODES):
        rng, ep_rng = jax.random.split(rng)
        all_qpos, all_qvel, all_mocap_pos, all_mocap_quat, all_done = rollout_fn(
            chaser_params, evader_params, ep_rng
        )

        # Trim to first done
        all_done_np = np.array(all_done)
        done_indices = np.where(all_done_np)[0]
        n_frames = (
            int(done_indices[0]) + 1 if len(done_indices) > 0 else len(all_done_np)
        )

        # Transfer to CPU and trim
        qpos_np = np.array(all_qpos[:n_frames])
        qvel_np = np.array(all_qvel[:n_frames])
        mocap_pos_np = np.array(all_mocap_pos[:n_frames])
        mocap_quat_np = np.array(all_mocap_quat[:n_frames])

        frames = render_trajectory(
            env.mj_model, qpos_np, qvel_np, mocap_pos_np, mocap_quat_np
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
