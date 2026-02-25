"""
Profile training to identify bottlenecks.

Usage:
    python -m rl.profile_training

Provides:
    1. Component-level timing (env step vs PPO update vs logging)
    2. JAX trace for TensorBoard visualization
    3. Environment-only throughput baseline
"""

import time

import jax
import jax.numpy as jnp

from environment.config import EnvironmentConfig
from environment.environment import TagEnvironment, observation_size
from rl.config import RLConfig
from rl.train import make_train


def time_block(label, fn, warmup=2, repeats=10):
    """Time a function with warmup, using block_until_ready."""
    # Warmup
    for _ in range(warmup):
        result = fn()
        jax.block_until_ready(result)

    # Timed runs
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    print(f"  {label}: {avg*1000:.1f}ms ± {std*1000:.1f}ms")
    return avg


def profile_components():
    print("=" * 60)
    print("COMPONENT-LEVEL PROFILING")
    print("=" * 60)

    rl_config = RLConfig()
    env_config = EnvironmentConfig()
    rng = jax.random.PRNGKey(42)

    timesteps_per_update = rl_config.num_steps * rl_config.num_envs
    print(f"\nConfig: {rl_config.num_envs} envs × {rl_config.num_steps} steps = {timesteps_per_update:,} timesteps/update")
    print(f"Physics substeps per action: {round(1.0 / (env_config.action_frequency * 0.005))}")
    print(f"Rays per agent: {env_config.n_rays}")

    # Build training functions
    init_fn, step_fn, num_updates, env, _, _ = make_train(rl_config, env_config)

    print("\nCompiling init...")
    t0 = time.perf_counter()
    runner_state = jax.jit(init_fn)(rng)
    jax.block_until_ready(runner_state)
    print(f"  Init (incl JIT compile): {time.perf_counter() - t0:.1f}s")

    step_jit = jax.jit(step_fn)

    print("\nCompiling step (first call)...")
    t0 = time.perf_counter()
    runner_state, metrics = step_jit(runner_state)
    jax.block_until_ready(metrics)
    print(f"  First step (incl JIT compile): {time.perf_counter() - t0:.1f}s")

    # Profile full step
    print("\nFull training step (env rollout + PPO update):")
    def full_step():
        nonlocal runner_state
        runner_state, metrics = step_jit(runner_state)
        return metrics

    avg_step = time_block("step_jit", full_step, warmup=3, repeats=20)
    sps = timesteps_per_update / avg_step
    print(f"\n  => {sps:,.0f} env steps/sec")

    # Profile the .item() overhead
    print("\nLogging overhead (.item() calls):")
    runner_state, metrics = step_jit(runner_state)
    jax.block_until_ready(metrics)

    # Time .item() extraction
    t0 = time.perf_counter()
    for _ in range(20):
        runner_state, metrics = step_jit(runner_state)
        for key, value in metrics.items():
            _ = value.item()
    elapsed = time.perf_counter() - t0
    print(f"  step + .item() per metric: {elapsed/20*1000:.1f}ms per update")

    # Compare: step only (no .item())
    t0 = time.perf_counter()
    for _ in range(20):
        runner_state, metrics = step_jit(runner_state)
    jax.block_until_ready(metrics)
    elapsed = time.perf_counter() - t0
    print(f"  step only (block at end):  {elapsed/20*1000:.1f}ms per update")


def profile_env_only():
    print("\n" + "=" * 60)
    print("ENVIRONMENT-ONLY THROUGHPUT")
    print("=" * 60)

    env_config = EnvironmentConfig()

    for n_envs in [256, 512, 1024, 2048, 4096]:
        env = TagEnvironment(env_config, n_envs)
        v_step = jax.vmap(env.step)
        v_reset = jax.vmap(env.reset)

        rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, n_envs)
        states, obs_c, obs_e = v_reset(keys)
        jax.block_until_ready(obs_c)

        actions = jnp.zeros((n_envs, 2))

        # Warmup
        for _ in range(3):
            states, *_ = v_step(states, actions, actions)
            jax.block_until_ready(states.step_count)

        # Benchmark env.step only
        n_iters = 100
        t0 = time.perf_counter()
        for _ in range(n_iters):
            states, *_ = v_step(states, actions, actions)
            jax.block_until_ready(states.step_count)
        elapsed = time.perf_counter() - t0
        sps = (n_iters * n_envs) / elapsed
        print(f"  {n_envs:5d} envs: {sps:>10,.0f} steps/sec  ({elapsed/n_iters*1000:.1f}ms/step)")


def profile_env_components():
    """Break down where time goes inside a single env step."""
    print("\n" + "=" * 60)
    print("ENVIRONMENT STEP BREAKDOWN")
    print("=" * 60)

    env_config = EnvironmentConfig()
    n_envs = 1024
    env = TagEnvironment(env_config, n_envs)

    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, n_envs)
    states, _, _ = jax.vmap(env.reset)(keys)
    jax.block_until_ready(states.step_count)

    actions = jnp.zeros((n_envs, 2))

    # Profile physics substeps only (no obs/rays)
    @jax.jit
    def physics_only(states, actions):
        def single_physics(state, chaser_action, evader_action):
            control_actions = jnp.concatenate([chaser_action, evader_action])
            mjx_data = state.mjx_data.replace(ctrl=control_actions)
            from mujoco import mjx as mjx_mod
            def substep(data, _):
                data = mjx_mod.step(env.mjx_model, data)
                return data, None
            mjx_data, _ = jax.lax.scan(substep, mjx_data, None, length=env.substeps_per_action)
            mjx_data = mjx_mod.forward(env.mjx_model, mjx_data)
            return mjx_data.sensordata
        return jax.vmap(single_physics)(states, actions, actions)

    # Profile ray casting only
    @jax.jit
    def rays_only(states):
        def single_rays(state):
            mjx_data = state.mjx_data
            sensor_data = mjx_data.sensordata
            ss = env.sensor_slices
            from environment.mujoco_data import quaternion_to_yaw
            # Chaser rays
            c_pos = sensor_data[ss.chaser_position]
            c_yaw = quaternion_to_yaw(sensor_data[ss.chaser_quaternion])
            c_dist, c_type = env._cast_rays(env.mjx_model, mjx_data, c_pos, c_yaw)
            # Evader rays
            e_pos = sensor_data[ss.evader_position]
            e_yaw = quaternion_to_yaw(sensor_data[ss.evader_quaternion])
            e_dist, e_type = env._cast_rays(env.mjx_model, mjx_data, e_pos, e_yaw)
            return c_dist, e_dist
        return jax.vmap(single_rays)(states)

    # Profile reset
    @jax.jit
    def reset_only(keys):
        return jax.vmap(env.reset)(keys)

    print("\nPhysics substeps (10 × mjx.step + mjx.forward):")
    time_block("physics", lambda: physics_only(states, actions), warmup=3, repeats=20)

    print("\nRay casting (2 agents × 64 rays):")
    time_block("rays", lambda: rays_only(states), warmup=3, repeats=20)

    rng = jax.random.PRNGKey(1)
    print("\nReset (full reset + mjx.forward):")
    def do_reset():
        nonlocal rng
        rng, k = jax.random.split(rng)
        return reset_only(jax.random.split(k, n_envs))
    time_block("reset", do_reset, warmup=3, repeats=20)


def jax_trace_profile():
    """Generate a JAX trace viewable in TensorBoard."""
    print("\n" + "=" * 60)
    print("JAX TRACE PROFILING")
    print("=" * 60)

    rl_config = RLConfig()
    env_config = EnvironmentConfig()
    rng = jax.random.PRNGKey(42)

    init_fn, step_fn, num_updates, env, _, _ = make_train(rl_config, env_config)
    runner_state = jax.jit(init_fn)(rng)
    step_jit = jax.jit(step_fn)

    # Warmup
    for _ in range(3):
        runner_state, _ = step_jit(runner_state)
    jax.block_until_ready(runner_state)

    trace_dir = "/tmp/jax-trace"
    print(f"\nRecording trace to {trace_dir}")
    print("View with: tensorboard --logdir /tmp/jax-trace")
    jax.profiler.start_trace(trace_dir)
    for _ in range(5):
        runner_state, metrics = step_jit(runner_state)
    jax.block_until_ready(metrics)
    jax.profiler.stop_trace()
    print("Trace saved!")


if __name__ == "__main__":
    import sys

    if "--trace" in sys.argv:
        jax_trace_profile()
    elif "--env" in sys.argv:
        profile_env_only()
    elif "--breakdown" in sys.argv:
        profile_env_components()
    else:
        profile_components()
        profile_env_only()
        profile_env_components()
        print("\n\nFor GPU-level trace: python -m rl.profile_training --trace")
        print("Then: tensorboard --logdir /tmp/jax-trace")
