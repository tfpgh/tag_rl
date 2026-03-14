import argparse
import pathlib
import sys
import time
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from mujoco import mjx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from environment.config import EnvironmentConfig
from environment.environment import TagEnvironment

OptionOverrides = Mapping[str, str | int | float | bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MJX option settings for speed and stability."
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--num-action-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-mode",
        choices=("random", "low_random", "forward", "zero"),
        default="random",
        help="Use random actions to stress the sim, or zero actions for a gentler test.",
    )
    parser.add_argument("--case-set", choices=("quick", "full"), default="quick")
    return parser.parse_args()


def benchmark_cases(
    case_set: str,
) -> list[tuple[str, dict[str, str | int | float | bool]]]:
    quick = [
        ("baseline", {}),
        ("dense", {"jacobian": "dense"}),
        ("newton_8", {"solver": "Newton", "iterations": 8, "ls_iterations": 8}),
        (
            "newton_8_dense",
            {
                "solver": "Newton",
                "iterations": 8,
                "ls_iterations": 8,
                "jacobian": "dense",
            },
        ),
        ("newton_6", {"solver": "Newton", "iterations": 6, "ls_iterations": 8}),
        (
            "newton_6_dense",
            {
                "solver": "Newton",
                "iterations": 6,
                "ls_iterations": 8,
                "jacobian": "dense",
            },
        ),
        ("newton_4", {"solver": "Newton", "iterations": 4, "ls_iterations": 6}),
        (
            "newton_4_dense",
            {
                "solver": "Newton",
                "iterations": 4,
                "ls_iterations": 6,
                "jacobian": "dense",
            },
        ),
        ("newton_2", {"solver": "Newton", "iterations": 2, "ls_iterations": 4}),
        (
            "newton_2_dense",
            {
                "solver": "Newton",
                "iterations": 2,
                "ls_iterations": 4,
                "jacobian": "dense",
            },
        ),
        (
            "newton_1_dense",
            {
                "solver": "Newton",
                "iterations": 1,
                "ls_iterations": 4,
                "jacobian": "dense",
            },
        ),
        ("dt_0p006", {"timestep": 0.006}),
        ("dt_0p008", {"timestep": 0.008}),
        (
            "dt_0p008_newton_2_dense",
            {
                "timestep": 0.008,
                "solver": "Newton",
                "iterations": 2,
                "ls_iterations": 4,
                "jacobian": "dense",
            },
        ),
    ]
    if case_set == "quick":
        return quick
    return quick + [
        (
            "newton_1_dense_dt_0p006",
            {
                "timestep": 0.006,
                "solver": "Newton",
                "iterations": 1,
                "ls_iterations": 4,
                "jacobian": "dense",
            },
        ),
        (
            "newton_1_dense_dt_0p01",
            {
                "timestep": 0.01,
                "solver": "Newton",
                "iterations": 1,
                "ls_iterations": 4,
                "jacobian": "dense",
            },
        ),
        ("cg_dense", {"solver": "CG", "iterations": 10, "jacobian": "dense"}),
    ]


def build_action_sequence(
    num_action_steps: int,
    num_envs: int,
    seed: int,
    action_mode: str,
) -> jax.Array:
    if action_mode == "zero":
        return jnp.zeros((num_action_steps, num_envs, 4), dtype=jnp.float32)
    if action_mode == "forward":
        return jnp.full((num_action_steps, num_envs, 4), 0.05, dtype=jnp.float32)
    if action_mode == "low_random":
        return jax.random.uniform(
            jax.random.PRNGKey(seed),
            (num_action_steps, num_envs, 4),
            minval=-0.2,
            maxval=0.2,
            dtype=jnp.float32,
        )
    return jax.random.uniform(
        jax.random.PRNGKey(seed),
        (num_action_steps, num_envs, 4),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )


def make_initial_state(
    env: TagEnvironment, num_envs: int, seed: int
) -> tuple[jax.Array, jax.Array]:
    reset_rngs = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    qpos, qvel, *_ = jax.vmap(env.reset_state)(reset_rngs)
    return qpos, qvel


def make_rollout(env: TagEnvironment):
    zero_ctrl = jnp.zeros((4,), dtype=jnp.float32)
    chaser_root = env.joint_qpos_slices.chaser_root
    evader_root = env.joint_qpos_slices.evader_root

    @jax.jit
    def rollout(
        qpos_batch: jax.Array,
        qvel_batch: jax.Array,
        action_sequence: jax.Array,
    ) -> dict[str, jax.Array]:
        def init_one(qpos: jax.Array, qvel: jax.Array) -> mjx.Data:
            data = env._template_mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=zero_ctrl)
            return env.forward(data)

        data = jax.vmap(init_one)(qpos_batch, qvel_batch)

        def action_step(
            carry: tuple[mjx.Data, jax.Array, jax.Array, jax.Array],
            ctrl_batch: jax.Array,
        ) -> tuple[tuple[mjx.Data, jax.Array, jax.Array, jax.Array], None]:
            data, all_finite, min_height, max_abs_qvel = carry

            def one_env_step(one_data: mjx.Data, one_ctrl: jax.Array) -> mjx.Data:
                one_data = one_data.replace(ctrl=one_ctrl)

                def substep(d: mjx.Data, _: None) -> tuple[mjx.Data, None]:
                    return mjx.step(env.mjx_model, d), None

                new_data, _ = jax.lax.scan(
                    substep,
                    one_data,
                    None,
                    length=env.substeps_per_action,
                )
                return new_data

            data = jax.vmap(one_env_step)(data, ctrl_batch)
            finite_now = jnp.all(jnp.isfinite(data.qpos), axis=1) & jnp.all(
                jnp.isfinite(data.qvel), axis=1
            )
            root_heights = jnp.stack(
                [
                    data.qpos[:, chaser_root.start + 2],
                    data.qpos[:, evader_root.start + 2],
                ],
                axis=1,
            )
            min_height = jnp.minimum(min_height, jnp.min(root_heights))
            max_abs_qvel = jnp.maximum(max_abs_qvel, jnp.max(jnp.abs(data.qvel)))
            return (data, all_finite & finite_now, min_height, max_abs_qvel), None

        initial_finite = jnp.ones((qpos_batch.shape[0],), dtype=jnp.bool_)
        initial_min_height = jnp.float32(jnp.inf)
        initial_max_abs_qvel = jnp.float32(0.0)
        (data, all_finite, min_height, max_abs_qvel), _ = jax.lax.scan(
            action_step,
            (data, initial_finite, initial_min_height, initial_max_abs_qvel),
            action_sequence,
        )

        chaser_qpos = data.qpos[:, chaser_root]
        evader_qpos = data.qpos[:, evader_root]
        return {
            "all_finite": all_finite,
            "chaser_qpos": chaser_qpos,
            "evader_qpos": evader_qpos,
            "mean_chaser_height": jnp.mean(chaser_qpos[:, 2]),
            "mean_evader_height": jnp.mean(evader_qpos[:, 2]),
            "min_height": min_height,
            "max_abs_qvel": max_abs_qvel,
        }

    return rollout


def root_rmse(current: jax.Array, reference: jax.Array) -> float:
    return float(jnp.sqrt(jnp.mean(jnp.square(current - reference))))


def run_case(
    name: str,
    overrides: OptionOverrides,
    config: EnvironmentConfig,
    num_envs: int,
    qpos0: jax.Array,
    qvel0: jax.Array,
    action_sequence: jax.Array,
    baseline_outputs: dict[str, jax.Array] | None,
) -> tuple[dict[str, float | str | bool | int], dict[str, jax.Array]]:
    env = TagEnvironment(config, num_envs, mjcf_option_overrides=overrides)
    rollout = make_rollout(env)

    t0 = time.time()
    outputs = rollout(qpos0, qvel0, action_sequence)
    jax.block_until_ready(outputs["chaser_qpos"])
    compile_plus_first_run_s = time.time() - t0

    t1 = time.time()
    outputs = rollout(qpos0, qvel0, action_sequence)
    jax.block_until_ready(outputs["chaser_qpos"])
    run_s = time.time() - t1

    action_steps_per_s = num_envs * action_sequence.shape[0] / run_s
    physics_steps_per_s = action_steps_per_s * env.substeps_per_action
    all_finite = bool(jnp.all(outputs["all_finite"]))
    finite_fraction = float(jnp.mean(outputs["all_finite"].astype(jnp.float32)))

    result: dict[str, float | str | bool | int] = {
        "case": name,
        "dt": float(env.mj_model.opt.timestep),
        "substeps": env.substeps_per_action,
        "compile_plus_first_run_s": round(compile_plus_first_run_s, 2),
        "run_s": round(run_s, 2),
        "action_steps_per_s": round(action_steps_per_s, 1),
        "physics_steps_per_s": round(physics_steps_per_s, 1),
        "all_finite": all_finite,
        "finite_fraction": round(finite_fraction, 4),
        "mean_chaser_height": round(float(outputs["mean_chaser_height"]), 4),
        "mean_evader_height": round(float(outputs["mean_evader_height"]), 4),
        "min_height": round(float(outputs["min_height"]), 4),
        "max_abs_qvel": round(float(outputs["max_abs_qvel"]), 4),
    }
    if baseline_outputs is not None:
        result["chaser_root_rmse_vs_baseline"] = round(
            root_rmse(outputs["chaser_qpos"], baseline_outputs["chaser_qpos"]), 5
        )
        result["evader_root_rmse_vs_baseline"] = round(
            root_rmse(outputs["evader_qpos"], baseline_outputs["evader_qpos"]), 5
        )
    return result, outputs


def main() -> None:
    args = parse_args()
    print(f"JAX devices: {[str(device) for device in jax.devices()]}")
    print(
        f"Benchmarking {args.num_envs} envs for {args.num_action_steps} action steps using {args.action_mode} actions"
    )

    config = EnvironmentConfig()
    baseline_env = TagEnvironment(config, args.num_envs)
    qpos0, qvel0 = make_initial_state(baseline_env, args.num_envs, args.seed)
    action_sequence = build_action_sequence(
        args.num_action_steps,
        args.num_envs,
        args.seed + 1,
        args.action_mode,
    )

    cases = benchmark_cases(args.case_set)
    baseline_result, baseline_outputs = run_case(
        cases[0][0],
        cases[0][1],
        config,
        args.num_envs,
        qpos0,
        qvel0,
        action_sequence,
        None,
    )
    print(baseline_result)

    for name, overrides in cases[1:]:
        result, _ = run_case(
            name,
            overrides,
            config,
            args.num_envs,
            qpos0,
            qvel0,
            action_sequence,
            baseline_outputs,
        )
        print(result)


if __name__ == "__main__":
    main()
