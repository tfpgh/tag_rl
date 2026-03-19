from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
from evosax.algorithms import Sep_CMA_ES

from sysid.dataset import DatasetConfig, build_prepared_dataset
from sysid.load_runs import discover_run_dirs, load_run
from sysid.replay import (
    decoded_values_to_params,
    make_dataset_evaluator,
    summarize_metrics,
)
from sysid.stats import summarize_run
from sysid.types import PreparedDataset, ReplayMetrics, RunData

LATENT_DIM = 8


def _replace_std_init(params: Any, std_init: float) -> Any:
    return params.replace(std_init=std_init)


def _format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {remaining:.0f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m"


def _log(message: str, quiet: bool) -> None:
    if not quiet:
        print(message, flush=True)


def _split_runs(run_dirs: list[str]) -> tuple[list[RunData], list[RunData]]:
    runs = [load_run(path) for path in discover_run_dirs(run_dirs)]
    if len(runs) <= 1:
        return runs, runs
    validation_count = max(1, len(runs) // 4)
    return runs[:-validation_count], runs[-validation_count:]


def _mean_metrics(metrics_list: list[ReplayMetrics]) -> ReplayMetrics:
    if not metrics_list:
        return ReplayMetrics(0.0, 0.0, 0.0, 0.0, float("inf"), 0)
    return ReplayMetrics(
        position_rmse_m=sum(item.position_rmse_m for item in metrics_list)
        / len(metrics_list),
        yaw_rmse_rad=sum(item.yaw_rmse_rad for item in metrics_list)
        / len(metrics_list),
        endpoint_position_error_m=sum(
            item.endpoint_position_error_m for item in metrics_list
        )
        / len(metrics_list),
        endpoint_yaw_error_rad=sum(item.endpoint_yaw_error_rad for item in metrics_list)
        / len(metrics_list),
        score=sum(item.score for item in metrics_list) / len(metrics_list),
        sample_count=sum(item.sample_count for item in metrics_list),
    )


def _decode_solution(evaluator, solution: jax.Array):
    values, action_delays, observation_delays = evaluator.decode_population(
        solution[None, :]
    )
    return decoded_values_to_params(values[0], action_delays[0], observation_delays[0])


def _recommend_config_update(
    best_params, train_runs: list[RunData]
) -> dict[str, object]:
    summaries = [summarize_run(run) for run in train_runs]
    position_noise = [summary["noise"]["position_noise_std_m"] for summary in summaries]
    yaw_noise = [summary["noise"]["yaw_noise_std_rad"] for summary in summaries]
    visibility = [summary["visibility"]["visible_fraction"] for summary in summaries]
    frame_drop_estimate = (
        max(0.0, 1.0 - sum(visibility) / len(visibility)) if visibility else 0.0
    )
    return {
        "nominal": asdict(best_params),
        "pipeline_randomization": {
            "max_action_delay_steps": int(best_params.action_delay_steps + 1),
            "max_observation_delay_steps": int(best_params.observation_delay_steps + 1),
            "position_noise_std_max": max(position_noise) if position_noise else 0.0,
            "yaw_noise_std_max": max(yaw_noise) if yaw_noise else 0.0,
            "frame_drop_probability_max": min(
                max(frame_drop_estimate * 1.5, 0.0), 0.25
            ),
        },
        "notes": [
            "Set nominal dynamics to the fitted values before widening randomization ranges.",
            "Observation delay is now fit directly; stale-observation behavior still needs explicit logging to fit well.",
        ],
    }


def _dataset_summary(dataset: PreparedDataset) -> dict[str, object]:
    return {
        "role": dataset.role,
        "segment_count": int(dataset.initial_poses.shape[0]),
        "segment_steps": int(dataset.controls.shape[1]),
        "run_names": list(dataset.run_names),
    }


def run_search(
    train_runs: list[RunData],
    validation_runs: list[RunData],
    dataset_config: DatasetConfig,
    population_size: int,
    generations: int,
    seed: int,
    std_init: float,
    quiet: bool,
) -> dict[str, object]:
    train_dataset = build_prepared_dataset(train_runs, config=dataset_config)
    validation_dataset = build_prepared_dataset(validation_runs, config=dataset_config)
    train_evaluator = make_dataset_evaluator(train_dataset)
    validation_evaluator = make_dataset_evaluator(validation_dataset)

    strategy = Sep_CMA_ES(
        population_size=population_size,
        solution=jnp.zeros((LATENT_DIM,), dtype=jnp.float32),
    )
    es_params = _replace_std_init(strategy.default_params, std_init)
    key = jax.random.key(seed)
    init_key, key = jax.random.split(key)
    state = strategy.init(
        init_key, jnp.zeros((LATENT_DIM,), dtype=jnp.float32), es_params
    )

    best_record: dict[str, Any] | None = None
    history: list[dict[str, object]] = []
    started_at = time.perf_counter()

    for generation in range(generations):
        generation_started_at = time.perf_counter()
        ask_key, tell_key, key = jax.random.split(key, 3)
        population, state = strategy.ask(ask_key, state, es_params)
        fitness = train_evaluator.evaluate_population_fitness(population)
        state, metrics = strategy.tell(tell_key, population, fitness, state, es_params)

        best_solution = cast(jax.Array, metrics["best_solution"])
        params = _decode_solution(train_evaluator, best_solution)
        train_metrics = train_evaluator.evaluate_population(best_solution[None, :])[0]
        validation_metrics = validation_evaluator.evaluate_population(
            best_solution[None, :]
        )[0]

        record = {
            "generation": generation,
            "params": asdict(params),
            "train": summarize_metrics(train_metrics),
            "validation": summarize_metrics(validation_metrics),
        }
        history.append(record)

        if (
            best_record is None
            or validation_metrics.score
            < cast(ReplayMetrics, best_record["validation_metrics"]).score
        ):
            best_record = {
                "params": params,
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
            }

        elapsed = time.perf_counter() - started_at
        completed = generation + 1
        eta = (elapsed / completed) * (generations - completed) if completed else 0.0
        _log(
            (
                f"gen={completed}/{generations} best_train={train_metrics.score:.4f} "
                f"best_val={validation_metrics.score:.4f} "
                f"delay={params.action_delay_steps}/{params.observation_delay_steps} "
                f"gen_time={_format_duration(time.perf_counter() - generation_started_at)} "
                f"eta={_format_duration(eta)}"
            ),
            quiet,
        )

    assert best_record is not None
    return {
        "train_dataset": _dataset_summary(train_dataset),
        "validation_dataset": _dataset_summary(validation_dataset),
        "best_params": asdict(cast(Any, best_record["params"])),
        "train_metrics": summarize_metrics(
            cast(ReplayMetrics, best_record["train_metrics"])
        ),
        "validation_metrics": summarize_metrics(
            cast(ReplayMetrics, best_record["validation_metrics"])
        ),
        "history": history,
        "recommended_config_update": _recommend_config_update(
            cast(Any, best_record["params"]), train_runs
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--std-init", type=float, default=0.75)
    parser.add_argument("--max-segment-steps", type=int, default=160)
    parser.add_argument("--min-segment-steps", type=int, default=12)
    parser.add_argument("--min-segment-duration-s", type=float, default=0.6)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = DatasetConfig(
        min_segment_duration_s=args.min_segment_duration_s,
        min_segment_steps=args.min_segment_steps,
        max_segment_steps=args.max_segment_steps,
    )
    train_runs, validation_runs = _split_runs(args.run_dirs)
    output = run_search(
        train_runs,
        validation_runs,
        dataset_config=dataset_config,
        population_size=args.population_size,
        generations=args.generations,
        seed=args.seed,
        std_init=args.std_init,
        quiet=args.quiet,
    )
    payload = json.dumps(output, indent=2, sort_keys=True)
    if args.output is not None:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
