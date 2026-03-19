from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
from evosax.algorithms import CMA_ES

from sysid.dataset import DatasetConfig, build_prepared_dataset
from sysid.load_runs import discover_run_dirs, load_run
from sysid.replay import (
    decoded_values_to_params,
    make_dataset_evaluator,
    summarize_metrics,
)
from sysid.stats import summarize_run
from sysid.types import PreparedDataset, ReplayMetrics, RunData

LATENT_DIM = 13
CONSOLE_PARAM_NAMES = (
    "motor_strength_scale",
    "wheel_damping_scale",
    "wheel_frictionloss_scale",
    "floor_friction_scale",
)


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


def _decode_mean(evaluator, mean: jax.Array):
    return _decode_solution(evaluator, mean)


def _normalized_param_position(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    if upper <= lower:
        return 0.5
    return (value - lower) / (upper - lower)


def _bound_proximity(
    value: float, bounds: tuple[float, float]
) -> dict[str, float | bool]:
    lower, upper = bounds
    if upper <= lower:
        return {
            "normalized": 0.5,
            "distance_to_lower": 0.0,
            "distance_to_upper": 0.0,
            "near_lower": False,
            "near_upper": False,
        }
    normalized = _normalized_param_position(value, bounds)
    return {
        "normalized": normalized,
        "distance_to_lower": value - lower,
        "distance_to_upper": upper - value,
        "near_lower": normalized <= 0.05,
        "near_upper": normalized >= 0.95,
    }


def _console_param_summary(params: Any) -> str:
    parts = []
    for name in CONSOLE_PARAM_NAMES:
        parts.append(f"{name.split('_')[0]}={getattr(params, name):.3f}")
    return " ".join(parts)


def _params_bound_summary(
    params: Any, search_bounds: dict[str, tuple[float, float]]
) -> dict[str, dict[str, float | bool]]:
    summary: dict[str, dict[str, float | bool]] = {}
    for name, value in asdict(params).items():
        bounds = search_bounds.get(name)
        if bounds is None:
            continue
        summary[name] = _bound_proximity(float(value), bounds)
    return summary


def _console_bound_hits(bound_summary: dict[str, dict[str, float | bool]]) -> str:
    hits: list[str] = []
    for name, info in bound_summary.items():
        if info["near_lower"]:
            hits.append(f"{name}=low")
        elif info["near_upper"]:
            hits.append(f"{name}=high")
    if not hits:
        return "bounds=none"
    return "bounds=" + ",".join(hits)


def _parameter_summary(
    history: list[dict[str, object]],
    best_params: Any,
    mean_params: Any,
    search_bounds: dict[str, tuple[float, float]],
    best_generation: int,
) -> dict[str, object]:
    param_names = list(asdict(best_params).keys())
    summary: dict[str, object] = {}
    tail_count = max(1, len(history) // 5)
    tail_history = history[-tail_count:]
    for name in param_names:
        best_value = float(getattr(best_params, name))
        mean_value = float(getattr(mean_params, name))
        trajectory = [
            float(cast(dict[str, Any], record["params"])[name]) for record in history
        ]
        tail_trajectory = [
            float(cast(dict[str, Any], record["params"])[name])
            for record in tail_history
        ]
        bounds = search_bounds.get(name)
        proximity = (
            _bound_proximity(best_value, bounds)
            if bounds is not None
            else {"normalized": None}
        )
        late_span = (
            max(tail_trajectory) - min(tail_trajectory) if tail_trajectory else 0.0
        )
        total_span = max(trajectory) - min(trajectory) if trajectory else 0.0
        summary[name] = {
            "best_value": best_value,
            "mean_value": mean_value,
            "initial_value": trajectory[0],
            "final_value": trajectory[-1],
            "min_visited": min(trajectory),
            "max_visited": max(trajectory),
            "total_span": total_span,
            "late_span": late_span,
            "best_generation": best_generation,
            "stability": "stable"
            if late_span <= 0.02 * max(total_span, 1e-6)
            else "moving",
            "bound_info": proximity,
        }
    return summary


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
            "max_action_delay_substeps": int(best_params.action_delay_substeps + 1),
            "max_observation_delay_substeps": int(
                best_params.observation_delay_substeps + 1
            ),
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
    segment_label_counts: dict[str, int] = {}
    for label in dataset.segment_labels:
        segment_label_counts[label] = segment_label_counts.get(label, 0) + 1
    unique_run_names = sorted(set(dataset.run_names))
    return {
        "role": dataset.role,
        "run_count": len(unique_run_names),
        "run_names": unique_run_names,
        "segment_count": int(dataset.initial_poses.shape[0]),
        "segment_steps": int(dataset.controls.shape[1]),
        "segment_label_counts": segment_label_counts,
    }


def _subset_dataset(dataset: PreparedDataset, label: str) -> PreparedDataset | None:
    indices = [
        index for index, value in enumerate(dataset.segment_labels) if value == label
    ]
    if not indices:
        return None
    selection = jnp.asarray(indices, dtype=jnp.int32)
    return PreparedDataset(
        initial_poses=dataset.initial_poses[selection],
        controls=dataset.controls[selection],
        references=dataset.references[selection],
        mask=dataset.mask[selection],
        role=dataset.role,
        run_names=tuple(dataset.run_names[index] for index in indices),
        segment_labels=tuple(dataset.segment_labels[index] for index in indices),
    )


def _per_label_metrics(
    dataset: PreparedDataset, best_solution: jax.Array
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for label in sorted(set(dataset.segment_labels)):
        subset = _subset_dataset(dataset, label)
        if subset is None:
            continue
        evaluator = make_dataset_evaluator(subset)
        metrics = evaluator.evaluate_population(best_solution[None, :])[0]
        results[label] = summarize_metrics(metrics)
    return results


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

    strategy = CMA_ES(
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
        mean_params = _decode_mean(train_evaluator, cast(jax.Array, state.mean))
        param_bound_summary = _params_bound_summary(
            params, train_evaluator.search_bounds
        )
        mean_param_bound_summary = _params_bound_summary(
            mean_params, train_evaluator.search_bounds
        )
        train_metrics = train_evaluator.evaluate_population(best_solution[None, :])[0]
        validation_metrics = validation_evaluator.evaluate_population(
            best_solution[None, :]
        )[0]
        best_so_far = float(metrics["best_fitness"])
        best_in_generation = float(metrics["best_fitness_in_generation"])
        train_val_gap = validation_metrics.score - train_metrics.score

        record = {
            "generation": generation,
            "params": asdict(params),
            "params_bound_summary": param_bound_summary,
            "mean_params": asdict(mean_params),
            "mean_params_bound_summary": mean_param_bound_summary,
            "train": summarize_metrics(train_metrics),
            "validation": summarize_metrics(validation_metrics),
            "best_fitness": best_so_far,
            "best_fitness_in_generation": best_in_generation,
            "train_val_gap": train_val_gap,
            "sigma": float(state.std),
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
                "best_solution": best_solution,
                "mean_params": mean_params,
                "best_generation": generation,
                "params_bound_summary": param_bound_summary,
            }

        elapsed = time.perf_counter() - started_at
        completed = generation + 1
        eta = (elapsed / completed) * (generations - completed) if completed else 0.0
        _log(
            (
                f"gen={completed}/{generations} best_train={train_metrics.score:.4f} "
                f"best_val={validation_metrics.score:.4f} "
                f"delay={params.action_delay_substeps}/{params.observation_delay_substeps} "
                f"sigma={float(state.std):.3f} gap={train_val_gap:+.4f} "
                f"best={best_so_far:.4f}/{best_in_generation:.4f} "
                f"{_console_param_summary(params)} "
                f"{_console_bound_hits(param_bound_summary)} "
                f"gen_time={_format_duration(time.perf_counter() - generation_started_at)} "
                f"eta={_format_duration(eta)}"
            ),
            quiet,
        )

    assert best_record is not None
    return {
        "search_bounds": train_evaluator.search_bounds,
        "device_count": train_evaluator.device_count,
        "train_dataset": _dataset_summary(train_dataset),
        "validation_dataset": _dataset_summary(validation_dataset),
        "best_params": asdict(cast(Any, best_record["params"])),
        "best_generation": int(best_record["best_generation"]),
        "best_mean_params": asdict(cast(Any, best_record["mean_params"])),
        "best_params_bound_summary": cast(Any, best_record["params_bound_summary"]),
        "train_metrics": summarize_metrics(
            cast(ReplayMetrics, best_record["train_metrics"])
        ),
        "validation_metrics": summarize_metrics(
            cast(ReplayMetrics, best_record["validation_metrics"])
        ),
        "train_metrics_by_label": _per_label_metrics(
            train_dataset, cast(jax.Array, best_record["best_solution"])
        ),
        "validation_metrics_by_label": _per_label_metrics(
            validation_dataset, cast(jax.Array, best_record["best_solution"])
        ),
        "history": history,
        "recommended_config_update": _recommend_config_update(
            cast(Any, best_record["params"]), train_runs
        ),
        "parameter_summary": _parameter_summary(
            history,
            cast(Any, best_record["params"]),
            cast(Any, best_record["mean_params"]),
            train_evaluator.search_bounds,
            int(best_record["best_generation"]),
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
    parser.add_argument("--transition-duration-s", type=float, default=0.25)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = DatasetConfig(
        min_segment_duration_s=args.min_segment_duration_s,
        transition_duration_s=args.transition_duration_s,
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
