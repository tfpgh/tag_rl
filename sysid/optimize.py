from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sysid.load_runs import discover_run_dirs, load_run
from sysid.replay import NominalParameters, evaluate_run, summarize_metrics
from sysid.types import ReplayMetrics, RunData

BOUNDS = {
    "floor_friction_scale": (0.9, 1.15),
    "wheel_friction_scale": (0.85, 1.2),
    "wheel_damping_scale": (0.75, 1.35),
    "wheel_frictionloss_scale": (0.75, 1.35),
    "motor_strength_scale": (0.75, 1.3),
    "motor_balance": (-0.2, 0.2),
}
PARAMETER_NAMES = tuple(BOUNDS.keys())


@dataclass(frozen=True, slots=True)
class CandidateResult:
    params: NominalParameters
    train_metrics: ReplayMetrics
    validation_metrics: ReplayMetrics


@dataclass(frozen=True, slots=True)
class SearchResult:
    best: CandidateResult
    history: list[dict[str, object]]


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


def _vector_from_params(params: NominalParameters) -> np.ndarray:
    return np.array(
        [getattr(params, name) for name in PARAMETER_NAMES], dtype=np.float64
    )


def _params_from_vector(
    vector: np.ndarray, action_delay_steps: int
) -> NominalParameters:
    payload = {
        name: float(value) for name, value in zip(PARAMETER_NAMES, vector, strict=True)
    }
    return NominalParameters(action_delay_steps=action_delay_steps, **payload)


def _clip_to_bounds(vector: np.ndarray) -> np.ndarray:
    clipped = vector.copy()
    for index, name in enumerate(PARAMETER_NAMES):
        low, high = BOUNDS[name]
        clipped[index] = np.clip(clipped[index], low, high)
    return clipped


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


def _evaluate_candidate(
    run_sets: tuple[list[RunData], list[RunData]], params: NominalParameters
) -> CandidateResult:
    train_runs, validation_runs = run_sets
    train_metrics = [evaluate_run(run, params) for run in train_runs]
    validation_metrics = [evaluate_run(run, params) for run in validation_runs]
    train_mean = _mean_metrics(train_metrics)
    validation_mean = _mean_metrics(validation_metrics)
    return CandidateResult(params, train_mean, validation_mean)


def _cross_entropy_search(
    run_sets: tuple[list[RunData], list[RunData]],
    action_delay_steps: int,
    seed: int,
    generations: int,
    population_size: int,
    elite_count: int,
    quiet: bool,
) -> SearchResult:
    rng = np.random.default_rng(seed)
    lows = np.array([BOUNDS[name][0] for name in PARAMETER_NAMES], dtype=np.float64)
    highs = np.array([BOUNDS[name][1] for name in PARAMETER_NAMES], dtype=np.float64)
    mean = _vector_from_params(NominalParameters())
    std = 0.25 * (highs - lows)

    best_result: CandidateResult | None = None
    history: list[dict[str, object]] = []
    started_at = time.perf_counter()
    for generation in range(generations):
        generation_started_at = time.perf_counter()
        raw_samples = rng.normal(
            loc=mean, scale=std, size=(population_size, len(PARAMETER_NAMES))
        )
        candidates = np.vstack([_clip_to_bounds(sample) for sample in raw_samples])
        results = [
            _evaluate_candidate(
                run_sets, _params_from_vector(candidate, action_delay_steps)
            )
            for candidate in candidates
        ]
        ranked = sorted(
            results,
            key=lambda item: (
                item.validation_metrics.score,
                item.train_metrics.score,
            ),
        )
        elites = ranked[:elite_count]
        elite_vectors = np.vstack([_vector_from_params(item.params) for item in elites])
        mean = elite_vectors.mean(axis=0)
        std = np.maximum(elite_vectors.std(axis=0), 0.05 * (highs - lows))
        best_generation = elites[0]
        history.append(
            {
                "generation": generation,
                "train": summarize_metrics(best_generation.train_metrics),
                "validation": summarize_metrics(best_generation.validation_metrics),
                "params": asdict(best_generation.params),
            }
        )
        elapsed = time.perf_counter() - started_at
        completed = generation + 1
        eta = (elapsed / completed) * (generations - completed) if completed else 0.0
        generation_duration = time.perf_counter() - generation_started_at
        _log(
            (
                f"delay={action_delay_steps} gen={completed}/{generations} "
                f"best_val={best_generation.validation_metrics.score:.4f} "
                f"gen_time={_format_duration(generation_duration)} "
                f"eta={_format_duration(eta)}"
            ),
            quiet,
        )
        if (
            best_result is None
            or best_generation.validation_metrics.score
            < best_result.validation_metrics.score
        ):
            best_result = best_generation

    assert best_result is not None
    return SearchResult(best_result, history)


def fit_nominal_parameters(
    train_runs: list[RunData],
    validation_runs: list[RunData],
    generations: int = 12,
    population_size: int = 32,
    elite_count: int = 8,
    seed: int = 0,
    quiet: bool = False,
) -> SearchResult:
    best: SearchResult | None = None
    run_sets = (train_runs, validation_runs)
    for action_delay_steps in range(3):
        _log(f"starting delay sweep {action_delay_steps}/2", quiet)
        result = _cross_entropy_search(
            run_sets,
            action_delay_steps=action_delay_steps,
            seed=seed + action_delay_steps,
            generations=generations,
            population_size=population_size,
            elite_count=elite_count,
            quiet=quiet,
        )
        if (
            best is None
            or result.best.validation_metrics.score < best.best.validation_metrics.score
        ):
            best = result
    assert best is not None
    return best


def _split_runs(run_dirs: list[str]) -> tuple[list[RunData], list[RunData]]:
    runs = [load_run(path) for path in discover_run_dirs(run_dirs)]
    if len(runs) <= 1:
        return runs, runs
    validation_count = max(1, len(runs) // 4)
    return runs[:-validation_count], runs[-validation_count:]


def _recommend_config_update(
    best_params: NominalParameters, train_runs: list[RunData]
) -> dict[str, object]:
    from sysid.stats import summarize_run

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
            "position_noise_std_max": max(position_noise) if position_noise else 0.0,
            "yaw_noise_std_max": max(yaw_noise) if yaw_noise else 0.0,
            "frame_drop_probability_max": min(
                max(frame_drop_estimate * 1.5, 0.0), 0.25
            ),
        },
        "notes": [
            "Set nominal dynamics to the fitted values before widening randomization ranges.",
            "Keep observation delay and stale-observation settings unchanged until raw/stale logging exists.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_runs, validation_runs = _split_runs(args.run_dirs)
    result = fit_nominal_parameters(
        train_runs,
        validation_runs,
        generations=args.generations,
        population_size=args.population_size,
        elite_count=args.elite_count,
        seed=args.seed,
        quiet=args.quiet,
    )
    best_params = result.best.params
    output = {
        "train_runs": [str(run.run_dir) for run in train_runs],
        "validation_runs": [str(run.run_dir) for run in validation_runs],
        "best_params": asdict(best_params),
        "train_metrics": summarize_metrics(result.best.train_metrics),
        "validation_metrics": summarize_metrics(result.best.validation_metrics),
        "history": result.history,
        "recommended_config_update": _recommend_config_update(best_params, train_runs),
    }
    payload = json.dumps(output, indent=2, sort_keys=True)
    if args.output is not None:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
