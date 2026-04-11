from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from runtime import jax_setup as _jax_setup  # noqa: F401

import jax
import numpy as np

from environment.config import EnvironmentConfig
from sysid.cma_es import CMAES
from sysid.dataset import WindowConfig, load_dataset_splits
from sysid.evaluator import SysIdEvaluator
from sysid.params import LATENT_DIM, PARAM_BOUNDS, PARAM_NAMES, latent_to_physical
from sysid.types import EvaluationSummary


def _physical_dict(latent: np.ndarray) -> dict[str, float]:
    params = latent_to_physical(jax.numpy.asarray(latent, dtype=jax.numpy.float32))
    return {name: float(params[name]) for name in PARAM_NAMES}


def _bound_diagnostics(physical_params: dict[str, float]) -> dict[str, object]:
    near_bounds: dict[str, str] = {}
    distance_to_bounds: dict[str, dict[str, float]] = {}
    for name, value in physical_params.items():
        bounds = PARAM_BOUNDS[name]
        span = bounds.high - bounds.low
        lower = float(value - bounds.low)
        upper = float(bounds.high - value)
        distance_to_bounds[name] = {
            "lower": lower,
            "upper": upper,
            "fraction_of_span": min(lower, upper) / span if span > 0 else 0.0,
        }
        threshold = max(0.02 * span, 1e-6)
        if lower <= threshold:
            near_bounds[name] = "lower"
        elif upper <= threshold:
            near_bounds[name] = "upper"
    return {
        "near_bounds": near_bounds,
        "near_bound_count": len(near_bounds),
        "distance_to_bounds": distance_to_bounds,
    }


def _summary_record(summary: EvaluationSummary) -> dict[str, object]:
    return {
        "windows": summary.windows,
        "score": asdict(summary.score),
        "metrics": summary.metrics,
        "maneuver_scores": summary.maneuver_scores,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit robot parameters with CMA-ES")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sysid"))
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--preroll-seconds", type=float, default=0.5)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    window_config = WindowConfig(
        window_seconds=args.window_seconds,
        preroll_seconds=args.preroll_seconds,
        stride_seconds=args.stride_seconds,
    )
    dataset = load_dataset_splits(args.data_root, window_config)
    evaluator = SysIdEvaluator.create(EnvironmentConfig(), window_config.num_substeps)
    train_windows = evaluator.prepare_batch(dataset.train)
    eval_windows = evaluator.prepare_batch(dataset.eval)
    train_population_eval = evaluator.build_population_evaluator(train_windows)
    train_summary_eval = evaluator.build_summary_evaluator(train_windows)
    eval_summary_eval = evaluator.build_summary_evaluator(eval_windows)
    optimizer = CMAES(LATENT_DIM, args.population_size, args.seed)

    best_latent: np.ndarray | None = None
    best_score = float("inf")
    best_train_summary: EvaluationSummary | None = None
    best_eval_summary: EvaluationSummary | None = None
    history_path = args.output_dir / "history.jsonl"
    best_path = args.output_dir / "best_params.json"

    with history_path.open("w", encoding="utf-8") as history_file:
        for generation in range(args.generations):
            population = optimizer.ask().astype(np.float32)
            train_terms = evaluator.evaluate_population(
                jax.numpy.asarray(population, dtype=jax.numpy.float32),
                train_windows,
                population_eval=train_population_eval,
            )
            total_scores = np.asarray(jax.device_get(train_terms["total"]))
            optimizer.tell(population, total_scores)

            best_index = int(np.argmin(total_scores))
            generation_best_latent = population[best_index].astype(np.float64)
            generation_best_score = float(total_scores[best_index])
            generation_train_summary = evaluator.evaluate_summary(
                jax.numpy.asarray(generation_best_latent, dtype=jax.numpy.float32),
                dataset.train,
                windows=train_windows,
                summary_eval=train_summary_eval,
            )
            generation_eval_summary = evaluator.evaluate_summary(
                jax.numpy.asarray(generation_best_latent, dtype=jax.numpy.float32),
                dataset.eval,
                windows=eval_windows,
                summary_eval=eval_summary_eval,
            )

            if generation_best_score < best_score:
                best_score = generation_best_score
                best_latent = generation_best_latent.copy()
                best_train_summary = generation_train_summary
                best_eval_summary = generation_eval_summary

            if (
                best_latent is None
                or best_train_summary is None
                or best_eval_summary is None
            ):
                raise RuntimeError("Missing global best state during optimization")

            generation_params = _physical_dict(generation_best_latent)
            global_best_params = _physical_dict(best_latent)
            record = {
                "generation": generation,
                "population_size": args.population_size,
                "train_population": {
                    "best_total": generation_best_score,
                    "mean_total": float(np.mean(total_scores)),
                    "std_total": float(np.std(total_scores)),
                },
                "generation_best": {
                    "params": generation_params,
                    "bounds": _bound_diagnostics(generation_params),
                    "train_summary": _summary_record(generation_train_summary),
                    "eval_summary": _summary_record(generation_eval_summary),
                },
                "global_best": {
                    "train_total": best_score,
                    "params": global_best_params,
                    "bounds": _bound_diagnostics(global_best_params),
                    "train_summary": _summary_record(best_train_summary),
                    "eval_summary": _summary_record(best_eval_summary),
                },
            }
            history_file.write(json.dumps(record) + "\n")
            history_file.flush()
            print(
                "generation={gen} train_best={train_best:.4f} train_mean={train_mean:.4f} "
                "gen_eval={gen_eval:.4f} global_best={global_best:.4f} "
                "eval_rmse={eval_rmse:.4f}m eval_yaw={eval_yaw:.2f}deg near_bounds={near_bounds} "
                "params={params}".format(
                    gen=generation,
                    train_best=generation_best_score,
                    train_mean=float(np.mean(total_scores)),
                    gen_eval=generation_eval_summary.score.total,
                    global_best=best_score,
                    eval_rmse=generation_eval_summary.metrics["position_rmse_m"],
                    eval_yaw=generation_eval_summary.metrics["yaw_mae_deg"],
                    near_bounds=record["generation_best"]["bounds"]["near_bounds"],
                    params=json.dumps(generation_params, sort_keys=True),
                )
            )

    if best_latent is None or best_train_summary is None or best_eval_summary is None:
        raise RuntimeError("No CMA-ES generations were executed")

    best_record = {
        "best_train_score": best_score,
        "best_params": _physical_dict(best_latent),
        "bounds": _bound_diagnostics(_physical_dict(best_latent)),
        "train_summary": _summary_record(best_train_summary),
        "eval_summary": _summary_record(best_eval_summary),
    }
    best_path.write_text(json.dumps(best_record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
