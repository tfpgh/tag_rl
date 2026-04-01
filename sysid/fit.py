from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from environment.config import EnvironmentConfig
from sysid.cma_es import CMAES
from sysid.dataset import WindowConfig, load_dataset_splits
from sysid.evaluator import SysIdEvaluator
from sysid.params import LATENT_DIM, PARAM_NAMES, latent_to_physical


def _physical_dict(latent: np.ndarray) -> dict[str, float]:
    params = latent_to_physical(jax.numpy.asarray(latent, dtype=jax.numpy.float32))
    return {name: float(params[name]) for name in PARAM_NAMES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit robot parameters with CMA-ES")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sysid"))
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
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
    history_path = args.output_dir / "history.jsonl"
    best_path = args.output_dir / "best_params.json"

    with history_path.open("w", encoding="utf-8") as history_file:
        for generation in range(args.generations):
            population = optimizer.ask().astype(np.float32)
            train_terms = evaluator.evaluate_population(
                population, train_windows, population_eval=train_population_eval
            )
            total_scores = np.asarray(jax.device_get(train_terms["total"]))
            optimizer.tell(
                population.astype(np.float64), total_scores.astype(np.float64)
            )
            best_index = int(np.argmin(total_scores))
            generation_best_latent = population[best_index].astype(np.float64)
            generation_best_score = float(total_scores[best_index])
            if generation_best_score < best_score:
                best_score = generation_best_score
                best_latent = generation_best_latent.copy()

            train_summary = evaluator.evaluate_summary(
                jax.numpy.asarray(generation_best_latent, dtype=jax.numpy.float32),
                dataset.train,
                windows=train_windows,
                summary_eval=train_summary_eval,
            )
            eval_summary = evaluator.evaluate_summary(
                jax.numpy.asarray(generation_best_latent, dtype=jax.numpy.float32),
                dataset.eval,
                windows=eval_windows,
                summary_eval=eval_summary_eval,
            )
            record = {
                "generation": generation,
                "population_size": args.population_size,
                "train_population": {
                    "best_total": generation_best_score,
                    "mean_total": float(np.mean(total_scores)),
                    "std_total": float(np.std(total_scores)),
                },
                "train_best_summary": {
                    "windows": train_summary.windows,
                    "score": asdict(train_summary.score),
                    "maneuver_scores": train_summary.maneuver_scores,
                },
                "eval_best_summary": {
                    "windows": eval_summary.windows,
                    "score": asdict(eval_summary.score),
                    "maneuver_scores": eval_summary.maneuver_scores,
                },
                "best_params": _physical_dict(generation_best_latent),
            }
            history_file.write(json.dumps(record) + "\n")
            history_file.flush()
            print(
                "generation={gen} train_best={train_best:.4f} train_mean={train_mean:.4f} "
                "train_pos={train_pos:.4f} train_yaw={train_yaw:.4f} "
                "eval_total={eval_total:.4f} eval_pos={eval_pos:.4f} eval_yaw={eval_yaw:.4f} "
                "params={params}".format(
                    gen=generation,
                    train_best=generation_best_score,
                    train_mean=float(np.mean(total_scores)),
                    train_pos=train_summary.score.position,
                    train_yaw=train_summary.score.yaw,
                    eval_total=eval_summary.score.total,
                    eval_pos=eval_summary.score.position,
                    eval_yaw=eval_summary.score.yaw,
                    params=json.dumps(
                        _physical_dict(generation_best_latent), sort_keys=True
                    ),
                )
            )

        if best_latent is None:
            raise RuntimeError("No CMA-ES generations were executed")
        best_record = {
            "best_train_score": best_score,
            "best_params": _physical_dict(best_latent),
            "train_summary": asdict(
                evaluator.evaluate_summary(
                    jax.numpy.asarray(best_latent, dtype=jax.numpy.float32),
                    dataset.train,
                    windows=train_windows,
                    summary_eval=train_summary_eval,
                ).score
            ),
            "eval_summary": asdict(
                evaluator.evaluate_summary(
                    jax.numpy.asarray(best_latent, dtype=jax.numpy.float32),
                    dataset.eval,
                    windows=eval_windows,
                    summary_eval=eval_summary_eval,
                ).score
            ),
        }
        best_path.write_text(json.dumps(best_record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
