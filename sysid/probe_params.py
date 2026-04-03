from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from environment.config import EnvironmentConfig
from sysid.dataset import WindowConfig, load_dataset_splits
from sysid.evaluator import SysIdEvaluator
from sysid.params import PARAM_BOUNDS, PARAM_NAMES, PARAM_SPECS


def _load_params(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"No rows found in {path}")
        payload = rows[-1]
    else:
        payload = json.loads(text)
    if "best_params" in payload:
        payload = payload["best_params"]
    elif "global_best" in payload and "params" in payload["global_best"]:
        payload = payload["global_best"]["params"]
    elif "generation_best" in payload and "params" in payload["generation_best"]:
        payload = payload["generation_best"]["params"]
    return {name: float(payload[name]) for name in PARAM_NAMES}


def _to_latent(physical: dict[str, float]) -> jax.Array:
    latent: list[float] = []
    for name, bounds in PARAM_SPECS:
        span = bounds.high - bounds.low
        fraction = (physical[name] - bounds.low) / span
        fraction = min(max(fraction, 1e-6), 1.0 - 1e-6)
        latent.append(float(np.log(fraction / (1.0 - fraction))))
    return jnp.asarray(latent, dtype=jnp.float32)


def _linspace_for_param(name: str, points: int) -> list[float]:
    bounds = PARAM_BOUNDS[name]
    if name.endswith("delay_substeps"):
        low = int(round(bounds.low))
        high = int(round(bounds.high))
        if high - low + 1 <= points:
            return [float(v) for v in range(low, high + 1)]
        return [float(v) for v in np.linspace(low, high, points).round().astype(int)]
    return [float(v) for v in np.linspace(bounds.low, bounds.high, points)]


def _candidate(base: dict[str, float], **updates: float) -> dict[str, float]:
    out = dict(base)
    out.update(updates)
    return out


def _build_probe_groups(
    base: dict[str, float], points: int
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {
        "baseline": [{"label": "baseline", "params": dict(base)}],
    }

    sweep_params = [
        "track_width_scale",
        "motor_strength_scale",
        "back_emf_scale",
        "motor_deadzone",
        "motor_time_constant_seconds",
        "wheel_friction_scale",
        "wheel_scrub_scale",
        "com_offset_x",
        "observation_delay_substeps",
        "command_delay_substeps",
    ]
    for name in sweep_params:
        groups[f"sweep:{name}"] = [
            {"label": f"{name}={value}", "params": _candidate(base, **{name: value})}
            for value in _linspace_for_param(name, points)
        ]

    track_values = _linspace_for_param("track_width_scale", 5)
    obs_values = _linspace_for_param("observation_delay_substeps", 5)
    groups["grid:track_vs_obs_delay"] = [
        {
            "label": f"track={track} obs={obs}",
            "params": _candidate(
                base, track_width_scale=track, observation_delay_substeps=obs
            ),
        }
        for track in track_values
        for obs in obs_values
    ]

    friction_values = _linspace_for_param("wheel_friction_scale", 5)
    scrub_values = _linspace_for_param("wheel_scrub_scale", 5)
    groups["grid:friction_vs_scrub"] = [
        {
            "label": f"wheel_friction={friction} scrub={scrub}",
            "params": _candidate(
                base, wheel_friction_scale=friction, wheel_scrub_scale=scrub
            ),
        }
        for friction in friction_values
        for scrub in scrub_values
    ]

    strength_values = _linspace_for_param("motor_strength_scale", 5)
    emf_values = _linspace_for_param("back_emf_scale", 5)
    groups["grid:strength_vs_emf"] = [
        {
            "label": f"strength={strength} back_emf={emf}",
            "params": _candidate(
                base, motor_strength_scale=strength, back_emf_scale=emf
            ),
        }
        for strength in strength_values
        for emf in emf_values
    ]

    deadzone_values = _linspace_for_param("motor_deadzone", 5)
    lag_values = _linspace_for_param("motor_time_constant_seconds", 5)
    groups["grid:deadzone_vs_lag"] = [
        {
            "label": f"deadzone={deadzone} lag={lag}",
            "params": _candidate(
                base,
                motor_deadzone=deadzone,
                motor_time_constant_seconds=lag,
            ),
        }
        for deadzone in deadzone_values
        for lag in lag_values
    ]

    groups["ablations"] = [
        {"label": "com_offset_x=0", "params": _candidate(base, com_offset_x=0.0)},
        {
            "label": "wheel_scrub=1.0",
            "params": _candidate(base, wheel_scrub_scale=1.0),
        },
        {
            "label": "motor_deadzone=0",
            "params": _candidate(base, motor_deadzone=0.0),
        },
        {
            "label": "motor_lag=0",
            "params": _candidate(base, motor_time_constant_seconds=0.0),
        },
        {
            "label": "track_width=1.04",
            "params": _candidate(base, track_width_scale=1.04),
        },
        {
            "label": "obs_delay=8",
            "params": _candidate(base, observation_delay_substeps=8.0),
        },
    ]

    return groups


def _evaluate_group(
    evaluator: SysIdEvaluator,
    train_windows,
    eval_windows,
    train_eval,
    eval_eval,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latents = jnp.stack([_to_latent(item["params"]) for item in candidates], axis=0)
    train_scores = jax.device_get(
        evaluator.evaluate_population(
            latents, train_windows, population_eval=train_eval
        )["total"]
    )
    eval_scores = jax.device_get(
        evaluator.evaluate_population(latents, eval_windows, population_eval=eval_eval)[
            "total"
        ]
    )
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(candidates):
        rows.append(
            {
                "label": item["label"],
                "params": item["params"],
                "train_total": float(train_scores[idx]),
                "eval_total": float(eval_scores[idx]),
            }
        )
    rows.sort(key=lambda row: row["eval_total"])
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe sysid parameter sensitivities around a chosen parameter set."
    )
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/sysid_probe.json"))
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--preroll-seconds", type=float, default=0.5)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--points", type=int, default=7)
    parser.add_argument(
        "--full-summary-top-k",
        type=int,
        default=5,
        help="Compute full train/eval metric summaries for the top K eval candidates overall.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    base_params = _load_params(args.params)
    window_config = WindowConfig(
        window_seconds=args.window_seconds,
        preroll_seconds=args.preroll_seconds,
        stride_seconds=args.stride_seconds,
    )
    dataset = load_dataset_splits(args.data_root, window_config)
    evaluator = SysIdEvaluator.create(EnvironmentConfig(), window_config.num_substeps)
    train_windows = evaluator.prepare_batch(dataset.train)
    eval_windows = evaluator.prepare_batch(dataset.eval)
    train_eval = evaluator.build_population_evaluator(train_windows)
    eval_eval = evaluator.build_population_evaluator(eval_windows)
    train_summary_eval = evaluator.build_summary_evaluator(train_windows)
    eval_summary_eval = evaluator.build_summary_evaluator(eval_windows)

    groups = _build_probe_groups(base_params, args.points)
    group_results: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for group_name, candidates in groups.items():
        results = _evaluate_group(
            evaluator, train_windows, eval_windows, train_eval, eval_eval, candidates
        )
        group_results[group_name] = results
        for row in results:
            merged = dict(row)
            merged["group"] = group_name
            all_rows.append(merged)

    all_rows.sort(key=lambda row: row["eval_total"])
    top_rows = all_rows[: max(args.full_summary_top_k, 1)]
    summaries: list[dict[str, Any]] = []
    for row in top_rows:
        latent = _to_latent(row["params"])
        train_summary = evaluator.evaluate_summary(
            latent,
            dataset.train,
            windows=train_windows,
            summary_eval=train_summary_eval,
        )
        eval_summary = evaluator.evaluate_summary(
            latent, dataset.eval, windows=eval_windows, summary_eval=eval_summary_eval
        )
        summaries.append(
            {
                "label": row["label"],
                "group": row["group"],
                "params": row["params"],
                "train_total": row["train_total"],
                "eval_total": row["eval_total"],
                "train_summary": {
                    "score": asdict(train_summary.score),
                    "metrics": train_summary.metrics,
                    "maneuver_scores": train_summary.maneuver_scores,
                    "windows": train_summary.windows,
                },
                "eval_summary": {
                    "score": asdict(eval_summary.score),
                    "metrics": eval_summary.metrics,
                    "maneuver_scores": eval_summary.maneuver_scores,
                    "windows": eval_summary.windows,
                },
            }
        )

    baseline = group_results["baseline"][0]
    report = {
        "base_params": base_params,
        "baseline": baseline,
        "best_overall": all_rows[:10],
        "group_results": group_results,
        "top_summaries": summaries,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Baseline:")
    print(
        f"  train_total={baseline['train_total']:.4f} eval_total={baseline['eval_total']:.4f}"
    )
    print("\nBest per group:")
    for group_name, results in group_results.items():
        best = results[0]
        delta = best["eval_total"] - baseline["eval_total"]
        print(
            f"  {group_name}: eval_total={best['eval_total']:.4f} delta={delta:+.4f} label={best['label']}"
        )
    print("\nTop overall:")
    for row in all_rows[:10]:
        print(
            f"  eval_total={row['eval_total']:.4f} train_total={row['train_total']:.4f} group={row['group']} label={row['label']}"
        )
    print(f"\nSaved probe report to {args.output}")


if __name__ == "__main__":
    main()
