from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from runtime import jax_setup as _jax_setup  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from environment.config import EnvironmentConfig
from sysid.dataset import WindowConfig, load_dataset_splits
from sysid.evaluator import SysIdEvaluator
from sysid.params import (
    PARAM_BOUNDS,
    PARAM_NAMES,
    PARAM_SPEC_BY_NAME,
    physical_to_latent,
)

CORE_SWEEP_PARAMS = [
    "track_width_scale",
    "wheel_radius_scale",
    "wheel_slide_friction_scale",
    "wheel_torsional_friction_scale",
    "wheel_rolling_friction_scale",
    "body_contact_friction_scale",
    "motor_strength_scale",
    "back_emf_scale",
    "motor_balance",
    "motor_time_constant_seconds",
    "command_delay_substeps",
    "observation_delay_substeps",
]

SECONDARY_SWEEP_PARAMS = [
    "caster_radius_scale",
    "caster_offset_x",
    "caster_slide_friction_scale",
    "caster_torsional_friction_scale",
    "wheel_joint_damping_scale",
    "wheel_joint_frictionloss_scale",
    "wheel_armature_scale",
    "com_offset_x",
    "com_offset_z",
]


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


def _load_params_from_history(path: Path, source: str) -> dict[str, float]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    if source == "last_generation":
        payload = rows[-1]["generation_best"]["params"]
    elif source == "best_train":
        payload = min(
            rows,
            key=lambda row: row["generation_best"]["train_summary"]["score"]["total"],
        )["generation_best"]["params"]
    elif source == "best_eval_generation":
        payload = min(
            rows,
            key=lambda row: row["generation_best"]["eval_summary"]["score"]["total"],
        )["generation_best"]["params"]
    elif source == "global_best":
        payload = rows[-1]["global_best"]["params"]
    else:
        raise ValueError(f"Unsupported history source: {source}")
    return {name: float(payload[name]) for name in PARAM_NAMES}


def _to_latent(physical: dict[str, float]) -> jax.Array:
    return physical_to_latent(physical)


def _linspace_for_param(name: str, points: int) -> list[float]:
    spec = PARAM_SPEC_BY_NAME[name]
    bounds = PARAM_BOUNDS[name]
    if spec.transform == "integer":
        low = int(round(bounds.low))
        high = int(round(bounds.high))
        if high - low + 1 <= points:
            return [float(v) for v in range(low, high + 1)]
        return [float(v) for v in np.linspace(low, high, points).round().astype(int)]
    if spec.transform == "log":
        return [float(v) for v in np.geomspace(bounds.low, bounds.high, points)]
    return [float(v) for v in np.linspace(bounds.low, bounds.high, points)]


def _local_values_for_param(
    name: str, base: dict[str, float], points: int, local_fraction: float
) -> list[float]:
    spec = PARAM_SPEC_BY_NAME[name]
    bounds = PARAM_BOUNDS[name]
    base_value = float(base[name])
    if spec.transform == "integer":
        span = bounds.high - bounds.low
        radius = 0.5 * local_fraction * span
        low = max(bounds.low, base_value - radius)
        high = min(bounds.high, base_value + radius)
        low_i = int(round(low))
        high_i = int(round(high))
        base_i = int(round(base_value))
        values = sorted(
            set(
                np.linspace(low_i, high_i, points).round().astype(int).tolist()
                + [base_i]
            )
        )
        return [float(v) for v in values]
    if spec.transform == "log":
        log_low = math.log(bounds.low)
        log_high = math.log(bounds.high)
        log_base = math.log(min(max(base_value, bounds.low), bounds.high))
        radius = 0.5 * local_fraction * (log_high - log_low)
        low = max(bounds.low, math.exp(log_base - radius))
        high = min(bounds.high, math.exp(log_base + radius))
        values = np.geomspace(low, high, points, dtype=np.float64)
        values = np.unique(
            np.concatenate([values, np.asarray([base_value], dtype=np.float64)])
        )
        return [float(v) for v in values.tolist()]
    span = bounds.high - bounds.low
    radius = 0.5 * local_fraction * span
    low = max(bounds.low, base_value - radius)
    high = min(bounds.high, base_value + radius)
    values = np.linspace(low, high, points, dtype=np.float64)
    values = np.unique(
        np.concatenate([values, np.asarray([base_value], dtype=np.float64)])
    )
    return [float(v) for v in values.tolist()]


def _candidate(base: dict[str, float], **updates: float) -> dict[str, float]:
    out = dict(base)
    out.update(updates)
    return out


def _build_probe_groups(
    base: dict[str, float],
    points: int,
    local_fraction: float,
    sweep_scope: str,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {
        "baseline": [{"label": "baseline", "params": dict(base)}],
    }

    if sweep_scope == "core":
        sweep_params = CORE_SWEEP_PARAMS
    elif sweep_scope == "all":
        sweep_params = CORE_SWEEP_PARAMS + SECONDARY_SWEEP_PARAMS
    else:
        raise ValueError(f"Unsupported sweep scope: {sweep_scope}")

    for name in sweep_params:
        full_values = _linspace_for_param(name, points)
        local_values = _local_values_for_param(name, base, points, local_fraction)
        groups[f"sweep:{name}"] = [
            {"label": f"{name}={value}", "params": _candidate(base, **{name: value})}
            for value in full_values
        ]
        groups[f"local:{name}"] = [
            {"label": f"{name}={value}", "params": _candidate(base, **{name: value})}
            for value in local_values
        ]

    track_values = _local_values_for_param("track_width_scale", base, 5, local_fraction)
    radius_values = _local_values_for_param(
        "wheel_radius_scale", base, 5, local_fraction
    )
    obs_values = _local_values_for_param(
        "observation_delay_substeps", base, 5, local_fraction
    )
    cmd_values = _local_values_for_param(
        "command_delay_substeps", base, 5, local_fraction
    )
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
    groups["grid:track_vs_radius"] = [
        {
            "label": f"track={track} radius={radius}",
            "params": _candidate(
                base, track_width_scale=track, wheel_radius_scale=radius
            ),
        }
        for track in track_values
        for radius in radius_values
    ]
    groups["grid:cmd_vs_obs_delay"] = [
        {
            "label": f"cmd={cmd} obs={obs}",
            "params": _candidate(
                base,
                command_delay_substeps=cmd,
                observation_delay_substeps=obs,
            ),
        }
        for cmd in cmd_values
        for obs in obs_values
    ]

    friction_values = _local_values_for_param(
        "wheel_slide_friction_scale", base, 5, local_fraction
    )
    torsional_values = _local_values_for_param(
        "wheel_torsional_friction_scale", base, 5, local_fraction
    )
    groups["grid:wheel_slide_vs_torsion"] = [
        {
            "label": f"wheel_slide={friction} torsion={torsion}",
            "params": _candidate(
                base,
                wheel_slide_friction_scale=friction,
                wheel_torsional_friction_scale=torsion,
            ),
        }
        for friction in friction_values
        for torsion in torsional_values
    ]

    strength_values = _local_values_for_param(
        "motor_strength_scale", base, 5, local_fraction
    )
    emf_values = _local_values_for_param("back_emf_scale", base, 5, local_fraction)
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

    lag_values = _local_values_for_param(
        "motor_time_constant_seconds", base, 5, local_fraction
    )
    groups["grid:balance_vs_lag"] = [
        {
            "label": f"balance={balance} lag={lag}",
            "params": _candidate(
                base,
                motor_balance=balance,
                motor_time_constant_seconds=lag,
            ),
        }
        for balance in _local_values_for_param("motor_balance", base, 5, local_fraction)
        for lag in lag_values
    ]

    groups["ablations"] = [
        {"label": "com_offset_x=0", "params": _candidate(base, com_offset_x=0.0)},
        {"label": "com_offset_z=0", "params": _candidate(base, com_offset_z=0.0)},
        {
            "label": "wheel_torsion=1.0",
            "params": _candidate(base, wheel_torsional_friction_scale=1.0),
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


def _safe_score(value: float) -> float:
    return value if np.isfinite(value) else float("inf")


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
        train_total = _safe_score(float(train_scores[idx]))
        eval_total = _safe_score(float(eval_scores[idx]))
        rows.append(
            {
                "label": item["label"],
                "params": item["params"],
                "train_total": train_total,
                "eval_total": eval_total,
            }
        )
    rows.sort(key=lambda row: row["eval_total"])
    return rows


def _sweep_diagnostics(
    name: str, results: list[dict[str, Any]], points: int
) -> dict[str, Any]:
    ordered = sorted(results, key=lambda row: float(row["params"][name]))
    values = [float(row["params"][name]) for row in ordered]
    scores = [float(row["eval_total"]) for row in ordered]
    best_index = int(np.argmin(scores))
    best_value = values[best_index]
    lower_edge = best_index == 0
    upper_edge = best_index == len(values) - 1
    return {
        "param": name,
        "best_value": best_value,
        "best_eval_total": scores[best_index],
        "best_index": best_index,
        "point_count": len(values),
        "lower_edge_best": lower_edge,
        "upper_edge_best": upper_edge,
        "suggests_expand_lower": lower_edge and len(values) >= max(points - 1, 3),
        "suggests_expand_upper": upper_edge and len(values) >= max(points - 1, 3),
        "values": values,
        "eval_totals": scores,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe sysid parameter sensitivities around a chosen parameter set."
    )
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/sysid_probe.json"))
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--preroll-seconds", type=float, default=0.5)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--points", type=int, default=7)
    parser.add_argument(
        "--local-fraction",
        type=float,
        default=0.25,
        help="Fraction of the full bound span to use for local sweeps around the base params.",
    )
    parser.add_argument(
        "--sweep-scope",
        choices=("core", "all"),
        default="core",
        help="Probe only core parameters or all parameters.",
    )
    parser.add_argument(
        "--history-source",
        choices=(
            "global_best",
            "best_eval_generation",
            "best_train",
            "last_generation",
        ),
        default="global_best",
        help="When --params points to history.jsonl, choose which parameter source to probe around.",
    )
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

    if args.params.suffix == ".jsonl":
        base_params = _load_params_from_history(args.params, args.history_source)
    else:
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

    groups = _build_probe_groups(
        base_params,
        args.points,
        args.local_fraction,
        args.sweep_scope,
    )
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
    sweep_diagnostics = {
        group_name: _sweep_diagnostics(
            group_name.split(":", 1)[1], results, args.points
        )
        for group_name, results in group_results.items()
        if group_name.startswith("local:") or group_name.startswith("sweep:")
    }
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
        "sweep_diagnostics": sweep_diagnostics,
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
    print("\nExpansion hints:")
    for group_name, info in sorted(sweep_diagnostics.items()):
        if info["suggests_expand_lower"] or info["suggests_expand_upper"]:
            direction = "lower" if info["suggests_expand_lower"] else "upper"
            print(
                f"  {group_name}: best at {direction} edge value={info['best_value']:.6g} eval_total={info['best_eval_total']:.4f}"
            )
    print(f"\nSaved probe report to {args.output}")


if __name__ == "__main__":
    main()
