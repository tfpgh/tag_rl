from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pose_xy(pose: dict | None) -> tuple[float, float] | None:
    if pose is None:
        return None
    return float(pose["x_m"]), float(pose["y_m"])


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python3 -m online.inspect_run data/real_runs/<run_dir>")
        raise SystemExit(1)

    run_dir = Path(sys.argv[1])
    samples = _load_jsonl(run_dir / "samples.jsonl")
    commands = _load_jsonl(run_dir / "commands.jsonl")
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))

    tracker = [sample["tracker"] for sample in samples]
    target_robot = [sample.get("target_robot") for sample in samples]
    visible_target = sum(
        1 for pose in target_robot if pose is not None and pose["visible"]
    )

    duration_s = 0.0
    if len(samples) >= 2:
        duration_s = samples[-1]["monotonic_time"] - samples[0]["monotonic_time"]

    print(f"run: {run_dir}")
    print(f"label: {metadata['teleop']['run_label']}")
    print(f"samples: {len(samples)}")
    print(f"commands: {len(commands)}")
    print(f"duration_s: {duration_s:.2f}")
    print(
        f"target_visible_fraction: {visible_target / len(samples):.3f}"
        if samples
        else "target_visible_fraction: 0.000"
    )
    print(f"mean_capture_ms: {_mean([item['capture_ms'] for item in tracker]):.2f}")
    print(f"mean_detector_ms: {_mean([item['detector_ms'] for item in tracker]):.2f}")
    print(f"mean_tracking_ms: {_mean([item['tracking_ms'] for item in tracker]):.2f}")
    print(f"mean_loop_ms: {_mean([item['loop_ms'] for item in tracker]):.2f}")
    poses_xy = [xy for pose in target_robot if (xy := _pose_xy(pose)) is not None]
    if poses_xy:
        xs = [xy[0] for xy in poses_xy]
        ys = [xy[1] for xy in poses_xy]
        print(f"target_x_range_m: {min(xs):.3f}..{max(xs):.3f}")
        print(f"target_y_range_m: {min(ys):.3f}..{max(ys):.3f}")


if __name__ == "__main__":
    main()
