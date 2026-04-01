from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from sysid.dataset import WindowConfig, load_dataset_splits


def _print_split_summary(name: str, batch) -> None:
    maneuvers = Counter(item.maneuver for item in batch.metadata)
    command_mean = sum(item.command_mean_abs for item in batch.metadata) / len(
        batch.metadata
    )
    translation_mean = sum(
        item.translation_distance_m for item in batch.metadata
    ) / len(batch.metadata)
    yaw_mean = sum(item.yaw_change_rad for item in batch.metadata) / len(batch.metadata)
    print(f"[{name}] windows={len(batch.metadata)}")
    print(
        "  mean command={:.3f} mean translation={:.3f}m mean yaw_change={:.3f}rad".format(
            command_mean, translation_mean, yaw_mean
        )
    )
    print("  maneuvers:")
    for label, count in sorted(maneuvers.items()):
        print(f"    {label}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sysid window segmentation")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--preroll-seconds", type=float, default=0.5)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    args = parser.parse_args()

    config = WindowConfig(
        window_seconds=args.window_seconds,
        preroll_seconds=args.preroll_seconds,
        stride_seconds=args.stride_seconds,
    )
    dataset = load_dataset_splits(args.data_root, config)
    _print_split_summary("train", dataset.train)
    _print_split_summary("eval", dataset.eval)


if __name__ == "__main__":
    main()
