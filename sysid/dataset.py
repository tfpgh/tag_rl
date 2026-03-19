from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from sysid.align import align_run
from sysid.segments import extract_command_segments
from sysid.types import PreparedDataset, RunData


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    min_segment_duration_s: float = 0.6
    transition_duration_s: float = 0.25
    min_segment_steps: int = 12
    max_segment_steps: int = 160


def _infer_target_role(run: RunData) -> str:
    if run.target_robot_tag_id == 5:
        return "evader"
    return "chaser"


def _role_to_str(role: str) -> str:
    if role not in {"chaser", "evader"}:
        raise ValueError(f"unsupported role: {role}")
    return role


def build_prepared_dataset(
    runs: list[RunData],
    target_hz: float | None = None,
    config: DatasetConfig | None = None,
) -> PreparedDataset:
    if not runs:
        raise ValueError("at least one run is required")
    dataset_config = config or DatasetConfig()
    role = _role_to_str(_infer_target_role(runs[0]))

    initial_poses: list[list[float]] = []
    controls: list[list[list[float]]] = []
    references: list[list[list[float]]] = []
    masks: list[list[float]] = []
    run_names: list[str] = []
    segment_labels: list[str] = []

    for run in runs:
        run_role = _role_to_str(_infer_target_role(run))
        if run_role != role:
            raise ValueError(
                "all runs in a dataset must use the same target robot role"
            )
        aligned = align_run(run, target_hz=target_hz)
        if len(aligned.times_s) < dataset_config.min_segment_steps + 1:
            continue

        segments = extract_command_segments(
            run.command_timeline,
            min_duration_s=dataset_config.min_segment_duration_s,
            transition_duration_s=dataset_config.transition_duration_s,
        )
        if not segments:
            _append_aligned_trajectory(
                aligned,
                run.run_dir.name,
                "full_run",
                dataset_config,
                initial_poses,
                controls,
                references,
                masks,
                run_names,
                segment_labels,
            )
            continue

        for segment in segments:
            if segment.label == "idle":
                continue
            indices = [
                index
                for index, time_s in enumerate(aligned.times_s)
                if segment.start_time <= time_s <= segment.end_time
            ]
            if len(indices) < dataset_config.min_segment_steps + 1:
                continue
            start_index = indices[0]
            end_index = indices[-1]
            _append_index_window(
                aligned,
                start_index,
                end_index,
                run.run_dir.name,
                segment.label,
                dataset_config,
                initial_poses,
                controls,
                references,
                masks,
                run_names,
                segment_labels,
            )

    if not initial_poses:
        raise ValueError("no usable aligned segments were found in the provided runs")

    return PreparedDataset(
        initial_poses=jnp.asarray(initial_poses, dtype=jnp.float32),
        controls=jnp.asarray(controls, dtype=jnp.float32),
        references=jnp.asarray(references, dtype=jnp.float32),
        mask=jnp.asarray(masks, dtype=jnp.float32),
        role=role,
        run_names=tuple(run_names),
        segment_labels=tuple(segment_labels),
    )


def _append_aligned_trajectory(
    aligned,
    run_label: str,
    segment_label: str,
    config: DatasetConfig,
    initial_poses: list[list[float]],
    controls: list[list[list[float]]],
    references: list[list[list[float]]],
    masks: list[list[float]],
    run_names: list[str],
    segment_labels: list[str],
) -> None:
    _append_index_window(
        aligned,
        0,
        len(aligned.times_s) - 1,
        run_label,
        segment_label,
        config,
        initial_poses,
        controls,
        references,
        masks,
        run_names,
        segment_labels,
    )


def _append_index_window(
    aligned,
    start_index: int,
    end_index: int,
    run_label: str,
    segment_label: str,
    config: DatasetConfig,
    initial_poses: list[list[float]],
    controls_out: list[list[list[float]]],
    references_out: list[list[list[float]]],
    masks_out: list[list[float]],
    run_names: list[str],
    segment_labels: list[str],
) -> None:
    chunk_start = start_index
    while chunk_start < end_index:
        chunk_end = min(chunk_start + config.max_segment_steps, end_index)
        step_count = chunk_end - chunk_start
        if step_count < config.min_segment_steps:
            break

        control_chunk = [
            [aligned.left[index], aligned.right[index]]
            for index in range(chunk_start, chunk_end)
        ]
        reference_chunk = [
            [aligned.x_m[index], aligned.y_m[index], aligned.yaw_rad[index]]
            for index in range(chunk_start + 1, chunk_end + 1)
        ]
        initial_pose = [
            aligned.x_m[chunk_start],
            aligned.y_m[chunk_start],
            aligned.yaw_rad[chunk_start],
        ]

        pad = config.max_segment_steps - step_count
        if pad > 0:
            control_chunk.extend([[0.0, 0.0]] * pad)
            reference_chunk.extend([[0.0, 0.0, 0.0]] * pad)
        mask = [1.0] * step_count + [0.0] * pad

        initial_poses.append(initial_pose)
        controls_out.append(control_chunk)
        references_out.append(reference_chunk)
        masks_out.append(mask)
        run_names.append(run_label)
        segment_labels.append(segment_label)
        chunk_start = chunk_end
