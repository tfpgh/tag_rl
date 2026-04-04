from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any, cast

import cv2
import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from environment.mjcf import generate_mjcf
from environment.motor import apply_deadzone
from environment.randomization import randomize_model
from online.app.render_run import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    OVERLAY_BG,
    OVERLAY_BORDER,
    TEXT_COLOR,
    PoseState,
    build_env_config,
    default_camera,
    draw_control_overlay,
    load_metadata,
    obstacle_mocap_buffers,
    pose_from_record,
    selected_samples,
    set_agent_pose,
)
from sysid.evaluator import SysIdEvaluator
from sysid.params import build_domain_and_pipeline_from_physical

REAL_COLOR = (110, 185, 255)
SIM_COLOR = (255, 176, 90)
TRAIL_REAL_COLOR = (60, 130, 220)
TRAIL_SIM_COLOR = (230, 140, 40)
ARENA_BG = (24, 27, 33)
ARENA_BORDER = (190, 196, 205)
OBSTACLE_COLOR = (130, 136, 150)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render recorded target pose against a simulated ghost from sysid params."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--keep-uncalibrated", action="store_true")
    parser.add_argument("--no-hold-last-pose", action="store_true")
    parser.add_argument("--trail-length", type=int, default=45)
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=5.0,
        help="Reset the simulated ghost to the recorded pose every N seconds. Use 0 to disable.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _extract_physical_params(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"No JSON rows found in {path}")
        payload = rows[-1]
    else:
        payload = json.loads(text)
    if "best_params" in payload:
        return {key: float(value) for key, value in payload["best_params"].items()}
    if "global_best" in payload and "params" in payload["global_best"]:
        return {
            key: float(value) for key, value in payload["global_best"]["params"].items()
        }
    if "generation_best" in payload and "params" in payload["generation_best"]:
        return {
            key: float(value)
            for key, value in payload["generation_best"]["params"].items()
        }
    return {key: float(value) for key, value in payload.items()}


def _iter_samples(run_dir: Path) -> list[dict[str, Any]]:
    return _load_jsonl(run_dir / "samples.jsonl")


def _build_command_substeps(
    commands: list[dict[str, Any]],
    start_time: float,
    end_time: float,
    substep_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    num_substeps = int(np.ceil((end_time - start_time) / substep_dt)) + 1
    times = start_time + np.arange(num_substeps, dtype=np.float64) * substep_dt
    command_times = np.array(
        [item["monotonic_time"] for item in commands], dtype=np.float64
    )
    command_values = (
        np.array(
            [[item["left_int16"], item["right_int16"]] for item in commands],
            dtype=np.float32,
        )
        / 32767.0
    )
    initial_index = max(
        int(np.searchsorted(command_times, start_time, side="right") - 1), 0
    )
    indices = np.searchsorted(command_times, times, side="right") - 1
    indices = np.clip(indices, 0, len(command_values) - 1)
    return command_values[indices], command_values[initial_index]


def _estimate_initial_velocity(samples: list[dict[str, Any]]) -> np.ndarray:
    visible = [sample for sample in samples if sample.get("target_robot") is not None]
    if len(visible) < 2:
        return np.zeros((3,), dtype=np.float32)
    first = visible[0]["target_robot"]
    second = visible[1]["target_robot"]
    dt = visible[1]["monotonic_time"] - visible[0]["monotonic_time"]
    if dt <= 0:
        return np.zeros((3,), dtype=np.float32)
    dyaw = np.arctan2(
        np.sin(second["yaw_rad"] - first["yaw_rad"]),
        np.cos(second["yaw_rad"] - first["yaw_rad"]),
    )
    return np.array(
        [
            (second["x_m"] - first["x_m"]) / dt,
            (second["y_m"] - first["y_m"]) / dt,
            dyaw / dt,
        ],
        dtype=np.float32,
    )


def _simulate_target_poses(
    evaluator: SysIdEvaluator,
    physical_params: dict[str, float],
    samples: list[dict[str, Any]],
    commands: list[dict[str, Any]],
) -> np.ndarray:
    visible = [sample for sample in samples if sample.get("target_robot") is not None]
    if not visible:
        raise ValueError("No target_robot poses available in selected samples")
    start_time = float(samples[0]["monotonic_time"])
    end_time = float(samples[-1]["monotonic_time"])
    command_substeps, initial_command = _build_command_substeps(
        commands, start_time, end_time, evaluator.substep_dt_seconds
    )
    first_pose = visible[0]["target_robot"]
    initial_pose = jnp.array(
        [first_pose["x_m"], first_pose["y_m"], first_pose["yaw_rad"]], dtype=jnp.float32
    )
    initial_velocity = jnp.asarray(
        _estimate_initial_velocity(samples), dtype=jnp.float32
    )
    domain_params, pipeline_params = build_domain_and_pipeline_from_physical(
        physical_params
    )
    model = randomize_model(
        evaluator.env.mjx_model, domain_params, evaluator.env.model_indices
    )
    data = evaluator._initial_data(model, initial_pose, initial_velocity)
    action_buffer = jnp.repeat(
        jnp.asarray(initial_command, dtype=jnp.float32)[None, :],
        evaluator.action_buffer_len,
        axis=0,
    )
    commands_jax = jnp.asarray(command_substeps, dtype=jnp.float32)

    def substep(
        carry: tuple[mjx.Data, jax.Array], proposed_action: jax.Array
    ) -> tuple[tuple[mjx.Data, jax.Array], jax.Array]:
        data, action_buffer = carry
        action_buffer = (
            jnp.roll(action_buffer, shift=-1, axis=0).at[-1].set(proposed_action)
        )
        delayed_action = action_buffer[-1 - pipeline_params.action_delay_substeps]
        motor_command = apply_deadzone(
            delayed_action, domain_params.chaser.motor_deadzone
        )
        data = data.replace(
            ctrl=jnp.concatenate([motor_command, jnp.zeros((2,), dtype=jnp.float32)])
        )
        data = mjx.step(model, data)
        qpos = data.qpos[evaluator.qpos_slices.chaser_root]
        qvel = data.qvel[evaluator.dof_slices.chaser_root]
        pose = jnp.array(
            [
                qpos[0],
                qpos[1],
                jnp.arctan2(
                    2.0 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
                    1.0 - 2.0 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
                ),
                qvel[0],
                qvel[1],
                qvel[5],
            ],
            dtype=jnp.float32,
        )
        return (data, action_buffer), pose

    _, pose_history = jax.lax.scan(substep, (data, action_buffer), commands_jax)
    relative_times = jnp.asarray(
        [sample["monotonic_time"] - start_time for sample in samples], dtype=jnp.float32
    )
    sample_indices = jnp.clip(
        jnp.rint(relative_times / evaluator.substep_dt_seconds).astype(jnp.int32)
        - 1
        - pipeline_params.observation_delay_substeps,
        0,
        pose_history.shape[0] - 1,
    )
    return np.asarray(jax.device_get(pose_history[sample_indices]))


def _segment_samples(
    samples: list[dict[str, Any]], segment_seconds: float
) -> list[list[dict[str, Any]]]:
    if segment_seconds <= 0.0 or not samples:
        return [samples]
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [samples[0]]
    segment_start = float(samples[0]["monotonic_time"])
    for sample in samples[1:]:
        sample_time = float(sample["monotonic_time"])
        if sample_time - segment_start >= segment_seconds:
            segments.append(current)
            current = [sample]
            segment_start = sample_time
        else:
            current.append(sample)
    if current:
        segments.append(current)
    return segments


def _simulate_target_poses_segmented(
    evaluator: SysIdEvaluator,
    physical_params: dict[str, float],
    samples: list[dict[str, Any]],
    commands: list[dict[str, Any]],
    segment_seconds: float,
) -> np.ndarray:
    segments = _segment_samples(samples, segment_seconds)
    simulated_segments = [
        _simulate_target_poses(evaluator, physical_params, segment, commands)
        for segment in segments
    ]
    return np.concatenate(simulated_segments, axis=0)


def _world_to_panel(
    x_m: float,
    y_m: float,
    panel_x: int,
    panel_y: int,
    panel_size: int,
    arena_w: float,
    arena_h: float,
) -> tuple[int, int]:
    px = panel_x + int(round(((x_m + arena_w / 2) / arena_w) * panel_size))
    py = panel_y + int(round((1.0 - ((y_m + arena_h / 2) / arena_h)) * panel_size))
    return px, py


def _draw_panel(
    frame: np.ndarray,
    sample: dict[str, Any],
    real_pose: PoseState | None,
    sim_pose: PoseState | None,
    real_trail: deque[tuple[float, float]],
    sim_trail: deque[tuple[float, float]],
    env_config,
    current_errors: dict[str, float] | None,
) -> None:
    panel_size = 320
    x0 = 24
    y0 = 24
    x1 = x0 + panel_size
    y1 = y0 + panel_size
    cv2.rectangle(frame, (x0, y0), (x1, y1), OVERLAY_BG, -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), OVERLAY_BORDER, 2)
    cv2.putText(
        frame,
        "Recorded vs Sim",
        (x0 + 12, y0 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    arena_margin = 16
    ax0 = x0 + arena_margin
    ay0 = y0 + 36
    arena_size = panel_size - arena_margin * 2
    ax1 = ax0 + arena_size
    ay1 = ay0 + arena_size
    cv2.rectangle(frame, (ax0, ay0), (ax1, ay1), ARENA_BG, -1)
    cv2.rectangle(frame, (ax0, ay0), (ax1, ay1), ARENA_BORDER, 2)

    for obstacle in sample.get("obstacles", []):
        pose = obstacle.get("pose")
        if pose is None:
            continue
        half = env_config.obstacle_width / 2
        c = np.cos(float(pose["yaw_rad"]))
        s = np.sin(float(pose["yaw_rad"]))
        corners = np.array(
            [[-half, -half], [half, -half], [half, half], [-half, half]],
            dtype=np.float32,
        )
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts_world = corners @ rot.T + np.array(
            [pose["x_m"], pose["y_m"]], dtype=np.float32
        )
        pts_panel = np.array(
            [
                _world_to_panel(
                    float(px),
                    float(py),
                    ax0,
                    ay0,
                    arena_size,
                    env_config.arena_width,
                    env_config.arena_height,
                )
                for px, py in pts_world
            ],
            dtype=np.int32,
        )
        cv2.polylines(
            frame, [pts_panel], isClosed=True, color=OBSTACLE_COLOR, thickness=2
        )

    def draw_trail(
        trail: deque[tuple[float, float]], color: tuple[int, int, int]
    ) -> None:
        if len(trail) < 2:
            return
        pts = np.array(
            [
                _world_to_panel(
                    x,
                    y,
                    ax0,
                    ay0,
                    arena_size,
                    env_config.arena_width,
                    env_config.arena_height,
                )
                for x, y in trail
            ],
            dtype=np.int32,
        )
        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

    draw_trail(real_trail, TRAIL_REAL_COLOR)
    draw_trail(sim_trail, TRAIL_SIM_COLOR)

    def draw_pose(
        pose: PoseState | None, color: tuple[int, int, int], label: str, row: int
    ) -> None:
        if pose is None:
            cv2.putText(
                frame,
                f"{label}: missing",
                (x0 + 12, y1 + 24 + row * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                cv2.LINE_AA,
            )
            return
        center = _world_to_panel(
            pose.x_m,
            pose.y_m,
            ax0,
            ay0,
            arena_size,
            env_config.arena_width,
            env_config.arena_height,
        )
        heading = (
            int(round(center[0] + 22.0 * np.cos(pose.yaw_rad))),
            int(round(center[1] - 22.0 * np.sin(pose.yaw_rad))),
        )
        cv2.circle(frame, center, 7, color, -1)
        cv2.line(frame, center, heading, color, 2)
        cv2.putText(
            frame,
            label,
            (x0 + 12, y1 + 24 + row * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

    draw_pose(real_pose, REAL_COLOR, "Recorded", 0)
    draw_pose(sim_pose, SIM_COLOR, "Simulated", 1)

    if current_errors is not None:
        cv2.putText(
            frame,
            f"Pos err: {current_errors['position_cm']:.1f} cm",
            (x0 + 150, y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Yaw err: {current_errors['yaw_deg']:.1f} deg",
            (x0 + 150, y1 + 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )


def _draw_metrics_overlay(
    frame: np.ndarray,
    current_errors: dict[str, float] | None,
    running_errors: dict[str, float],
    params_label: str,
) -> None:
    panel_w = 420
    panel_h = 120
    margin = 24
    x0 = 24
    y0 = frame.shape[0] - panel_h - margin
    x1 = x0 + panel_w
    y1 = y0 + panel_h
    cv2.rectangle(frame, (x0, y0), (x1, y1), OVERLAY_BG, -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), OVERLAY_BORDER, 2)
    cv2.putText(
        frame,
        params_label,
        (x0 + 14, y0 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    if current_errors is not None:
        cv2.putText(
            frame,
            f"Current pos {current_errors['position_cm']:.1f} cm",
            (x0 + 14, y0 + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            REAL_COLOR,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Current yaw {current_errors['yaw_deg']:.1f} deg",
            (x0 + 14, y0 + 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            REAL_COLOR,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"Running mean pos {running_errors['position_cm']:.1f} cm",
        (x0 + 220, y0 + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        SIM_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Running mean yaw {running_errors['yaw_deg']:.1f} deg",
        (x0 + 220, y0 + 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        SIM_COLOR,
        2,
        cv2.LINE_AA,
    )


def render_compare(args: argparse.Namespace) -> None:
    metadata = load_metadata(args.run_dir)
    all_samples = _iter_samples(args.run_dir)
    samples = selected_samples(
        all_samples,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        keep_uncalibrated=args.keep_uncalibrated,
    )
    if not samples:
        raise ValueError("No samples available after filtering.")
    commands = _load_jsonl(args.run_dir / "commands.jsonl")
    params = _extract_physical_params(args.params)

    max_logged_obstacles = max(len(sample.get("obstacles", [])) for sample in samples)
    env_config = build_env_config(metadata, max_logged_obstacles)
    evaluator = SysIdEvaluator.create(
        env_config,
        num_substeps=int(
            np.ceil(
                (samples[-1]["monotonic_time"] - samples[0]["monotonic_time"]) / 0.005
            )
        )
        + 1,
    )
    simulated = _simulate_target_poses_segmented(
        evaluator, params, samples, commands, args.segment_seconds
    )

    model = mujoco.MjModel.from_xml_string(generate_mjcf(env_config, mode="render"))
    data = mujoco.MjData(model)
    qpos_slices = evaluator.qpos_slices
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = default_camera(env_config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    hold_last_pose = not args.no_hold_last_pose
    real_pose: PoseState | None = None
    real_trail: deque[tuple[float, float]] = deque(maxlen=args.trail_length)
    sim_trail: deque[tuple[float, float]] = deque(maxlen=args.trail_length)
    running_pos_errs: list[float] = []
    running_yaw_errs: list[float] = []
    params_label = f"Params: {args.params.name}"

    with imageio.get_writer(args.output, fps=args.fps, codec="h264") as writer:
        writer = cast(Any, writer)
        for index, (sample, sim_pose_array) in enumerate(
            zip(samples, simulated, strict=True), start=1
        ):
            qpos = np.zeros(model.nq, dtype=np.float64)
            qvel = np.zeros(model.nv, dtype=np.float64)

            real_pose = pose_from_record(
                sample.get("target_robot"), real_pose, hold_last_pose
            )
            sim_pose = PoseState(
                x_m=float(sim_pose_array[0]),
                y_m=float(sim_pose_array[1]),
                yaw_rad=float(sim_pose_array[2]),
            )

            set_agent_pose(qpos, qpos_slices.chaser_root, real_pose, env_config.agent_z)
            set_agent_pose(qpos, qpos_slices.evader_root, sim_pose, env_config.agent_z)
            mocap_pos, mocap_quat = obstacle_mocap_buffers(model, sample)

            data.qpos[:] = qpos
            data.qvel[:] = qvel
            if model.nmocap > 0:
                data.mocap_pos[:] = mocap_pos
                data.mocap_quat[:] = mocap_quat
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = renderer.render().copy()

            current_errors: dict[str, float] | None = None
            if real_pose is not None:
                real_trail.append((real_pose.x_m, real_pose.y_m))
                pos_err = float(
                    np.hypot(sim_pose.x_m - real_pose.x_m, sim_pose.y_m - real_pose.y_m)
                )
                yaw_err = float(
                    np.degrees(
                        np.arctan2(
                            np.sin(sim_pose.yaw_rad - real_pose.yaw_rad),
                            np.cos(sim_pose.yaw_rad - real_pose.yaw_rad),
                        )
                    )
                )
                running_pos_errs.append(pos_err)
                running_yaw_errs.append(abs(yaw_err))
                current_errors = {
                    "position_cm": pos_err * 100.0,
                    "yaw_deg": abs(yaw_err),
                }
            sim_trail.append((sim_pose.x_m, sim_pose.y_m))

            draw_control_overlay(frame, sample)
            _draw_panel(
                frame,
                sample,
                real_pose,
                sim_pose,
                real_trail,
                sim_trail,
                env_config,
                current_errors,
            )
            _draw_metrics_overlay(
                frame,
                current_errors,
                {
                    "position_cm": (100.0 * float(np.mean(running_pos_errs)))
                    if running_pos_errs
                    else 0.0,
                    "yaw_deg": float(np.mean(running_yaw_errs))
                    if running_yaw_errs
                    else 0.0,
                },
                params_label,
            )
            writer.append_data(frame)
            if index % 100 == 0 or index == len(samples):
                print(f"Rendered {index}/{len(samples)} frames")

    renderer.close()
    print(f"Saved comparison video to {args.output}")


def main() -> None:
    args = parse_args()
    render_compare(args)


if __name__ == "__main__":
    main()
