from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import imageio
import mujoco
import numpy as np

from environment.config import EnvironmentConfig
from environment.mjcf import generate_mjcf
from environment.mujoco_data import JointQposSlices

DEFAULT_FPS = 20.0
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
OVERLAY_BG = (18, 20, 24)
OVERLAY_BORDER = (120, 130, 145)
TEXT_COLOR = (235, 238, 243)
LEFT_COLOR = (110, 185, 255)
RIGHT_COLOR = (140, 225, 180)
HIDDEN_OBSTACLE_Z = -10.0


@dataclass(slots=True)
class PoseState:
    x_m: float
    y_m: float
    yaw_rad: float


def yaw_to_quaternion_np(yaw_rad: float) -> np.ndarray:
    half_yaw = yaw_rad / 2.0
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a logged real-world run inside the MuJoCo arena."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument(
        "--keep-uncalibrated",
        action="store_true",
        help="Keep frames before calibration becomes valid.",
    )
    parser.add_argument(
        "--no-hold-last-pose",
        action="store_true",
        help="Do not reuse the last visible pose when a sample is missing one.",
    )
    return parser.parse_args()


def load_metadata(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "metadata.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def iter_samples(run_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with (run_dir / "samples.jsonl").open("r", encoding="utf-8") as file:
        for line in file:
            samples.append(json.loads(line))
    return samples


def build_env_config(
    metadata: dict[str, Any], max_logged_obstacles: int
) -> EnvironmentConfig:
    env_config = EnvironmentConfig()
    arena = metadata.get("arena", {})
    tag_size_m = float(arena.get("tag_size_m", 0.1))
    tag_center_width_m = float(
        arena.get("tag_center_width_m", env_config.arena_width + tag_size_m)
    )
    tag_center_height_m = float(
        arena.get("tag_center_height_m", env_config.arena_height + tag_size_m)
    )
    env_config.arena.arena_width = tag_center_width_m - tag_size_m
    env_config.arena.arena_height = tag_center_height_m - tag_size_m

    obstacle_size_m = arena.get("obstacle_size_m")
    if obstacle_size_m is not None:
        env_config.layout_randomization.obstacle_width = float(obstacle_size_m)

    env_config.layout_randomization.max_obstacles = max(
        env_config.layout_randomization.max_obstacles,
        max_logged_obstacles,
    )
    return env_config


def default_camera(env_config: EnvironmentConfig) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    camera.distance = max(env_config.arena_width, env_config.arena_height) * 1.2
    camera.azimuth = 90.0
    camera.elevation = -88.0
    return camera


def pose_from_record(
    record: dict[str, Any] | None,
    previous: PoseState | None,
    hold_last_pose: bool,
) -> PoseState | None:
    if record is not None:
        return PoseState(
            x_m=float(record["x_m"]),
            y_m=float(record["y_m"]),
            yaw_rad=float(record["yaw_rad"]),
        )
    if hold_last_pose:
        return previous
    return None


def set_agent_pose(
    qpos: np.ndarray,
    root_slice: slice,
    pose: PoseState | None,
    agent_z: float,
) -> None:
    if pose is None:
        qpos[root_slice] = np.array([0.0, 0.0, HIDDEN_OBSTACLE_Z, 1.0, 0.0, 0.0, 0.0])
        return
    quat = yaw_to_quaternion_np(pose.yaw_rad)
    qpos[root_slice] = np.array([pose.x_m, pose.y_m, agent_z, *quat], dtype=np.float64)


def obstacle_mocap_buffers(
    model: mujoco.MjModel,
    sample: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    mocap_pos = np.zeros((model.nmocap, 3), dtype=np.float64)
    mocap_quat = np.zeros((model.nmocap, 4), dtype=np.float64)
    mocap_quat[:, 0] = 1.0

    for mocap_id in range(model.nmocap):
        mocap_pos[mocap_id] = np.array([0.0, 0.0, HIDDEN_OBSTACLE_Z], dtype=np.float64)

    for obstacle_index, obstacle in enumerate(sample.get("obstacles", [])):
        pose = obstacle.get("pose")
        if pose is None:
            continue
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{obstacle_index}"
        )
        if body_id < 0:
            continue
        mocap_id = int(model.body_mocapid[body_id])
        mocap_pos[mocap_id] = np.array(
            [float(pose["x_m"]), float(pose["y_m"]), 0.05], dtype=np.float64
        )
        mocap_quat[mocap_id] = yaw_to_quaternion_np(float(pose["yaw_rad"]))
    return mocap_pos, mocap_quat


def draw_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    value: float,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(frame, (x, y), (x + width, y + height), (55, 58, 66), thickness=-1)
    mid_x = x + width // 2
    cv2.line(frame, (mid_x, y), (mid_x, y + height), (180, 185, 194), 1)

    magnitude = int(round((width / 2) * min(abs(value), 1.0)))
    if value >= 0:
        x0 = mid_x
        x1 = mid_x + magnitude
    else:
        x0 = mid_x - magnitude
        x1 = mid_x
    cv2.rectangle(frame, (x0, y + 2), (x1, y + height - 2), color, thickness=-1)


def draw_control_overlay(frame: np.ndarray, sample: dict[str, Any]) -> None:
    command = sample.get("command", {})
    left = float(command.get("left", 0.0))
    right = float(command.get("right", 0.0))
    left_int16 = int(command.get("left_int16", 0))
    right_int16 = int(command.get("right_int16", 0))

    panel_w = 350
    panel_h = 140
    margin = 24
    x0 = frame.shape[1] - panel_w - margin
    y0 = margin
    x1 = x0 + panel_w
    y1 = y0 + panel_h

    cv2.rectangle(frame, (x0, y0), (x1, y1), OVERLAY_BG, thickness=-1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), OVERLAY_BORDER, thickness=2)

    cv2.putText(
        frame,
        "Wheel Commands",
        (x0 + 16, y0 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"L {left:+.2f} ({left_int16:+d})",
        (x0 + 16, y0 + 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        LEFT_COLOR,
        2,
        cv2.LINE_AA,
    )
    draw_bar(frame, x0 + 16, y0 + 68, panel_w - 32, 20, left, LEFT_COLOR)

    cv2.putText(
        frame,
        f"R {right:+.2f} ({right_int16:+d})",
        (x0 + 16, y0 + 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        RIGHT_COLOR,
        2,
        cv2.LINE_AA,
    )
    draw_bar(frame, x0 + 16, y0 + 118, panel_w - 32, 20, right, RIGHT_COLOR)


def selected_samples(
    samples: list[dict[str, Any]],
    start_frame: int,
    end_frame: int | None,
    keep_uncalibrated: bool,
) -> list[dict[str, Any]]:
    selected = [
        sample for sample in samples if int(sample["frame_index"]) >= start_frame
    ]
    if end_frame is not None:
        selected = [
            sample for sample in selected if int(sample["frame_index"]) <= end_frame
        ]
    if keep_uncalibrated:
        return selected
    return [
        sample for sample in selected if bool(sample.get("calibration_valid", False))
    ]


def render_run(args: argparse.Namespace) -> None:
    metadata = load_metadata(args.run_dir)
    all_samples = iter_samples(args.run_dir)
    samples = selected_samples(
        all_samples,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        keep_uncalibrated=args.keep_uncalibrated,
    )
    if not samples:
        raise ValueError("No samples available after filtering.")

    max_logged_obstacles = max(len(sample.get("obstacles", [])) for sample in samples)
    env_config = build_env_config(metadata, max_logged_obstacles)
    model = mujoco.MjModel.from_xml_string(generate_mjcf(env_config, mode="render"))
    data = mujoco.MjData(model)
    qpos_slices = JointQposSlices(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = default_camera(env_config)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    hold_last_pose = not args.no_hold_last_pose
    chaser_pose: PoseState | None = None
    evader_pose: PoseState | None = None

    with imageio.get_writer(args.output, fps=args.fps, codec="h264") as writer:
        writer = cast(Any, writer)
        for index, sample in enumerate(samples, start=1):
            qpos = np.zeros(model.nq, dtype=np.float64)
            qvel = np.zeros(model.nv, dtype=np.float64)

            chaser_pose = pose_from_record(
                sample.get("chaser"), chaser_pose, hold_last_pose
            )
            evader_pose = pose_from_record(
                sample.get("evader"), evader_pose, hold_last_pose
            )

            set_agent_pose(
                qpos, qpos_slices.chaser_root, chaser_pose, env_config.agent_z
            )
            set_agent_pose(
                qpos, qpos_slices.evader_root, evader_pose, env_config.agent_z
            )

            mocap_pos, mocap_quat = obstacle_mocap_buffers(model, sample)

            data.qpos[:] = qpos
            data.qvel[:] = qvel
            if model.nmocap > 0:
                data.mocap_pos[:] = mocap_pos
                data.mocap_quat[:] = mocap_quat

            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = renderer.render().copy()
            draw_control_overlay(frame, sample)
            writer.append_data(frame)

            if index % 100 == 0 or index == len(samples):
                print(f"Rendered {index}/{len(samples)} frames")

    renderer.close()
    print(f"Saved video to {args.output}")


def main() -> None:
    args = parse_args()
    render_run(args)


if __name__ == "__main__":
    main()
