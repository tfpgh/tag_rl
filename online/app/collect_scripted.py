from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import time
import tty
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

import cv2

from online.control.teleop import TeleopController
from online.core.config import TrackerConfig
from online.core.state import BoardState, Pose2D
from online.data.logger import RunLogger
from online.tracking.tracker import ARENA_WINDOW_NAME, RAW_WINDOW_NAME, BoardTracker

SAFE_LINEAR_COMMAND = 0.55
START_POSE_CENTER = "centered, facing +x"
START_POSE_CENTER_REVERSE = "centered, facing -x"


@dataclass(frozen=True, slots=True)
class Activation:
    name: str
    left: float
    right: float
    active_duration_s: float
    start_pose: str
    pre_roll_s: float = 0.50
    post_roll_s: float = 0.50


@dataclass(frozen=True, slots=True)
class ActivationRecord:
    activation_index: int
    activation_name: str
    status: str
    scheduled_left: float
    scheduled_right: float
    active_duration_s: float
    pre_roll_s: float
    post_roll_s: float
    start_pose: str
    started_at: float | None
    ended_at: float | None
    wall_started_at: str | None
    wall_ended_at: str | None
    abort_reason: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step through hard-coded sysid activations and log one dataset split."
    )
    parser.add_argument("--split", choices=("train", "eval"), required=True)
    parser.add_argument(
        "--show-preview",
        choices=("none", "raw", "arena", "both"),
        default="arena",
    )
    parser.add_argument(
        "--stop-on-target-loss-s",
        type=float,
        default=0.25,
        help="Abort the active activation if the tracked robot disappears this long.",
    )
    parser.add_argument(
        "--activation-scale",
        type=float,
        default=1.0,
        help="Uniform scale factor applied to all hard-coded wheel commands.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the activation sequence for the selected split and exit.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the initial confirmation prompt.",
    )
    return parser.parse_args()


def _base_activations() -> list[Activation]:
    return [
        Activation("deadzone_forward", 0.07, 0.07, 0.70, START_POSE_CENTER),
        Activation("deadzone_reverse", -0.07, -0.07, 0.70, START_POSE_CENTER_REVERSE),
        Activation("small_forward_step", 0.11, 0.11, 0.80, START_POSE_CENTER),
        Activation("small_reverse_step", -0.11, -0.11, 0.80, START_POSE_CENTER_REVERSE),
        Activation("forward_low", 0.17, 0.17, 0.75, START_POSE_CENTER),
        Activation("forward_medium", 0.26, 0.26, 0.60, START_POSE_CENTER),
        Activation("reverse_low", -0.17, -0.17, 0.75, START_POSE_CENTER_REVERSE),
        Activation("reverse_medium", -0.26, -0.26, 0.60, START_POSE_CENTER_REVERSE),
        Activation("pivot_left_low", -0.13, 0.13, 0.75, START_POSE_CENTER),
        Activation("pivot_right_low", 0.13, -0.13, 0.75, START_POSE_CENTER),
        Activation("pivot_left_medium", -0.21, 0.21, 0.65, START_POSE_CENTER),
        Activation("pivot_right_medium", 0.21, -0.21, 0.65, START_POSE_CENTER),
        Activation("arc_left_gentle", 0.14, 0.24, 0.80, START_POSE_CENTER),
        Activation("arc_right_gentle", 0.24, 0.14, 0.80, START_POSE_CENTER),
        Activation("arc_left_medium", 0.12, 0.30, 0.70, START_POSE_CENTER),
        Activation("arc_right_medium", 0.30, 0.12, 0.70, START_POSE_CENTER),
        Activation("arc_left_strong", 0.08, 0.33, 0.60, START_POSE_CENTER),
        Activation("arc_right_strong", 0.33, 0.08, 0.60, START_POSE_CENTER),
        Activation("forward_pulse", 0.20, 0.20, 0.40, START_POSE_CENTER),
        Activation("reverse_pulse", -0.20, -0.20, 0.40, START_POSE_CENTER_REVERSE),
    ]


def build_activations(split: str, scale: float) -> list[Activation]:
    repetitions = 4 if split == "train" else 2
    activations: list[Activation] = []
    for repetition in range(repetitions):
        for activation in _base_activations():
            activations.append(
                Activation(
                    name=f"{activation.name}_rep{repetition + 1}",
                    left=max(-1.0, min(1.0, activation.left * scale)),
                    right=max(-1.0, min(1.0, activation.right * scale)),
                    active_duration_s=activation.active_duration_s,
                    start_pose=activation.start_pose,
                    pre_roll_s=activation.pre_roll_s,
                    post_roll_s=activation.post_roll_s,
                )
            )
    return activations


def _preview_flags(mode: str) -> tuple[bool, bool]:
    return mode in ("raw", "both"), mode in ("arena", "both")


def _target_pose(
    board_state: BoardState, robot_tag_id: int, config: TrackerConfig
) -> Pose2D | None:
    if robot_tag_id == config.arena.chaser_tag_id:
        return board_state.chaser
    if robot_tag_id == config.arena.evader_tag_id:
        return board_state.evader
    return None


def _render_preview(
    tracker: BoardTracker,
    frame,
    board_state: BoardState,
    show_raw: bool,
    show_arena: bool,
) -> None:
    if not (show_raw or show_arena):
        return
    raw_view, arena_view = tracker.render_debug_views(frame, board_state)
    if show_raw:
        cv2.imshow(RAW_WINDOW_NAME, raw_view)
    if show_arena:
        cv2.imshow(ARENA_WINDOW_NAME, arena_view)


def _write_segment_record(file: TextIO, record: ActivationRecord) -> None:
    file.write(json.dumps(asdict(record)) + "\n")
    file.flush()


def _update_metadata(
    logger: RunLogger, split: str, activations: list[Activation]
) -> None:
    metadata_path = logger.run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["collection"] = {
        "mode": "scripted",
        "split": split,
        "activation_count": len(activations),
        "repetitions": 4 if split == "train" else 2,
        "created_at": datetime.now().isoformat(),
    }
    metadata["scripted_activations"] = [
        asdict(activation) for activation in activations
    ]
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _drain_stdin() -> None:
    while select.select([sys.stdin], [], [], 0.0)[0]:
        sys.stdin.read(1)


def _status_lines(
    logger: RunLogger,
    split: str,
    activation_index: int,
    activation_count: int,
    activation: Activation | None,
    board_state: BoardState,
    target_pose: Pose2D | None,
    packets_sent: int,
    message: str,
) -> list[str]:
    lines = [
        "  Scripted SysID Collection",
        "  -------------------------",
        f"  split: {split}",
        f"  run dir: {logger.run_dir}",
        f"  packets sent: {packets_sent}",
        f"  calibration: {'ok' if board_state.calibration.valid else 'waiting'} ({board_state.calibration.samples_collected}/{board_state.calibration.required_samples})",
        f"  target visible: {'yes' if target_pose is not None and target_pose.visible else 'no'}",
    ]
    if activation is not None:
        lines.extend(
            [
                f"  activation: {activation_index + 1}/{activation_count} {activation.name}",
                f"  command: left={activation.left:+.2f} right={activation.right:+.2f} for {activation.active_duration_s:.2f}s",
                f"  reset pose: {activation.start_pose}",
                f"  segment timing: pre={activation.pre_roll_s:.2f}s active={activation.active_duration_s:.2f}s post={activation.post_roll_s:.2f}s",
            ]
        )
    lines.extend(
        [
            f"  message: {message}",
            "  controls: Enter run | s skip | q quit | space stop",
        ]
    )
    return lines


def _draw_status(lines: list[str]) -> None:
    sys.stdout.write("\033[2J\033[H")
    for line in lines:
        sys.stdout.write(f"{line}\n")
    sys.stdout.flush()


def _wait_for_activation_start(
    tracker: BoardTracker,
    logger: RunLogger,
    teleop: TeleopController,
    config: TrackerConfig,
    split: str,
    activation_index: int,
    activations: list[Activation],
    show_raw: bool,
    show_arena: bool,
) -> str:
    activation = activations[activation_index]
    teleop.set_tank_command(0.0, 0.0)
    _drain_stdin()
    while True:
        frame, board_state = tracker.process_next_frame()
        target_pose = _target_pose(board_state, config.teleop.robot_tag_id, config)
        teleop.pop_events()
        _render_preview(tracker, frame, board_state, show_raw, show_arena)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return "quit"
        if key == ord("s"):
            return "skip"
        if key == ord(" "):
            teleop.set_tank_command(0.0, 0.0)

        lines = _status_lines(
            logger,
            split,
            activation_index,
            len(activations),
            activation,
            board_state,
            target_pose,
            teleop.snapshot().packets_sent,
            "Reset robot, then press Enter when calibration and visibility are good.",
        )
        _draw_status(lines)

        if select.select([sys.stdin], [], [], 0.0)[0]:
            key_text = sys.stdin.read(1)
            if key_text in ("q", "\x1b"):
                return "quit"
            if key_text == "s":
                return "skip"
            if key_text == " ":
                teleop.set_tank_command(0.0, 0.0)
                continue
            if key_text in ("\n", "\r"):
                ready = (
                    board_state.calibration.valid
                    and target_pose is not None
                    and target_pose.visible
                )
                if ready:
                    return "run"


def _run_activation(
    tracker: BoardTracker,
    logger: RunLogger,
    teleop: TeleopController,
    config: TrackerConfig,
    activation: Activation,
    activation_index: int,
    activation_count: int,
    show_raw: bool,
    show_arena: bool,
    stop_on_target_loss_s: float,
    frame_index: int,
    split: str,
) -> tuple[str, str | None, int]:
    teleop.pop_events()
    start_time = time.monotonic()
    last_visible_time = start_time
    total_duration_s = (
        activation.pre_roll_s + activation.active_duration_s + activation.post_roll_s
    )
    while True:
        frame, board_state = tracker.process_next_frame()
        now = time.monotonic()
        elapsed = now - start_time
        if elapsed < activation.pre_roll_s:
            phase = "pre"
            teleop.set_tank_command(0.0, 0.0)
        elif elapsed < activation.pre_roll_s + activation.active_duration_s:
            phase = "active"
            teleop.set_tank_command(activation.left, activation.right)
        else:
            phase = "post"
            teleop.set_tank_command(0.0, 0.0)
        target_pose = _target_pose(board_state, config.teleop.robot_tag_id, config)
        if (
            target_pose is not None
            and target_pose.visible
            and board_state.calibration.valid
        ):
            last_visible_time = now
        elif now - last_visible_time > stop_on_target_loss_s:
            teleop.set_tank_command(0.0, 0.0)
            return "aborted", "target visibility lost", frame_index

        command_state = teleop.snapshot()
        logger.log_sample(frame_index, now, board_state, command_state)
        for event in teleop.pop_events():
            logger.log_command(event)

        _render_preview(tracker, frame, board_state, show_raw, show_arena)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            teleop.set_tank_command(0.0, 0.0)
            return "quit", "operator requested quit", frame_index + 1
        if key == ord(" "):
            teleop.set_tank_command(0.0, 0.0)
            return "aborted", "operator emergency stop", frame_index + 1

        lines = _status_lines(
            logger,
            split,
            activation_index,
            activation_count,
            activation,
            board_state,
            target_pose,
            command_state.packets_sent,
            f"Running {activation.name}: phase={phase} {elapsed:.2f}/{total_duration_s:.2f}s",
        )
        _draw_status(lines)

        if elapsed >= total_duration_s:
            teleop.set_tank_command(0.0, 0.0)
            return "completed", None, frame_index + 1
        frame_index += 1


def _print_activation_list(split: str, activations: list[Activation]) -> None:
    print(f"split={split} activations={len(activations)}")
    for index, activation in enumerate(activations, start=1):
        print(
            f"{index:02d}. {activation.name}: left={activation.left:+.2f} "
            f"right={activation.right:+.2f} active={activation.active_duration_s:.2f}s "
            f"pre={activation.pre_roll_s:.2f}s post={activation.post_roll_s:.2f}s "
            f"start_pose={activation.start_pose}"
        )


def main() -> None:
    args = parse_args()
    activations = build_activations(args.split, args.activation_scale)
    if args.list:
        _print_activation_list(args.split, activations)
        return

    if any(
        max(abs(activation.left), abs(activation.right)) > SAFE_LINEAR_COMMAND
        for activation in activations
    ):
        raise ValueError(
            f"Hard-coded activations exceeded the safety limit of {SAFE_LINEAR_COMMAND:.2f}."
        )

    config = TrackerConfig()
    config.teleop.run_label = f"sysid_{args.split}_scripted"

    show_raw, show_arena = _preview_flags(args.show_preview)
    if show_raw:
        cv2.namedWindow(RAW_WINDOW_NAME, cv2.WINDOW_NORMAL)
    if show_arena:
        cv2.namedWindow(ARENA_WINDOW_NAME, cv2.WINDOW_NORMAL)

    tracker = BoardTracker(config)
    teleop = TeleopController(config.teleop)
    logger = RunLogger(config, root=Path("data") / args.split)
    _update_metadata(logger, args.split, activations)
    segments_path = logger.run_dir / "segments.jsonl"
    segments_file = segments_path.open("w", encoding="utf-8")

    if not args.yes:
        _print_activation_list(args.split, activations)
        response = input(
            "Press Enter to start the scripted collector, or type 'q' to exit: "
        )
        if response.strip().lower() == "q":
            segments_file.close()
            logger.close()
            tracker.close()
            cv2.destroyAllWindows()
            return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    teleop.start()

    try:
        tty.setcbreak(fd)
        frame_index = 0
        for activation_index, activation in enumerate(activations):
            action = _wait_for_activation_start(
                tracker,
                logger,
                teleop,
                config,
                args.split,
                activation_index,
                activations,
                show_raw,
                show_arena,
            )
            if action == "quit":
                break
            if action == "skip":
                _write_segment_record(
                    segments_file,
                    ActivationRecord(
                        activation_index=activation_index,
                        activation_name=activation.name,
                        status="skipped",
                        scheduled_left=activation.left,
                        scheduled_right=activation.right,
                        active_duration_s=activation.active_duration_s,
                        pre_roll_s=activation.pre_roll_s,
                        post_roll_s=activation.post_roll_s,
                        start_pose=activation.start_pose,
                        started_at=None,
                        ended_at=None,
                        wall_started_at=None,
                        wall_ended_at=None,
                        abort_reason=None,
                    ),
                )
                continue

            started_at = time.monotonic()
            wall_started_at = datetime.now().isoformat()
            status, abort_reason, frame_index = _run_activation(
                tracker,
                logger,
                teleop,
                config,
                activation,
                activation_index,
                len(activations),
                show_raw,
                show_arena,
                args.stop_on_target_loss_s,
                frame_index,
                args.split,
            )
            ended_at = time.monotonic()
            wall_ended_at = datetime.now().isoformat()
            _write_segment_record(
                segments_file,
                ActivationRecord(
                    activation_index=activation_index,
                    activation_name=activation.name,
                    status=status,
                    scheduled_left=activation.left,
                    scheduled_right=activation.right,
                    active_duration_s=activation.active_duration_s,
                    pre_roll_s=activation.pre_roll_s,
                    post_roll_s=activation.post_roll_s,
                    start_pose=activation.start_pose,
                    started_at=started_at,
                    ended_at=ended_at,
                    wall_started_at=wall_started_at,
                    wall_ended_at=wall_ended_at,
                    abort_reason=abort_reason,
                ),
            )
            if status == "quit":
                break
    finally:
        teleop.set_tank_command(0.0, 0.0)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        teleop.stop()
        segments_file.close()
        logger.close()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
