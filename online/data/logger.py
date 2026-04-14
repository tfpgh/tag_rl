from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from online.control.teleop import CommandEvent, CommandState
from online.core.config import TrackerConfig
from online.core.state import BoardState, ObstacleState, Pose2D, TagDetection


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _pose_dict(pose: Pose2D | None) -> dict[str, Any] | None:
    if pose is None:
        return None
    return {
        "x_m": pose.x_m,
        "y_m": pose.y_m,
        "yaw_rad": pose.yaw_rad,
        "visible": pose.visible,
        "last_seen_timestamp": pose.last_seen_timestamp,
    }


def _obstacle_dict(obstacle: ObstacleState) -> dict[str, Any]:
    return {
        "tag_id": obstacle.tag_id,
        "size_m": obstacle.size_m,
        "pose": _pose_dict(obstacle.pose),
    }


def _detection_dict(detection: TagDetection) -> dict[str, Any]:
    return {
        "tag_id": detection.tag_id,
        "center_px": detection.center_px.tolist(),
        "decision_margin": detection.decision_margin,
        "hamming": detection.hamming,
    }


class RunLogger:
    def __init__(self, config: TrackerConfig, root: Path | None = None) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = config.teleop.run_label.replace(" ", "_")
        base = root or Path("data")
        self.run_dir = base / f"{timestamp}_{label}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._samples_path = self.run_dir / "samples.jsonl"
        self._commands_path = self.run_dir / "commands.jsonl"
        self._samples_file = self._samples_path.open("w", encoding="utf-8")
        self._commands_file = self._commands_path.open("w", encoding="utf-8")
        self._robot_tag_id = config.teleop.robot_tag_id
        self._chaser_tag_id = config.arena.chaser_tag_id
        self._evader_tag_id = config.arena.evader_tag_id
        self._write_metadata(config)

    def close(self) -> None:
        self._samples_file.close()
        self._commands_file.close()

    def log_sample(
        self,
        frame_index: int,
        monotonic_time: float,
        board_state: BoardState,
        command_state: CommandState,
    ) -> None:
        record = {
            "frame_index": frame_index,
            "monotonic_time": monotonic_time,
            "wall_timestamp": board_state.timestamp,
            "calibration_valid": board_state.calibration.valid,
            "tracker": {
                "fps": board_state.stats.fps,
                "capture_ms": board_state.stats.capture_ms,
                "detector_ms": board_state.stats.detector_ms,
                "tracking_ms": board_state.stats.tracking_ms,
                "loop_ms": board_state.stats.loop_ms,
                "visible_tags": board_state.stats.visible_tags,
            },
            "chaser": _pose_dict(board_state.chaser),
            "evader": _pose_dict(board_state.evader),
            "target_robot": _pose_dict(self._target_pose(board_state)),
            "obstacles": [
                _obstacle_dict(obstacle) for obstacle in board_state.obstacles
            ],
            "detections": [
                _detection_dict(detection) for detection in board_state.detections
            ],
            "command": asdict(command_state),
        }
        self._samples_file.write(json.dumps(record) + "\n")
        self._samples_file.flush()

    def log_command(self, event: CommandEvent) -> None:
        self._commands_file.write(json.dumps(asdict(event)) + "\n")
        self._commands_file.flush()

    def _target_pose(self, board_state: BoardState) -> Pose2D | None:
        if self._robot_tag_id == self._chaser_tag_id:
            return board_state.chaser
        if self._robot_tag_id == self._evader_tag_id:
            return board_state.evader
        return None

    def _write_metadata(self, config: TrackerConfig) -> None:
        metadata = {
            "created_at": datetime.now().isoformat(),
            "teleop": asdict(config.teleop),
            "camera": asdict(config.camera),
            "detector": asdict(config.detector),
            "arena": asdict(config.arena),
            "world_units": "meters",
            "tracker": {
                "calibration_frames": config.calibration_frames,
                "use_roi_tracking": config.use_roi_tracking,
                "roi_padding_scale": config.roi_padding_scale,
                "min_roi_size_px": config.min_roi_size_px,
                "max_roi_size_px": config.max_roi_size_px,
            },
        }
        (self.run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    def update_metadata(self, updates: dict[str, Any]) -> None:
        metadata_path = self.run_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _deep_update(metadata, updates)
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
