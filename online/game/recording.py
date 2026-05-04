from __future__ import annotations

import time
from datetime import datetime
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from online.control.teleop import CommandEvent, CommandState
from online.data.logger import RunLogger
from online.game.config import GameRuntimeConfig
from online.game.control import RobotCommand


@dataclass(slots=True)
class AgentInternalsSample:
    value: float
    action_mean: np.ndarray
    action_stddev: np.ndarray
    action_raw: np.ndarray
    hidden_state: np.ndarray
    observation: np.ndarray


def _command_state(
    command: RobotCommand, packets_sent: int, monotonic_time: float
) -> CommandState:
    return CommandState(
        monotonic_time=monotonic_time,
        linear=0.5 * (command.left + command.right),
        angular=0.5 * (command.right - command.left),
        left=command.left,
        right=command.right,
        left_int16=command.left_int16,
        right_int16=command.right_int16,
        packets_sent=packets_sent,
    )


class LiveGameRecorder:
    def __init__(self, config: GameRuntimeConfig, checkpoint_path: str) -> None:
        self._config = config
        self._checkpoint_path = checkpoint_path
        self._recording_split: str = "off"
        self._active_split: str | None = None
        self._chaser_logger: RunLogger | None = None
        self._evader_logger: RunLogger | None = None
        self._packets_sent = 0
        self._game_index = 0
        self._last_recorded_time: float | None = None

    @property
    def recording_split(self) -> str:
        return self._recording_split

    @property
    def active(self) -> bool:
        return self._active_split is not None

    def set_recording_split(self, split: str) -> None:
        if split not in {"off", "train", "eval", "showcase"}:
            raise ValueError(f"Unsupported recording split: {split}")
        self._recording_split = split

    def start_game(self, board_state: Any) -> None:
        if self._recording_split == "off" or self.active:
            return
        split_root = (
            Path(self._config.recording_root)
            / self._recording_split
            / self._config.recording_games_dirname
        )
        split_root.mkdir(parents=True, exist_ok=True)
        self._game_index += 1
        self._packets_sent = 0
        self._last_recorded_time = None
        self._active_split = self._recording_split
        self._chaser_logger = self._make_logger("chaser", split_root)
        self._evader_logger = self._make_logger("evader", split_root)
        metadata = {
            "collection": {
                "mode": "game",
                "split": self._active_split,
                "created_at": datetime.now().isoformat(),
                "checkpoint_path": self._checkpoint_path,
                "game_index": self._game_index,
            },
            "game": {
                "match_duration_s": self._config.match_duration_s,
                "action_output_scale": self._config.action_output_scale,
                "chaser": asdict(self._config.chaser),
                "evader": asdict(self._config.evader),
                "initial_visible_tags": board_state.stats.visible_tags,
            },
        }
        self._chaser_logger.update_metadata(metadata)
        self._evader_logger.update_metadata(metadata)

    def record_step(
        self,
        frame_index: int,
        monotonic_time: float,
        board_state: Any,
        chaser_command: RobotCommand,
        evader_command: RobotCommand,
        chaser_internals: AgentInternalsSample | None = None,
        evader_internals: AgentInternalsSample | None = None,
    ) -> None:
        if (
            not self.active
            or self._chaser_logger is None
            or self._evader_logger is None
        ):
            return
        self._packets_sent += 1
        self._last_recorded_time = monotonic_time
        chaser_state = _command_state(
            chaser_command, self._packets_sent, monotonic_time
        )
        evader_state = _command_state(
            evader_command, self._packets_sent, monotonic_time
        )
        self._chaser_logger.log_sample(
            frame_index, monotonic_time, board_state, chaser_state
        )
        self._evader_logger.log_sample(
            frame_index, monotonic_time, board_state, evader_state
        )
        self._chaser_logger.log_command(
            CommandEvent(
                monotonic_time=monotonic_time,
                left_int16=chaser_command.left_int16,
                right_int16=chaser_command.right_int16,
                packets_sent=self._packets_sent,
            )
        )
        self._evader_logger.log_command(
            CommandEvent(
                monotonic_time=monotonic_time,
                left_int16=evader_command.left_int16,
                right_int16=evader_command.right_int16,
                packets_sent=self._packets_sent,
            )
        )
        if (
            self._active_split == "showcase"
            and chaser_internals is not None
            and evader_internals is not None
        ):
            self._chaser_logger.log_policy_internals(
                frame_index=frame_index,
                monotonic_time=monotonic_time,
                value=chaser_internals.value,
                action_mean=chaser_internals.action_mean,
                action_stddev=chaser_internals.action_stddev,
                action_raw=chaser_internals.action_raw,
                hidden_state=chaser_internals.hidden_state,
                observation=chaser_internals.observation,
            )
            self._evader_logger.log_policy_internals(
                frame_index=frame_index,
                monotonic_time=monotonic_time,
                value=evader_internals.value,
                action_mean=evader_internals.action_mean,
                action_stddev=evader_internals.action_stddev,
                action_raw=evader_internals.action_raw,
                hidden_state=evader_internals.hidden_state,
                observation=evader_internals.observation,
            )

    def finish_game(self, reason: str) -> list[str]:
        if not self.active:
            return []
        run_dirs = [
            str(self._chaser_logger.run_dir) if self._chaser_logger is not None else "",
            str(self._evader_logger.run_dir) if self._evader_logger is not None else "",
        ]
        updates = {
            "collection": {
                "mode": "game",
                "split": self._active_split,
                "closed_at": datetime.now().isoformat(),
                "end_reason": reason,
                "last_recorded_monotonic_time": self._last_recorded_time,
            }
        }
        if self._chaser_logger is not None:
            self._chaser_logger.update_metadata(updates)
            self._chaser_logger.close()
            self._chaser_logger = None
        if self._evader_logger is not None:
            self._evader_logger.update_metadata(updates)
            self._evader_logger.close()
            self._evader_logger = None
        self._active_split = None
        self._packets_sent = 0
        self._last_recorded_time = None
        return [run_dir for run_dir in run_dirs if run_dir]

    def _make_logger(self, robot_name: str, root: Path) -> RunLogger:
        tracker_config = replace(
            self._config.tracker,
            teleop=replace(
                self._config.tracker.teleop,
                robot_tag_id=(
                    self._config.chaser.tag_id
                    if robot_name == "chaser"
                    else self._config.evader.tag_id
                ),
                run_label=(
                    f"tag_game_{self._recording_split}_game{self._game_index:03d}_{robot_name}"
                ),
            ),
        )
        return RunLogger(tracker_config, root=root)
