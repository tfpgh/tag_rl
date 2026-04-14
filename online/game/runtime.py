from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import jax
import numpy as np

from online.game.config import GameRuntimeConfig
from online.game.control import DualRobotController, RobotCommand
from online.game.inference import DualPolicyRunner, sync_live_env_config
from online.game.observations import LiveObservations, build_live_observations
from online.tracking.tracker import BoardTracker


@dataclass(slots=True)
class RuntimeEvent:
    timestamp: float
    message: str


class TagGameRuntime:
    def __init__(self, config: GameRuntimeConfig, udp_enabled: bool = True) -> None:
        self.config = config
        self.policy_runner = DualPolicyRunner(config.checkpoint_path)
        self.env_config = sync_live_env_config(
            self.policy_runner.env_config,
            config.tracker.arena.board_width_m,
            config.tracker.arena.board_height_m,
            config.tracker.arena.obstacle_size_m,
        )
        self.control_hz = config.control_hz or float(self.env_config.action_frequency)
        self.match_duration_s = config.match_duration_s or float(
            self.env_config.episode_max_length
        )
        self.tag_distance = (
            2.0 * self.env_config.agent_radius * self.env_config.tag_distance_factor
        )
        self.tracker = BoardTracker(config.tracker)
        self.controller = DualRobotController(config.chaser, config.evader, udp_enabled)
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._mode = config.modes.idle
        self._phase = config.phases.waiting
        self._status = "Booting"
        self._match_start_monotonic: float | None = None
        self._frame_timestamp: float | None = None
        self._latest_snapshot: dict[str, Any] = {}
        self._latest_jpeg: bytes | None = None
        self._latest_raw_action = {
            "chaser": RobotCommand(0.0, 0.0).as_dict(),
            "evader": RobotCommand(0.0, 0.0).as_dict(),
        }
        self._latest_final_action = {
            "chaser": RobotCommand(0.0, 0.0).as_dict(),
            "evader": RobotCommand(0.0, 0.0).as_dict(),
        }
        self._latest_observation: LiveObservations | None = None
        self._last_ready_time: float | None = None
        self._events: deque[RuntimeEvent] = deque(maxlen=20)
        self._push_event("runtime_initialized")

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.controller.close()
        self.tracker.close()

    def handle_action(self, action: str) -> None:
        actions = self.config.actions
        modes = self.config.modes
        phases = self.config.phases
        with self._lock:
            if action == actions.arm:
                if self._mode != modes.estop:
                    self._mode = modes.armed
                    self._status = "System armed"
                    self._push_event("armed")
            elif action == actions.disarm:
                self._mode = modes.idle
                self._phase = phases.ready
                self._match_start_monotonic = None
                self.policy_runner.reset()
                self._status = "System disarmed"
                self._push_event("disarmed")
            elif action == actions.start:
                if self._mode == modes.armed:
                    self._mode = modes.running
                    self._phase = phases.active
                    self._match_start_monotonic = None
                    self.policy_runner.reset()
                    self._status = "Waiting for first active tick"
                    self._push_event("game_started")
            elif action == actions.stop:
                self._mode = modes.armed
                self._phase = phases.ready
                self._match_start_monotonic = None
                self.policy_runner.reset()
                self._status = "Game stopped"
                self._push_event("game_stopped")
            elif action == actions.reset:
                self._phase = phases.ready
                self._match_start_monotonic = None
                self.policy_runner.reset()
                self._status = "Game state reset"
                self._push_event("game_reset")
            elif action == actions.estop:
                self._mode = modes.estop
                self._phase = phases.stopped
                self._match_start_monotonic = None
                self.policy_runner.reset()
                self._status = "Emergency stop engaged"
                self._push_event("estop")
            elif action == actions.clear_estop:
                self._mode = modes.idle
                self._phase = phases.ready
                self._match_start_monotonic = None
                self.policy_runner.reset()
                self._status = "Emergency stop cleared"
                self._push_event("estop_cleared")
            else:
                raise ValueError(f"Unsupported action: {action}")

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest_snapshot)

    def get_jpeg_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def _run_loop(self) -> None:
        modes = self.config.modes
        interval = 1.0 / self.control_hz
        while self._running:
            loop_started = time.perf_counter()
            try:
                frame, board_state = self.tracker.process_next_frame()
                now = time.monotonic()
                snapshot = self._step_runtime(now, frame, board_state)
                jpeg = self._render_dashboard_frame(frame, board_state, snapshot)
                with self._lock:
                    self._latest_snapshot = snapshot
                    self._latest_jpeg = jpeg
            except Exception as exc:
                self.controller.zero_all()
                with self._lock:
                    self._mode = modes.estop
                    self._status = f"Runtime error: {exc}"
                    self._latest_snapshot = {
                        "runtime": {
                            "mode": self._mode,
                            "phase": self._phase,
                            "status": self._status,
                        }
                    }
                time.sleep(0.10)
            elapsed = time.perf_counter() - loop_started
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _step_runtime(
        self, now: float, frame: np.ndarray, board_state
    ) -> dict[str, Any]:
        del frame
        modes = self.config.modes
        phases = self.config.phases
        self._frame_timestamp = board_state.timestamp
        ready, reason = self._ready_state(now, board_state)
        if ready:
            self._last_ready_time = now
            if self._mode != modes.running:
                self._phase = phases.ready
        elif self._mode == modes.running:
            self._mode = modes.armed
            self._phase = phases.ready
            self.policy_runner.reset()
            self._status = reason
            self._push_event("tracking_not_ready")

        final_chaser = RobotCommand(0.0, 0.0)
        final_evader = RobotCommand(0.0, 0.0)
        raw_chaser = RobotCommand(0.0, 0.0)
        raw_evader = RobotCommand(0.0, 0.0)
        episode_progress = self._episode_progress(now)
        observations = None

        if self._mode == modes.running and ready:
            if self._match_start_monotonic is None:
                self._match_start_monotonic = now
                episode_progress = 0.0
            observations = build_live_observations(
                board_state, self.env_config, episode_progress
            )
            if observations is None:
                self._mode = modes.armed
                self._phase = phases.ready
                self.policy_runner.reset()
                self._status = "Observation build failed"
                self._push_event("observation_build_failed")
            else:
                outputs = self.policy_runner.step(
                    observations.chaser, observations.evader
                )
                raw_chaser = RobotCommand(
                    left=float(jax.device_get(outputs.chaser_action[0])),
                    right=float(jax.device_get(outputs.chaser_action[1])),
                )
                raw_evader = RobotCommand(
                    left=float(jax.device_get(outputs.evader_action[0])),
                    right=float(jax.device_get(outputs.evader_action[1])),
                )
                final_chaser, final_evader, terminal_reason = self._apply_game_rules(
                    now, board_state, raw_chaser, raw_evader
                )
                if terminal_reason is not None:
                    self._mode = modes.armed
                    self._status = terminal_reason
                    self.policy_runner.reset()

        if self._mode == modes.estop:
            final_chaser = RobotCommand(0.0, 0.0)
            final_evader = RobotCommand(0.0, 0.0)
            self._phase = phases.stopped

        self.controller.send(final_chaser, final_evader)
        self._latest_observation = observations
        self._latest_raw_action = {
            "chaser": raw_chaser.as_dict(),
            "evader": raw_evader.as_dict(),
        }
        self._latest_final_action = {
            "chaser": final_chaser.as_dict(),
            "evader": final_evader.as_dict(),
        }
        if self._mode not in (modes.running, modes.estop):
            self.controller.zero_all()
        return self._build_snapshot(now, board_state, ready, reason)

    def _episode_progress(self, now: float) -> float:
        if self._match_start_monotonic is None:
            return 0.0
        return max(
            0.0, min(1.0, (now - self._match_start_monotonic) / self.match_duration_s)
        )

    def _ready_state(self, now: float, board_state) -> tuple[bool, str]:
        if self._mode == self.config.modes.estop:
            return False, "Emergency stop engaged"
        if not board_state.calibration.valid:
            return False, "Arena calibration incomplete"
        if board_state.chaser is None or board_state.evader is None:
            return False, "Missing robot detections"
        if not board_state.chaser.visible or not board_state.evader.visible:
            if (
                self._last_ready_time is not None
                and now - self._last_ready_time <= self.config.target_loss_timeout_s
            ):
                return True, "Tracking grace window"
            return False, "Robot visibility lost"
        if (
            self._frame_timestamp is not None
            and now - self._frame_timestamp > self.config.stale_frame_timeout_s
        ):
            return False, "Tracker frame stale"
        return True, "Ready"

    def _apply_game_rules(
        self,
        now: float,
        board_state,
        chaser: RobotCommand,
        evader: RobotCommand,
    ) -> tuple[RobotCommand, RobotCommand, str | None]:
        if self._match_start_monotonic is None:
            return RobotCommand(0.0, 0.0), RobotCommand(0.0, 0.0), None
        elapsed = now - self._match_start_monotonic
        if elapsed < float(self.env_config.chaser_freeze_seconds):
            chaser = RobotCommand(0.0, 0.0)
        if board_state.chaser is None or board_state.evader is None:
            return RobotCommand(0.0, 0.0), RobotCommand(0.0, 0.0), "Tracking lost"
        distance = float(
            np.hypot(
                board_state.chaser.x_m - board_state.evader.x_m,
                board_state.chaser.y_m - board_state.evader.y_m,
            )
        )
        if distance < self.tag_distance:
            self._phase = self.config.phases.tagged
            self._push_event("tagged")
            return RobotCommand(0.0, 0.0), RobotCommand(0.0, 0.0), "Tagged"
        if elapsed >= self.match_duration_s:
            self._phase = self.config.phases.time_up
            self._push_event("time_up")
            return RobotCommand(0.0, 0.0), RobotCommand(0.0, 0.0), "Time up"
        self._phase = self.config.phases.active
        self._status = "Game active"
        return chaser, evader, None

    def _build_snapshot(
        self, now: float, board_state, ready: bool, ready_reason: str
    ) -> dict[str, Any]:
        frame_age = (
            None if self._frame_timestamp is None else now - self._frame_timestamp
        )
        chaser_pose = (
            None
            if board_state.chaser is None
            else {
                "x_m": board_state.chaser.x_m,
                "y_m": board_state.chaser.y_m,
                "yaw_rad": board_state.chaser.yaw_rad,
            }
        )
        evader_pose = (
            None
            if board_state.evader is None
            else {
                "x_m": board_state.evader.x_m,
                "y_m": board_state.evader.y_m,
                "yaw_rad": board_state.evader.yaw_rad,
            }
        )
        return {
            "runtime": {
                "mode": self._mode,
                "phase": self._phase,
                "status": self._status,
                "ready": ready,
                "ready_reason": ready_reason,
                "control_hz": self.control_hz,
                "frame_age_s": frame_age,
            },
            "game": {
                "elapsed_s": 0.0
                if self._match_start_monotonic is None
                else now - self._match_start_monotonic,
                "duration_s": self.match_duration_s,
                "episode_progress": self._episode_progress(now),
                "tag_distance_m": self.tag_distance,
            },
            "tracker": {
                "fps": board_state.stats.fps,
                "capture_ms": board_state.stats.capture_ms,
                "detector_ms": board_state.stats.detector_ms,
                "tracking_ms": board_state.stats.tracking_ms,
                "loop_ms": board_state.stats.loop_ms,
                "visible_tags": board_state.stats.visible_tags,
                "calibration_valid": board_state.calibration.valid,
            },
            "robots": {
                "chaser_pose": chaser_pose,
                "evader_pose": evader_pose,
                "chaser_command_raw": self._latest_raw_action["chaser"],
                "evader_command_raw": self._latest_raw_action["evader"],
                "chaser_command_sent": self._latest_final_action["chaser"],
                "evader_command_sent": self._latest_final_action["evader"],
            },
            "policy": {
                "chaser": self._policy_summary(
                    None
                    if self._latest_observation is None
                    else self._latest_observation.chaser,
                    self._latest_raw_action["chaser"],
                ),
                "evader": self._policy_summary(
                    None
                    if self._latest_observation is None
                    else self._latest_observation.evader,
                    self._latest_raw_action["evader"],
                ),
            },
            "events": [asdict(event) for event in self._events],
        }

    def _policy_summary(
        self, observation: Any, action: dict[str, float | int]
    ) -> dict[str, Any]:
        if observation is None:
            return {"observation_ready": False, "action": action}
        observation_host = np.asarray(jax.device_get(observation), dtype=np.float32)
        n_rays = self.env_config.n_rays
        ray_dist = observation_host[:n_rays]
        ray_type = observation_host[n_rays : 2 * n_rays]
        return {
            "observation_ready": True,
            "episode_progress": float(observation_host[-1]),
            "mean_ray_distance": float(np.mean(ray_dist)),
            "agent_hit_fraction": float(np.mean(ray_type)),
            "action": action,
        }

    def _render_dashboard_frame(
        self, frame: np.ndarray, board_state, snapshot: dict[str, Any]
    ) -> bytes | None:
        raw_view, arena_view = self.tracker.render_debug_views(frame, board_state)
        target_height = min(raw_view.shape[0], arena_view.shape[0])
        raw_small = cv2.resize(
            raw_view,
            (int(raw_view.shape[1] * target_height / raw_view.shape[0]), target_height),
        )
        arena_small = cv2.resize(
            arena_view,
            (
                int(arena_view.shape[1] * target_height / arena_view.shape[0]),
                target_height,
            ),
        )
        combined = cv2.hconcat([raw_small, arena_small])
        overlay_lines = [
            f"mode={snapshot['runtime']['mode']} phase={snapshot['runtime']['phase']}",
            f"status={snapshot['runtime']['status']}",
            f"ready={snapshot['runtime']['ready']} fps={snapshot['tracker']['fps']:.1f}",
            "chaser L/R={left:+.2f}/{right:+.2f} evader L/R={eleft:+.2f}/{eright:+.2f}".format(
                left=self._latest_final_action["chaser"]["left"],
                right=self._latest_final_action["chaser"]["right"],
                eleft=self._latest_final_action["evader"]["left"],
                eright=self._latest_final_action["evader"]["right"],
            ),
        ]
        for index, line in enumerate(overlay_lines):
            y = 28 + index * 26
            cv2.putText(
                combined,
                line,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (235, 240, 245),
                2,
                cv2.LINE_AA,
            )
        ok, encoded = cv2.imencode(
            ".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        )
        if not ok:
            return None
        return encoded.tobytes()

    def _push_event(self, message: str) -> None:
        self._events.appendleft(RuntimeEvent(timestamp=time.time(), message=message))
