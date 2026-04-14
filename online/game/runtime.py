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
from online.game.observations import LiveObservationBuilder, LiveObservations
from online.game.recording import LiveGameRecorder
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
        self.observation_builder = LiveObservationBuilder(self.env_config)
        self.observation_builder.warmup()
        self.control_hz = config.control_hz or float(self.env_config.action_frequency)
        self.match_duration_s = config.match_duration_s or float(
            self.env_config.episode_max_length
        )
        self.max_steps = int(self.env_config.episode_max_length) * int(
            self.env_config.action_frequency
        )
        self.freeze_steps = int(self.env_config.chaser_freeze_seconds) * int(
            self.env_config.action_frequency
        )
        self.tag_distance = (
            2.0 * self.env_config.agent_radius * self.env_config.tag_distance_factor
        )
        self.tracker = BoardTracker(config.tracker)
        self.controller = DualRobotController(config.chaser, config.evader, udp_enabled)
        self.recorder = LiveGameRecorder(config, config.checkpoint_path)
        self._tracker_thread: threading.Thread | None = None
        self._control_thread: threading.Thread | None = None
        self._render_thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._mode = config.modes.idle
        self._phase = config.phases.waiting
        self._status = "Booting"
        self._match_start_monotonic: float | None = None
        self._match_step_count = 0
        self._frame_timestamp: float | None = None
        self._latest_snapshot: dict[str, Any] = {}
        self._latest_jpeg: bytes | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_board_state = None
        self._last_control_loop_monotonic: float | None = None
        self._last_control_loop_dt_s: float | None = None
        self._last_render_loop_monotonic: float | None = None
        self._last_render_loop_dt_s: float | None = None
        self._last_observation_build_ms: float | None = None
        self._last_inference_ms: float | None = None
        self._last_send_ms: float | None = None
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
        self._recording_frame_index = 0
        self._push_event("runtime_initialized")

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tracker_thread = threading.Thread(target=self._tracker_loop, daemon=True)
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._tracker_thread.start()
        self._control_thread.start()
        self._render_thread.start()

    def stop(self) -> None:
        self._running = False
        for thread in (self._tracker_thread, self._control_thread, self._render_thread):
            if thread is not None:
                thread.join(timeout=2.0)
        self._finish_recording("runtime_stopped")
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
                self._finish_recording("disarmed")
                self._mode = modes.idle
                self._phase = phases.ready
                self._match_start_monotonic = None
                self._match_step_count = 0
                self.policy_runner.reset()
                self._status = "System disarmed"
                self._push_event("disarmed")
            elif action == actions.start:
                if self._mode == modes.armed:
                    self._mode = modes.running
                    self._phase = phases.active
                    self._match_start_monotonic = None
                    self._match_step_count = 0
                    self.policy_runner.reset()
                    self._status = "Waiting for first active tick"
                    self._push_event("game_started")
            elif action == actions.stop:
                self._finish_recording("game_stopped")
                self._mode = modes.armed
                self._phase = phases.ready
                self._match_start_monotonic = None
                self._match_step_count = 0
                self.policy_runner.reset()
                self._status = "Game stopped"
                self._push_event("game_stopped")
            elif action == actions.reset:
                self._finish_recording("game_reset")
                self._phase = phases.ready
                self._match_start_monotonic = None
                self._match_step_count = 0
                self.policy_runner.reset()
                self._status = "Game state reset"
                self._push_event("game_reset")
            elif action == actions.estop:
                self._finish_recording("estop")
                self._mode = modes.estop
                self._phase = phases.stopped
                self._match_start_monotonic = None
                self._match_step_count = 0
                self.policy_runner.reset()
                self._status = "Emergency stop engaged"
                self._push_event("estop")
            elif action == actions.clear_estop:
                self._mode = modes.idle
                self._phase = phases.ready
                self._match_start_monotonic = None
                self._match_step_count = 0
                self.policy_runner.reset()
                self._status = "Emergency stop cleared"
                self._push_event("estop_cleared")
            elif action == actions.record_off:
                self.recorder.set_recording_split("off")
                self._finish_recording("recording_disabled")
                self._status = "Recording disabled"
                self._push_event("recording_off")
            elif action == actions.record_train:
                self.recorder.set_recording_split("train")
                self._status = "Recording armed for train games"
                self._push_event("recording_train")
            elif action == actions.record_eval:
                self.recorder.set_recording_split("eval")
                self._status = "Recording armed for eval games"
                self._push_event("recording_eval")
            else:
                raise ValueError(f"Unsupported action: {action}")

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest_snapshot)

    def get_jpeg_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def _tracker_loop(self) -> None:
        while self._running:
            try:
                frame, board_state = self.tracker.process_next_frame()
                with self._lock:
                    self._latest_frame = frame
                    self._latest_board_state = board_state
            except Exception as exc:
                self.controller.zero_all()
                with self._lock:
                    self._finish_recording("tracker_error")
                    self._mode = self.config.modes.estop
                    self._status = f"Tracker error: {exc}"
                    self._latest_snapshot = {
                        "runtime": {
                            "mode": self._mode,
                            "phase": self._phase,
                            "status": self._status,
                        }
                    }
                time.sleep(0.10)

    def _control_loop(self) -> None:
        modes = self.config.modes
        interval = 1.0 / self.control_hz
        while self._running:
            loop_started = time.perf_counter()
            try:
                with self._lock:
                    board_state = self._latest_board_state
                if board_state is None:
                    time.sleep(0.01)
                    continue
                now = time.monotonic()
                if self._last_control_loop_monotonic is not None:
                    self._last_control_loop_dt_s = (
                        now - self._last_control_loop_monotonic
                    )
                self._last_control_loop_monotonic = now
                snapshot = self._step_runtime(now, board_state)
                with self._lock:
                    self._latest_snapshot = snapshot
            except Exception as exc:
                self.controller.zero_all()
                with self._lock:
                    self._finish_recording("runtime_error")
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

    def _render_loop(self) -> None:
        interval = 1.0 / self.config.render_hz
        while self._running:
            loop_started = time.perf_counter()
            try:
                render_now = time.monotonic()
                with self._lock:
                    frame = (
                        None
                        if self._latest_frame is None
                        else self._latest_frame.copy()
                    )
                    board_state = self._latest_board_state
                    snapshot = dict(self._latest_snapshot)
                if frame is not None and board_state is not None and snapshot:
                    jpeg = self._render_dashboard_frame(frame, board_state)
                    with self._lock:
                        self._latest_jpeg = jpeg
                if self._last_render_loop_monotonic is not None:
                    self._last_render_loop_dt_s = (
                        render_now - self._last_render_loop_monotonic
                    )
                self._last_render_loop_monotonic = render_now
            except Exception:
                pass
            elapsed = time.perf_counter() - loop_started
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _step_runtime(self, now: float, board_state) -> dict[str, Any]:
        modes = self.config.modes
        phases = self.config.phases
        self._frame_timestamp = board_state.timestamp
        ready, reason = self._ready_state(now, board_state)
        if ready:
            self._last_ready_time = now
            if self._mode != modes.running:
                self._phase = phases.ready
        elif self._mode == modes.running:
            self._finish_recording("tracking_not_ready")
            self._mode = modes.armed
            self._phase = phases.ready
            self._match_step_count = 0
            self.policy_runner.reset()
            self._status = reason
            self._push_event("tracking_not_ready")

        target_chaser = RobotCommand(0.0, 0.0)
        target_evader = RobotCommand(0.0, 0.0)
        final_chaser = RobotCommand(0.0, 0.0)
        final_evader = RobotCommand(0.0, 0.0)
        raw_chaser = RobotCommand(0.0, 0.0)
        raw_evader = RobotCommand(0.0, 0.0)
        episode_progress = self._episode_progress()
        observations = None

        if self._mode == modes.running and ready:
            if self._match_start_monotonic is None:
                self._match_start_monotonic = now
                self._match_step_count = 0
                self._recording_frame_index = 0
                self.recorder.start_game(board_state)
            episode_progress = self._episode_progress()
            observation_started = time.perf_counter()
            observations = self.observation_builder.build(board_state, episode_progress)
            self._last_observation_build_ms = (
                time.perf_counter() - observation_started
            ) * 1000.0
            if observations is None:
                self._finish_recording("observation_build_failed")
                self._mode = modes.armed
                self._phase = phases.ready
                self.policy_runner.reset()
                self._status = "Observation build failed"
                self._push_event("observation_build_failed")
            else:
                inference_started = time.perf_counter()
                outputs = self.policy_runner.step(
                    observations.chaser, observations.evader
                )
                self._last_inference_ms = (
                    time.perf_counter() - inference_started
                ) * 1000.0
                raw_chaser = RobotCommand(
                    left=float(outputs.chaser_action[0]),
                    right=float(outputs.chaser_action[1]),
                )
                raw_evader = RobotCommand(
                    left=float(outputs.evader_action[0]),
                    right=float(outputs.evader_action[1]),
                )
                target_chaser, target_evader, terminal_reason = self._apply_game_rules(
                    now, board_state, raw_chaser, raw_evader
                )
                if terminal_reason is not None:
                    self._finish_recording(terminal_reason.lower().replace(" ", "_"))
                    self._mode = modes.armed
                    self._match_step_count = 0
                    self._status = terminal_reason
                    self.policy_runner.reset()
                else:
                    self.recorder.record_step(
                        self._recording_frame_index,
                        now,
                        board_state,
                        target_chaser,
                        target_evader,
                    )
                    self._recording_frame_index += 1
                    self._match_step_count = min(
                        self._match_step_count + 1, self.max_steps
                    )

        if self._mode == modes.estop:
            target_chaser = RobotCommand(0.0, 0.0)
            target_evader = RobotCommand(0.0, 0.0)
            self._phase = phases.stopped

        final_chaser = target_chaser
        final_evader = target_evader

        send_started = time.perf_counter()
        self.controller.send(final_chaser, final_evader)
        self._last_send_ms = (time.perf_counter() - send_started) * 1000.0
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

    def _episode_progress(self) -> float:
        if self.max_steps <= 0:
            return 0.0
        return max(0.0, min(1.0, self._match_step_count / self.max_steps))

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
        if self._match_step_count < self.freeze_steps:
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
        if self._match_step_count >= self.max_steps:
            self._phase = self.config.phases.time_up
            self._push_event("time_up")
            return RobotCommand(0.0, 0.0), RobotCommand(0.0, 0.0), "Time up"
        self._phase = self.config.phases.active
        self._status = "Game active"
        return self._scale_command(chaser), self._scale_command(evader), None

    def _scale_command(self, command: RobotCommand) -> RobotCommand:
        scale = self.config.action_output_scale
        return RobotCommand(left=command.left * scale, right=command.right * scale)

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
                "control_target_hz": self.control_hz,
                "control_dt_s": self._last_control_loop_dt_s,
                "control_actual_hz": None
                if not self._last_control_loop_dt_s
                else 1.0 / self._last_control_loop_dt_s,
                "render_target_hz": self.config.render_hz,
                "render_dt_s": self._last_render_loop_dt_s,
                "render_actual_hz": None
                if not self._last_render_loop_dt_s
                else 1.0 / self._last_render_loop_dt_s,
                "observation_build_ms": self._last_observation_build_ms,
                "inference_ms": self._last_inference_ms,
                "send_ms": self._last_send_ms,
                "action_output_scale": self.config.action_output_scale,
                "frame_age_s": frame_age,
                "recording_split": self.recorder.recording_split,
                "recording_active": self.recorder.active,
            },
            "game": {
                "elapsed_s": 0.0
                if self._match_start_monotonic is None
                else now - self._match_start_monotonic,
                "duration_s": self.match_duration_s,
                "step_count": self._match_step_count,
                "max_steps": self.max_steps,
                "episode_progress": self._episode_progress(),
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

    def _render_dashboard_frame(self, frame: np.ndarray, board_state) -> bytes | None:
        raw_view, arena_view = self.tracker.render_debug_views(frame, board_state)
        raw_view = self._scale_frame(raw_view)
        arena_view = self._scale_frame(arena_view)
        target_width = min(raw_view.shape[1], arena_view.shape[1])
        raw_small = cv2.resize(
            raw_view,
            (target_width, int(raw_view.shape[0] * target_width / raw_view.shape[1])),
        )
        arena_small = cv2.resize(
            arena_view,
            (
                target_width,
                int(arena_view.shape[0] * target_width / arena_view.shape[1]),
            ),
        )
        combined = cv2.vconcat([raw_small, arena_small])
        ok, encoded = cv2.imencode(
            ".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        )
        if not ok:
            return None
        return encoded.tobytes()

    def _scale_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] <= self.config.dashboard_frame_width:
            return frame
        scale = self.config.dashboard_frame_width / frame.shape[1]
        return cv2.resize(
            frame,
            (self.config.dashboard_frame_width, int(round(frame.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _push_event(self, message: str) -> None:
        self._events.appendleft(RuntimeEvent(timestamp=time.time(), message=message))

    def _finish_recording(self, reason: str) -> None:
        for run_dir in self.recorder.finish_game(reason):
            self._push_event(f"recorded_run:{run_dir}")
