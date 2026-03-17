from __future__ import annotations

import threading
import time

import numpy as np

from online.config import DemoConfig
from online.observation import ObservationBuilder
from online.policy_runner import PolicyRunner
from online.robot_io import RobotClient
from online.runtime_state import RuntimeState


def _slew_limit(
    previous: np.ndarray, proposed: np.ndarray, max_step: float
) -> np.ndarray:
    delta = np.clip(proposed - previous, -max_step, max_step)
    return previous + delta


class ControlWorker(threading.Thread):
    def __init__(
        self,
        config: DemoConfig,
        state: RuntimeState,
        policy: PolicyRunner,
        observation_builder: ObservationBuilder,
        chaser_robot: RobotClient,
        evader_robot: RobotClient,
    ) -> None:
        super().__init__(name="control-worker", daemon=True)
        self.config = config
        self.state = state
        self.policy = policy
        self.observation_builder = observation_builder
        self.chaser_robot = chaser_robot
        self.evader_robot = evader_robot
        self._episode_start = time.time()
        self._episode_step = 0
        self._reset_count = 0
        self._last_chaser = np.zeros(2, dtype=np.float32)
        self._last_evader = np.zeros(2, dtype=np.float32)
        self._last_loop = 0.0

    def _safe_stop(self, reason: str) -> None:
        chaser_state = self.chaser_robot.stop()
        evader_state = self.evader_robot.stop()

        def update(snapshot) -> None:  # type: ignore[no-untyped-def]
            snapshot.policy.enabled = False
            snapshot.policy.last_reason = reason
            snapshot.chaser_command = chaser_state
            snapshot.evader_command = evader_state

        self.state.mutate_snapshot(update)

    def reset_episode(self, reason: str) -> None:
        self.policy.reset()
        self._episode_start = time.time()
        self._episode_step = 0
        self._reset_count += 1
        self._last_chaser = np.zeros(2, dtype=np.float32)
        self._last_evader = np.zeros(2, dtype=np.float32)

        def update(snapshot) -> None:  # type: ignore[no-untyped-def]
            snapshot.policy.reset_count = self._reset_count
            snapshot.policy.last_reason = reason

        self.state.mutate_snapshot(update)

    def run(self) -> None:
        dt = 1.0 / self.config.control.hz
        self.reset_episode("startup")
        while not self.state.stop_event().is_set():
            loop_start = time.time()
            snap = self.state.snapshot()
            operator = snap.operator
            try:
                if operator.emergency_stop or not operator.control_enabled:
                    self._safe_stop("operator-disabled")
                elif not snap.world.ready:
                    self._safe_stop("world-not-ready")
                    self.reset_episode("world-not-ready")
                elif snap.stats.world_age_s > self.config.control.world_state_timeout_s:
                    self._safe_stop("world-stale")
                    self.reset_episode("world-stale")
                else:
                    chaser_obs, evader_obs, progress = self.observation_builder.build(
                        snap.world,
                        self._episode_step,
                    )
                    output = self.policy.step(chaser_obs, evader_obs)
                    chaser_action = output.chaser_action.copy()
                    evader_action = output.evader_action.copy()
                    if self.config.control.enable_chaser_freeze:
                        freeze_s = self.policy.env_config.chaser_freeze_seconds
                        if time.time() - self._episode_start < freeze_s:
                            chaser_action[:] = 0.0
                    chaser_action = _slew_limit(
                        self._last_chaser,
                        chaser_action,
                        self.config.control.max_command_step,
                    )
                    evader_action = _slew_limit(
                        self._last_evader,
                        evader_action,
                        self.config.control.max_command_step,
                    )
                    self._last_chaser = chaser_action
                    self._last_evader = evader_action
                    chaser_state = self.chaser_robot.send(
                        float(chaser_action[0]), float(chaser_action[1])
                    )
                    evader_state = self.evader_robot.send(
                        float(evader_action[0]), float(evader_action[1])
                    )
                    self._episode_step += 1

                    def update(snapshot) -> None:  # type: ignore[no-untyped-def]
                        snapshot.policy.timestamp = time.time()
                        snapshot.policy.enabled = True
                        snapshot.policy.episode_progress = progress
                        snapshot.policy.chaser_action = chaser_action.tolist()
                        snapshot.policy.evader_action = evader_action.tolist()
                        snapshot.policy.observation_size = output.observation_size
                        snapshot.policy.last_reason = "running"
                        snapshot.chaser_command = chaser_state
                        snapshot.evader_command = evader_state

                    self.state.mutate_snapshot(update)
            except Exception as exc:
                self._safe_stop(f"control-error: {exc}")
                self.reset_episode("control-exception")

                def update_error(snapshot) -> None:  # type: ignore[no-untyped-def]
                    snapshot.stats.last_error = str(exc)

                self.state.mutate_snapshot(update_error)

            elapsed = time.time() - loop_start
            hz = (
                0.0
                if self._last_loop == 0.0
                else 1.0 / max(1e-6, loop_start - self._last_loop)
            )
            self._last_loop = loop_start

            def update_stats(snapshot) -> None:  # type: ignore[no-untyped-def]
                snapshot.stats.control_ms = elapsed * 1000.0
                snapshot.stats.control_hz = hz

            self.state.mutate_snapshot(update_stats)
            time.sleep(max(0.0, dt - (time.time() - loop_start)))
