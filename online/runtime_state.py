from __future__ import annotations

import copy
import threading
import time

import numpy as np

from online.types import Snapshot


class RuntimeState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot = Snapshot()
        self._raw_frame: np.ndarray | None = None
        self._annotated_frame: np.ndarray | None = None
        self._annotated_jpeg: bytes | None = None
        self._stop_event = threading.Event()

    def stop_event(self) -> threading.Event:
        return self._stop_event

    def request_stop(self) -> None:
        self._stop_event.set()

    def _refresh_errors(self) -> None:
        self._snapshot.stats.last_error = " | ".join(
            error
            for error in (
                self._snapshot.stats.camera_error,
                self._snapshot.stats.detection_error,
                self._snapshot.stats.control_error,
            )
            if error
        )

    def set_raw_frame(
        self, frame: np.ndarray, *, timestamp: float, frame_id: int
    ) -> None:
        with self._lock:
            self._raw_frame = frame.copy()
            self._snapshot.frame.frame_id = frame_id
            self._snapshot.frame.timestamp = timestamp
            self._snapshot.frame.width = int(frame.shape[1])
            self._snapshot.frame.height = int(frame.shape[0])

    def get_raw_frame(self) -> tuple[np.ndarray | None, int, float]:
        with self._lock:
            frame = None if self._raw_frame is None else self._raw_frame.copy()
            return frame, self._snapshot.frame.frame_id, self._snapshot.frame.timestamp

    def set_annotated_frame(self, frame: np.ndarray, jpeg: bytes | None = None) -> None:
        with self._lock:
            self._annotated_frame = frame.copy()
            self._annotated_jpeg = jpeg

    def get_annotated_jpeg(self) -> bytes | None:
        with self._lock:
            return self._annotated_jpeg

    def mutate_snapshot(self, mutator) -> None:  # type: ignore[no-untyped-def]
        with self._lock:
            mutator(self._snapshot)
            self._refresh_errors()
            now = time.time()
            self._snapshot.frame.age_s = (
                0.0
                if self._snapshot.frame.timestamp <= 0.0
                else max(0.0, now - self._snapshot.frame.timestamp)
            )
            self._snapshot.stats.frame_age_s = self._snapshot.frame.age_s
            self._snapshot.stats.world_age_s = (
                0.0
                if self._snapshot.world.timestamp <= 0.0
                else max(0.0, now - self._snapshot.world.timestamp)
            )

    def snapshot(self) -> Snapshot:
        with self._lock:
            snap = copy.deepcopy(self._snapshot)
        now = time.time()
        snap.frame.age_s = (
            0.0 if snap.frame.timestamp <= 0.0 else max(0.0, now - snap.frame.timestamp)
        )
        snap.stats.frame_age_s = snap.frame.age_s
        snap.stats.world_age_s = (
            0.0 if snap.world.timestamp <= 0.0 else max(0.0, now - snap.world.timestamp)
        )
        return snap
