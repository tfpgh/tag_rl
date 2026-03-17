from __future__ import annotations

import threading
import time

import cv2

from online.config import CameraConfig
from online.runtime_state import RuntimeState


class CameraWorker(threading.Thread):
    def __init__(self, config: CameraConfig, state: RuntimeState) -> None:
        super().__init__(name="camera-worker", daemon=True)
        self.config = config
        self.state = state
        self._capture = cv2.VideoCapture(config.device_index, config.api_preference)
        self._capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*config.mjpg_fourcc),
        )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self._capture.set(cv2.CAP_PROP_FPS, config.fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, config.buffersize)
        self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, config.auto_exposure)
        self._capture.set(cv2.CAP_PROP_EXPOSURE, config.exposure)
        self._frame_id = 0
        self._last_ts = 0.0

    def run(self) -> None:
        try:
            while not self.state.stop_event().is_set():
                ok, frame = self._capture.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                now = time.time()
                self._frame_id += 1
                fps = (
                    0.0
                    if self._last_ts == 0.0
                    else 1.0 / max(1e-6, now - self._last_ts)
                )
                self._last_ts = now
                self.state.set_raw_frame(frame, timestamp=now, frame_id=self._frame_id)

                def update(snapshot) -> None:  # type: ignore[no-untyped-def]
                    snapshot.stats.capture_fps = fps

                self.state.mutate_snapshot(update)
        finally:
            self._capture.release()
