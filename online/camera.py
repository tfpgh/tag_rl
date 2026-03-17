from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np

from online.config import CameraConfig
from online.runtime_state import RuntimeState


class CameraWorker(threading.Thread):
    def __init__(self, config: CameraConfig, state: RuntimeState) -> None:
        super().__init__(name="camera-worker", daemon=True)
        self.config = config
        self.state = state
        self._capture = self._open_capture()
        self._frame_id = 0
        self._last_ts = 0.0

    def _api_candidates(self) -> list[int | None]:
        candidates: list[int | None] = []
        if self.config.api_preference is not None:
            candidates.append(self.config.api_preference)
        if sys.platform.startswith("linux"):
            v4l2 = getattr(cv2, "CAP_V4L2", None)
            if v4l2 is not None and v4l2 not in candidates:
                candidates.append(v4l2)
        if sys.platform == "darwin":
            avfoundation = getattr(cv2, "CAP_AVFOUNDATION", None)
            if avfoundation is not None and avfoundation not in candidates:
                candidates.append(avfoundation)
        candidates.append(None)
        return candidates

    def _configure_capture(self, capture: cv2.VideoCapture) -> None:
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*self.config.mjpg_fourcc),
        )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffersize)
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.config.auto_exposure)
        capture.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)

    def _open_capture(self) -> cv2.VideoCapture:
        for api_preference in self._api_candidates():
            capture = (
                cv2.VideoCapture(self.config.device_index)
                if api_preference is None
                else cv2.VideoCapture(self.config.device_index, api_preference)
            )
            if capture.isOpened():
                self._configure_capture(capture)
                self.state.mutate_snapshot(
                    lambda snapshot: setattr(
                        snapshot.stats,
                        "camera_error",
                        f"camera backend {api_preference if api_preference is not None else 'default'}",
                    )
                )
                return capture
            capture.release()
        raise RuntimeError(f"Unable to open camera device {self.config.device_index}")

    def run(self) -> None:
        try:
            while not self.state.stop_event().is_set():
                ok, frame = self._capture.read()
                if not ok:
                    self.state.mutate_snapshot(
                        lambda snapshot: setattr(
                            snapshot.stats,
                            "last_error",
                            "camera read failed",
                        )
                    )
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
                ok_jpeg, encoded = cv2.imencode(".jpg", frame)
                if ok_jpeg:
                    self.state.set_annotated_frame(frame, jpeg=encoded.tobytes())

                def update(snapshot) -> None:  # type: ignore[no-untyped-def]
                    snapshot.stats.capture_fps = fps
                    snapshot.stats.camera_error = ""

                self.state.mutate_snapshot(update)
        except Exception as exc:
            self.state.mutate_snapshot(
                lambda snapshot: setattr(
                    snapshot.stats, "camera_error", f"camera error: {exc}"
                )
            )
        finally:
            self._capture.release()
