from __future__ import annotations

import time

import cv2
import numpy as np

from online.core.config import CameraConfig


class CameraError(RuntimeError):
    pass


class CameraStream:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        if config.backend is None:
            self._capture = cv2.VideoCapture(config.device_index)
        else:
            self._capture = cv2.VideoCapture(config.device_index, config.backend)
        if not self._capture.isOpened():
            raise CameraError(f"failed to open camera {config.device_index}")
        self._configure()

    def _configure(self) -> None:
        config = self.config
        if config.mjpg:
            self._capture.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG"),  # pyright: ignore[reportAttributeAccessIssue]
            )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
        self._capture.set(cv2.CAP_PROP_FPS, config.fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, config.buffer_size)
        if config.auto_exposure is not None:
            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, config.auto_exposure)
        if config.exposure is not None:
            self._capture.set(cv2.CAP_PROP_EXPOSURE, config.exposure)

    def read(self) -> tuple[float, np.ndarray]:
        ok, frame = self._capture.read()
        if not ok:
            raise CameraError("failed to read camera frame")
        return time.time(), frame

    def release(self) -> None:
        self._capture.release()
