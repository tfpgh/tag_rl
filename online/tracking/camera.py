from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from online.core.config import CameraConfig


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CameraInfo:
    backend: str
    width: int
    height: int
    fps: float
    fourcc: str
    mode: float
    codec_pixel_format: str


def _decode_fourcc(value: float) -> str:
    code = int(round(value))
    chars = [chr((code >> shift) & 0xFF) for shift in (0, 8, 16, 24)]
    text = "".join(chars).strip("\x00")
    return text or f"0x{code:08x}"


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

    def info(self) -> CameraInfo:
        backend = "unknown"
        if hasattr(self._capture, "getBackendName"):
            try:
                backend = str(self._capture.getBackendName())
            except cv2.error:
                backend = "unknown"

        codec_pixel_format = "unknown"
        if hasattr(cv2, "CAP_PROP_CODEC_PIXEL_FORMAT"):
            raw_format = self._capture.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT)
            codec_pixel_format = _decode_fourcc(raw_format)

        return CameraInfo(
            backend=backend,
            width=int(round(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            height=int(round(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            fps=float(self._capture.get(cv2.CAP_PROP_FPS)),
            fourcc=_decode_fourcc(self._capture.get(cv2.CAP_PROP_FOURCC)),
            mode=float(self._capture.get(cv2.CAP_PROP_MODE)),
            codec_pixel_format=codec_pixel_format,
        )

    def release(self) -> None:
        self._capture.release()
