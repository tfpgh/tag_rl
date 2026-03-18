from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import cv2
import numpy as np
from pupil_apriltags import Detector

from online.config import TrackerConfig
from online.state import TagDetection


class AprilTagTracker:
    def __init__(self, config: TrackerConfig) -> None:
        detector = config.detector
        self._detector = Detector(
            families=detector.family,
            nthreads=detector.threads,
            quad_decimate=detector.quad_decimate,
        )

    def detect(self, frame: np.ndarray) -> list[TagDetection]:
        gray = cast(np.ndarray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        raw_detections = cast(Sequence[Any], self._detector.detect(gray))
        detections: list[TagDetection] = []
        for detection in raw_detections:
            detections.append(
                TagDetection(
                    tag_id=int(detection.tag_id),
                    center_px=np.asarray(detection.center, dtype=np.float32),
                    corners_px=np.asarray(detection.corners, dtype=np.float32),
                    decision_margin=float(getattr(detection, "decision_margin", 0.0)),
                    hamming=int(getattr(detection, "hamming", 0)),
                )
            )
        return detections
