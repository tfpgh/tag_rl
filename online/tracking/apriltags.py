from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import cv2
import numpy as np
from pupil_apriltags import Detector

from online.core.config import TrackerConfig
from online.core.state import TagDetection


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
        return self.detect_gray(gray)

    def detect_gray(self, gray: np.ndarray) -> list[TagDetection]:
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

    def detect_tag_in_roi(
        self,
        gray: np.ndarray,
        roi: tuple[int, int, int, int],
        tag_id: int,
    ) -> TagDetection | None:
        x0, y0, x1, y1 = roi
        cropped = gray[y0:y1, x0:x1]
        detections = self.detect_gray(cropped)
        matches: list[TagDetection] = []
        for detection in detections:
            if detection.tag_id != tag_id:
                continue
            matches.append(
                TagDetection(
                    tag_id=detection.tag_id,
                    center_px=detection.center_px
                    + np.array([x0, y0], dtype=np.float32),
                    corners_px=detection.corners_px
                    + np.array([x0, y0], dtype=np.float32),
                    decision_margin=detection.decision_margin,
                    hamming=detection.hamming,
                )
            )
        if not matches:
            return None
        return max(matches, key=lambda detection: detection.decision_margin)
