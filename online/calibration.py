from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from online.config import ArenaCalibrationConfig
from online.types import CalibrationState, TagDetection

CORNER_TAG_TO_WORLD_INDEX = {0: 0, 1: 1, 2: 2, 3: 3}


@dataclass(slots=True)
class CalibrationResult:
    state: CalibrationState
    homography: np.ndarray | None


class ArenaCalibrator:
    def __init__(
        self, config: ArenaCalibrationConfig, arena_width: float, arena_height: float
    ) -> None:
        self.config = config
        self._arena_width = arena_width
        self._arena_height = arena_height
        self._homography: np.ndarray | None = None
        self._stable_count = 0
        self._last_seen = 0.0

    def _world_corners(self) -> np.ndarray:
        half_w = self._arena_width / 2
        half_h = self._arena_height / 2
        return np.array(
            [
                [-half_w, -half_h],
                [half_w, -half_h],
                [half_w, half_h],
                [-half_w, half_h],
            ],
            dtype=np.float32,
        )

    def _choose_inward_corner(
        self, detection: TagDetection, detections_by_id: dict[int, TagDetection]
    ) -> np.ndarray:
        corners = np.asarray(detection.corners_px, dtype=np.float32)
        if detection.tag_id not in CORNER_TAG_TO_WORLD_INDEX:
            return corners.mean(axis=0)
        target_id = (
            2
            if detection.tag_id == 0
            else 3
            if detection.tag_id == 1
            else 0
            if detection.tag_id == 2
            else 1
        )
        target = np.asarray(
            detections_by_id[target_id].center_px,
            dtype=np.float32,
        )
        distances = np.linalg.norm(corners - target[None, :], axis=1)
        return corners[int(np.argmin(distances))]

    def update(self, detections: list[TagDetection]) -> CalibrationResult:
        now = time.time()
        detections_by_id = {det.tag_id: det for det in detections}
        have_all = all(
            tag_id in detections_by_id for tag_id in self.config.corner_tag_ids
        )
        status = "uncalibrated"
        if have_all:
            src_points = np.array(
                [
                    self._choose_inward_corner(
                        detections_by_id[tag_id], detections_by_id
                    )
                    for tag_id in self.config.corner_tag_ids
                ],
                dtype=np.float32,
            )
            homography = cv2.getPerspectiveTransform(src_points, self._world_corners())
            self._homography = homography
            self._last_seen = now
            self._stable_count += 1
            status = (
                "ready"
                if self._stable_count >= self.config.stable_frames_required
                else "partial"
            )
        else:
            if (
                self._homography is not None
                and now - self._last_seen <= self.config.drop_tolerance_s
            ):
                status = "degraded"
            else:
                self._stable_count = 0
                self._homography = None

        state = CalibrationState(
            status=status,
            stable_count=self._stable_count,
            last_update_s=self._last_seen,
            source_tag_ids=list(
                self.config.corner_tag_ids + self.config.auxiliary_tag_ids
            ),
            homography=None if self._homography is None else self._homography.tolist(),
            arena_corners_world=[tuple(pt) for pt in self._world_corners().tolist()],
        )
        return CalibrationResult(state=state, homography=self._homography)

    def image_to_world(
        self, homography: np.ndarray, image_points: np.ndarray
    ) -> np.ndarray:
        transformed = cv2.perspectiveTransform(image_points[None, :, :], homography)
        return transformed[0]
