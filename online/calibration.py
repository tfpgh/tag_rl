from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from online.config import ArenaCalibrationConfig
from online.types import CalibrationState, TagDetection


@dataclass(slots=True)
class CalibrationResult:
    state: CalibrationState
    homography: np.ndarray | None
    display_homography: np.ndarray | None
    display_size: tuple[int, int]
    game_border_points: np.ndarray


class ArenaCalibrator:
    def __init__(
        self, config: ArenaCalibrationConfig, arena_width: float, arena_height: float
    ) -> None:
        self.config = config
        self._arena_width = arena_width
        self._arena_height = arena_height
        self._homography: np.ndarray | None = None
        self._display_homography: np.ndarray | None = None
        self._stable_count = 0
        self._last_seen = 0.0

    def _world_corners(self) -> np.ndarray:
        half_w = self._arena_width / 2
        half_h = self._arena_height / 2
        inset = self.config.tag_size_m / 2
        return np.array(
            [
                [-half_w - inset, -half_h - inset],
                [half_w + inset, -half_h - inset],
                [half_w + inset, half_h + inset],
                [-half_w - inset, half_h + inset],
            ],
            dtype=np.float32,
        )

    def _display_geometry(self) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
        frame_w = 1920
        stats_h = 35
        buffer_px = int(0.01 * frame_w)
        mat_w_px = frame_w - 2 * buffer_px
        px_per_meter = mat_w_px / self.config.mat_width_m
        mat_h_px = int(round(self.config.mat_height_m * px_per_meter))
        frame_h = stats_h + buffer_px * 2 + mat_h_px
        x0 = buffer_px
        y0 = stats_h + buffer_px
        x1 = buffer_px + mat_w_px
        y1 = stats_h + buffer_px + mat_h_px
        tag_center_inset_px = int(round((self.config.tag_size_m / 2) * px_per_meter))
        game_border_inset_px = int(round(self.config.tag_size_m * px_per_meter))
        tag_center_points = np.array(
            [
                [x0 + tag_center_inset_px, y0 + tag_center_inset_px],
                [x1 - tag_center_inset_px, y0 + tag_center_inset_px],
                [x1 - tag_center_inset_px, y1 - tag_center_inset_px],
                [x0 + tag_center_inset_px, y1 - tag_center_inset_px],
            ],
            dtype=np.float32,
        )
        game_border_points = np.array(
            [
                [x0 + game_border_inset_px, y0 + game_border_inset_px],
                [x1 - game_border_inset_px, y0 + game_border_inset_px],
                [x1 - game_border_inset_px, y1 - game_border_inset_px],
                [x0 + game_border_inset_px, y1 - game_border_inset_px],
            ],
            dtype=np.float32,
        )
        return tag_center_points, (frame_w, frame_h), game_border_points

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
                    detections_by_id[tag_id].center_px
                    for tag_id in self.config.corner_tag_ids
                ],
                dtype=np.float32,
            )
            homography = cv2.getPerspectiveTransform(src_points, self._world_corners())
            display_points, display_size, game_border_points = self._display_geometry()
            display_homography = cv2.getPerspectiveTransform(src_points, display_points)
            self._homography = homography
            self._display_homography = display_homography
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

        _, display_size, game_border_points = self._display_geometry()

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
        return CalibrationResult(
            state=state,
            homography=self._homography,
            display_homography=self._display_homography,
            display_size=display_size,
            game_border_points=game_border_points,
        )

    def image_to_world(
        self, homography: np.ndarray, image_points: np.ndarray
    ) -> np.ndarray:
        transformed = cv2.perspectiveTransform(image_points[None, :, :], homography)
        return transformed[0]

    def warp_to_board(
        self,
        frame: np.ndarray,
        display_homography: np.ndarray,
        display_size: tuple[int, int],
    ) -> np.ndarray:
        return cv2.warpPerspective(frame, display_homography, display_size)
