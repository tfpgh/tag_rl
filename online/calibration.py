from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from online.config import ArenaConfig, TrackerConfig, ViewConfig
from online.state import ArenaCalibration, TagDetection


@dataclass(frozen=True, slots=True)
class ArenaViewLayout:
    frame_width: int
    frame_height: int
    stats_height: int
    buffer_px: int
    board_width_px: int
    board_height_px: int
    pixels_per_mm: float
    destination_tag_centers_px: np.ndarray
    board_width_mm: float
    board_height_mm: float


def corner_tag_centers_mm(arena: ArenaConfig) -> np.ndarray:
    half_w = arena.tag_center_width_mm / 2.0
    half_h = arena.tag_center_height_mm / 2.0
    return np.array(
        [[-half_w, half_h], [half_w, half_h], [half_w, -half_h], [-half_w, -half_h]],
        dtype=np.float32,
    )


def build_arena_view_layout(arena: ArenaConfig, view: ViewConfig) -> ArenaViewLayout:
    buffer_px = int(view.buffer_fraction * view.frame_width)
    board_width_px = view.frame_width - 2 * buffer_px
    pixels_per_mm = board_width_px / arena.board_width_mm
    board_height_px = int(round(arena.board_height_mm * pixels_per_mm))
    frame_height = view.stats_height + buffer_px + board_height_px + buffer_px

    board_left_px = buffer_px
    board_top_px = view.stats_height + buffer_px
    board_right_px = board_left_px + board_width_px
    board_bottom_px = board_top_px + board_height_px
    half_tag_px = 0.5 * arena.tag_size_mm * pixels_per_mm

    return ArenaViewLayout(
        frame_width=view.frame_width,
        frame_height=frame_height,
        stats_height=view.stats_height,
        buffer_px=buffer_px,
        board_width_px=board_width_px,
        board_height_px=board_height_px,
        pixels_per_mm=pixels_per_mm,
        destination_tag_centers_px=np.array(
            [
                [board_left_px - half_tag_px, board_top_px - half_tag_px],
                [board_right_px + half_tag_px, board_top_px - half_tag_px],
                [board_right_px + half_tag_px, board_bottom_px + half_tag_px],
                [board_left_px - half_tag_px, board_bottom_px + half_tag_px],
            ],
            dtype=np.float32,
        ),
        board_width_mm=arena.board_width_mm,
        board_height_mm=arena.board_height_mm,
    )


class CornerTagCalibrator:
    def __init__(self, config: TrackerConfig, layout: ArenaViewLayout) -> None:
        self._arena = config.arena
        self._required_samples = config.calibration_frames
        self._layout = layout
        self._samples = {tag_id: [] for tag_id in self._arena.corner_tag_ids}
        self._world_tag_centers = corner_tag_centers_mm(self._arena)
        self._image_to_warp: np.ndarray | None = None
        self._image_to_world: np.ndarray | None = None

    def update(self, detections: list[TagDetection]) -> ArenaCalibration:
        if not self.is_calibrated:
            self._collect_samples(detections)
            if self.samples_collected >= self._required_samples:
                self._solve()
        return self.snapshot()

    @property
    def is_calibrated(self) -> bool:
        return self._image_to_warp is not None and self._image_to_world is not None

    @property
    def samples_collected(self) -> int:
        return min(len(samples) for samples in self._samples.values())

    def _collect_samples(self, detections: list[TagDetection]) -> None:
        for detection in detections:
            if detection.tag_id not in self._samples:
                continue
            samples = self._samples[detection.tag_id]
            if len(samples) < self._required_samples:
                samples.append(detection.center_px)

    def _solve(self) -> None:
        image_points = np.array(
            [
                np.mean(self._samples[tag_id], axis=0)
                for tag_id in self._arena.corner_tag_ids
            ],
            dtype=np.float32,
        )
        self._image_to_warp = cv2.getPerspectiveTransform(
            image_points, self._layout.destination_tag_centers_px
        )
        self._image_to_world = cv2.getPerspectiveTransform(
            image_points, self._world_tag_centers
        )

    def snapshot(self) -> ArenaCalibration:
        return ArenaCalibration(
            valid=self.is_calibrated,
            samples_collected=self.samples_collected,
            required_samples=self._required_samples,
            image_to_warp=None
            if self._image_to_warp is None
            else self._image_to_warp.copy(),
            image_to_world=None
            if self._image_to_world is None
            else self._image_to_world.copy(),
            warp_size_px=(self._layout.frame_width, self._layout.frame_height),
            pixels_per_mm=self._layout.pixels_per_mm,
        )


def transform_points(
    matrix: np.ndarray | None, points: np.ndarray
) -> np.ndarray | None:
    if matrix is None or points.size == 0:
        return None
    return cv2.perspectiveTransform(
        points.reshape(-1, 1, 2).astype(np.float32), matrix
    ).reshape(-1, 2)
