from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from online.state import (
    ArenaCalibration,
    ArenaViewConfig,
    CORNER_TAG_IDS,
    MAT_H_MM,
    MAT_W_MM,
    TAG_CENTER_H_MM,
    TAG_CENTER_W_MM,
    TagDetection,
    TrackerConfig,
)


WORLD_TAG_CENTERS_MM = np.array(
    [
        [-TAG_CENTER_W_MM / 2.0, TAG_CENTER_H_MM / 2.0],
        [TAG_CENTER_W_MM / 2.0, TAG_CENTER_H_MM / 2.0],
        [TAG_CENTER_W_MM / 2.0, -TAG_CENTER_H_MM / 2.0],
        [-TAG_CENTER_W_MM / 2.0, -TAG_CENTER_H_MM / 2.0],
    ],
    dtype=np.float32,
)


@dataclass(slots=True)
class ArenaViewLayout:
    frame_width: int
    frame_height: int
    stats_height: int
    buffer_px: int
    board_width_px: int
    board_height_px: int
    pixels_per_mm: float
    destination_tag_centers_px: np.ndarray


def build_arena_view_layout(config: ArenaViewConfig) -> ArenaViewLayout:
    buffer_px = int(config.buffer_fraction * config.frame_width)
    board_width_px = config.frame_width - 2 * buffer_px
    pixels_per_mm = board_width_px / MAT_W_MM
    board_height_px = int(round(MAT_H_MM * pixels_per_mm))
    frame_height = config.stats_height + buffer_px + board_height_px + buffer_px
    inset_px = (MAT_W_MM - TAG_CENTER_W_MM) * 0.5 * pixels_per_mm
    inset_y_px = (MAT_H_MM - TAG_CENTER_H_MM) * 0.5 * pixels_per_mm
    x0 = buffer_px + inset_px
    x1 = buffer_px + board_width_px - inset_px
    y0 = config.stats_height + buffer_px + inset_y_px
    y1 = config.stats_height + buffer_px + board_height_px - inset_y_px
    destination_tag_centers_px = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        dtype=np.float32,
    )
    return ArenaViewLayout(
        frame_width=config.frame_width,
        frame_height=frame_height,
        stats_height=config.stats_height,
        buffer_px=buffer_px,
        board_width_px=board_width_px,
        board_height_px=board_height_px,
        pixels_per_mm=pixels_per_mm,
        destination_tag_centers_px=destination_tag_centers_px,
    )


class CornerTagCalibrator:
    def __init__(self, config: TrackerConfig, layout: ArenaViewLayout) -> None:
        self._required_samples = config.calibration_frames
        self._layout = layout
        self._samples: dict[int, list[np.ndarray]] = {
            tag_id: [] for tag_id in CORNER_TAG_IDS
        }
        self._image_to_warp: np.ndarray | None = None
        self._image_to_world: np.ndarray | None = None

    def update(self, detections: list[TagDetection]) -> ArenaCalibration:
        if self._image_to_warp is None or self._image_to_world is None:
            for detection in detections:
                if detection.tag_id not in self._samples:
                    continue
                samples = self._samples[detection.tag_id]
                if len(samples) < self._required_samples:
                    samples.append(detection.center_px)
            if self.samples_collected >= self._required_samples:
                src = np.array(
                    [
                        np.mean(self._samples[tag_id], axis=0)
                        for tag_id in CORNER_TAG_IDS
                    ],
                    dtype=np.float32,
                )
                self._image_to_warp = cv2.getPerspectiveTransform(
                    src, self._layout.destination_tag_centers_px
                )
                self._image_to_world = cv2.getPerspectiveTransform(
                    src, WORLD_TAG_CENTERS_MM
                )
        return self.snapshot()

    @property
    def samples_collected(self) -> int:
        return min(len(samples) for samples in self._samples.values())

    def snapshot(self) -> ArenaCalibration:
        return ArenaCalibration(
            valid=self._image_to_warp is not None and self._image_to_world is not None,
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
    reshaped = points.reshape(-1, 1, 2).astype(np.float32)
    transformed = cv2.perspectiveTransform(reshaped, matrix)
    return transformed.reshape(-1, 2)
