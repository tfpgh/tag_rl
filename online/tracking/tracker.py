from __future__ import annotations

import math
import time

import cv2
import numpy as np

from online.core.config import TrackerConfig
from online.core.state import (
    ArenaCalibration,
    BoardState,
    ObstacleState,
    Pose2D,
    TagDetection,
    TrackingStats,
)
from online.tracking.apriltags import AprilTagTracker
from online.tracking.calibration import (
    CornerTagCalibrator,
    build_arena_view_layout,
    transform_points,
)
from online.tracking.camera import CameraStream

RAW_WINDOW_NAME = "Tracker Raw"
ARENA_WINDOW_NAME = "Tracker Arena"
RAW_OVERLAY_COLOR = (180, 255, 180)
HEADER_BG_COLOR = (15, 15, 18)
HEADER_TEXT_COLOR = (180, 255, 180)
CALIBRATED_COLOR = (80, 220, 255)
CALIBRATING_COLOR = (100, 200, 255)
CHASER_COLOR = (120, 170, 255)
EVADER_COLOR = (80, 220, 180)
OBSTACLE_COLOR = (180, 180, 80)
ARENA_BORDER_COLOR = (50, 60, 255)


class BoardTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.layout = build_arena_view_layout(config.arena, config.view)
        self.camera = CameraStream(config.camera)
        self.detector = AprilTagTracker(config)
        self.calibrator = CornerTagCalibrator(config, self.layout)
        self._previous_frame_time: float | None = None
        self._last_detections: dict[int, TagDetection] = {}
        self._frame_index = 0

    def close(self) -> None:
        self.camera.release()

    def process_next_frame(self) -> tuple[np.ndarray, BoardState]:
        frame_start = time.perf_counter()
        frame_timestamp, frame = self.camera.read()
        capture_ms = (time.perf_counter() - frame_start) * 1000.0
        loop_start = time.perf_counter()
        detector_start = time.perf_counter()
        detections = self._detect_tags(frame)
        detector_ms = (time.perf_counter() - detector_start) * 1000.0
        self._frame_index += 1

        tracking_start = time.perf_counter()
        calibration = self.calibrator.update(detections)
        fps = 0.0
        if (
            self._previous_frame_time is not None
            and frame_timestamp > self._previous_frame_time
        ):
            fps = 1.0 / (frame_timestamp - self._previous_frame_time)
        self._previous_frame_time = frame_timestamp

        detection_map = {detection.tag_id: detection for detection in detections}
        chaser_pose = self._pose_from_detection(
            detection_map.get(self.config.arena.chaser_tag_id),
            calibration,
            frame_timestamp,
        )
        evader_pose = self._pose_from_detection(
            detection_map.get(self.config.arena.evader_tag_id),
            calibration,
            frame_timestamp,
        )
        obstacles = self._obstacles_from_detections(
            detections, calibration, frame_timestamp
        )
        tracking_ms = (time.perf_counter() - tracking_start) * 1000.0

        loop_ms = (time.perf_counter() - loop_start) * 1000.0
        stats = TrackingStats(
            fps=fps,
            frame_width=int(frame.shape[1]),
            frame_height=int(frame.shape[0]),
            capture_ms=capture_ms,
            detector_ms=detector_ms,
            tracking_ms=tracking_ms,
            loop_ms=loop_ms,
            visible_tags=len(detections),
        )
        board_state = BoardState(
            timestamp=frame_timestamp,
            calibration=calibration,
            chaser=chaser_pose,
            evader=evader_pose,
            obstacles=obstacles,
            detections=detections,
            stats=stats,
        )
        return frame, board_state

    def _detect_tags(self, frame: np.ndarray) -> list[TagDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        expected_tag_ids = self._expected_tag_ids()
        if not self.config.use_roi_tracking or not self._last_detections:
            return self._full_frame_detect(gray)

        roi_detections: dict[int, TagDetection] = {}
        for tag_id in expected_tag_ids:
            previous = self._last_detections.get(tag_id)
            if previous is None:
                continue
            roi = self._roi_for_detection(previous, frame.shape)
            detection = self.detector.detect_tag_in_roi(gray, roi, tag_id)
            if detection is not None:
                roi_detections[tag_id] = detection

        missing_robot = any(
            tag_id not in roi_detections
            for tag_id in (
                self.config.arena.chaser_tag_id,
                self.config.arena.evader_tag_id,
            )
        )
        missing_corner = any(
            tag_id not in roi_detections for tag_id in self.config.arena.corner_tag_ids
        )
        if missing_robot or (not self.calibrator.is_calibrated and missing_corner):
            return self._full_frame_detect(gray)

        self._last_detections.update(roi_detections)
        return list(roi_detections.values())

    def _full_frame_detect(self, gray: np.ndarray) -> list[TagDetection]:
        detections = self.detector.detect_gray(gray)
        self._last_detections.update(
            {detection.tag_id: detection for detection in detections}
        )
        return detections

    def _expected_tag_ids(self) -> tuple[int, ...]:
        arena = self.config.arena
        return (
            *arena.corner_tag_ids,
            arena.chaser_tag_id,
            arena.evader_tag_id,
            *arena.obstacle_tag_ids,
        )

    def _roi_for_detection(
        self, detection: TagDetection, frame_shape: tuple[int, int, int]
    ) -> tuple[int, int, int, int]:
        frame_h, frame_w = frame_shape[:2]
        span_x = float(
            np.max(detection.corners_px[:, 0]) - np.min(detection.corners_px[:, 0])
        )
        span_y = float(
            np.max(detection.corners_px[:, 1]) - np.min(detection.corners_px[:, 1])
        )
        base_size = max(span_x, span_y)
        roi_size = int(round(base_size * self.config.roi_padding_scale))
        roi_size = max(self.config.min_roi_size_px, roi_size)
        roi_size = min(self.config.max_roi_size_px, roi_size)
        half = roi_size // 2
        cx = int(round(float(detection.center_px[0])))
        cy = int(round(float(detection.center_px[1])))
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(frame_w, cx + half)
        y1 = min(frame_h, cy + half)
        return x0, y0, x1, y1

    def render_debug_views(
        self, frame: np.ndarray, board_state: BoardState
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = frame.copy()
        self._draw_raw_overlay(raw, board_state)

        arena = np.zeros(
            (self.layout.frame_height, self.layout.frame_width, 3), dtype=np.uint8
        )
        arena[:] = (30, 30, 35)
        if (
            board_state.calibration.valid
            and board_state.calibration.image_to_warp is not None
        ):
            arena = cv2.warpPerspective(
                frame,
                board_state.calibration.image_to_warp,
                board_state.calibration.warp_size_px,
            )
        self._draw_arena_overlay(arena, board_state)
        return raw, arena

    def _pose_from_detection(
        self,
        detection: TagDetection | None,
        calibration: ArenaCalibration,
        timestamp: float,
    ) -> Pose2D | None:
        if detection is None or calibration.image_to_world is None:
            return None
        world_center = transform_points(
            calibration.image_to_world, detection.center_px[None, :]
        )
        world_edge = transform_points(
            calibration.image_to_world, detection.corners_px[[0, 1], :]
        )
        if world_center is None or world_edge is None:
            return None
        center = world_center[0]
        edge_delta = world_edge[1] - world_edge[0]
        yaw = math.atan2(float(edge_delta[1]), float(edge_delta[0])) + math.pi
        pose = Pose2D(
            x_m=float(center[0]),
            y_m=float(center[1]),
            yaw_rad=yaw,
            visible=True,
            last_seen_timestamp=timestamp,
        )
        return pose

    def _obstacles_from_detections(
        self,
        detections: list[TagDetection],
        calibration: ArenaCalibration,
        timestamp: float,
    ) -> list[ObstacleState]:
        obstacles: list[ObstacleState] = []
        obstacle_tag_ids = set(self.config.arena.obstacle_tag_ids)
        for detection in detections:
            if detection.tag_id not in obstacle_tag_ids:
                continue
            pose = self._pose_from_detection(detection, calibration, timestamp)
            if pose is None:
                continue
            obstacles.append(
                ObstacleState(
                    tag_id=detection.tag_id,
                    pose=pose,
                    size_m=self.config.arena.obstacle_size_m,
                )
            )
        return obstacles

    def _draw_raw_overlay(self, frame: np.ndarray, board_state: BoardState) -> None:
        for detection in board_state.detections:
            corners = detection.corners_px.astype(int)
            for index in range(4):
                cv2.line(
                    frame,
                    tuple(corners[index]),
                    tuple(corners[(index + 1) % 4]),
                    RAW_OVERLAY_COLOR,
                    2,
                )
            center = tuple(detection.center_px.astype(int))
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

    def _draw_arena_overlay(self, frame: np.ndarray, board_state: BoardState) -> None:
        self._draw_arena_bounds(frame)
        self._draw_pose(frame, board_state.chaser, "Chaser", CHASER_COLOR)
        self._draw_pose(frame, board_state.evader, "Evader", EVADER_COLOR)
        for obstacle in board_state.obstacles:
            self._draw_obstacle(frame, obstacle)

    def _draw_arena_bounds(self, frame: np.ndarray) -> None:
        top = self.layout.stats_height + self.layout.buffer_px
        left = self.layout.buffer_px
        bottom = top + self.layout.board_height_px
        right = left + self.layout.board_width_px
        cv2.rectangle(frame, (left, top), (right, bottom), ARENA_BORDER_COLOR, 2)

    def _draw_pose(
        self,
        frame: np.ndarray,
        pose: Pose2D | None,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        if pose is None:
            return
        center = self._world_to_view(pose.x_m, pose.y_m)
        heading = (
            int(round(center[0] + 45.0 * math.cos(pose.yaw_rad))),
            int(round(center[1] - 45.0 * math.sin(pose.yaw_rad))),
        )
        cv2.circle(frame, center, 10, color, -1)
        cv2.line(frame, center, heading, color, 3)
        cv2.putText(
            frame,
            f"{label} ({pose.x_m:.3f}, {pose.y_m:.3f})m {math.degrees(pose.yaw_rad):.0f}deg",
            (center[0] + 12, center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    def _draw_obstacle(self, frame: np.ndarray, obstacle: ObstacleState) -> None:
        center = self._world_to_view(obstacle.pose.x_m, obstacle.pose.y_m)
        half = int(round(0.5 * obstacle.size_m * self.layout.pixels_per_m))
        corners = np.array(
            [[-half, -half], [half, -half], [half, half], [-half, half]],
            dtype=np.float32,
        )
        c = math.cos(obstacle.pose.yaw_rad)
        s = math.sin(obstacle.pose.yaw_rad)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
        rotated = corners @ rotation.T
        rotated[:, 0] += center[0]
        rotated[:, 1] = center[1] - rotated[:, 1]
        pts = rotated.astype(int)
        cv2.polylines(frame, [pts], isClosed=True, color=OBSTACLE_COLOR, thickness=2)
        cv2.putText(
            frame,
            f"obs {obstacle.tag_id}",
            (center[0] + 10, center[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            OBSTACLE_COLOR,
            2,
        )

    def _world_to_view(self, x_m: float, y_m: float) -> tuple[int, int]:
        left = self.layout.buffer_px
        top = self.layout.stats_height + self.layout.buffer_px
        x = left + int(
            round((x_m + 0.5 * self.layout.board_width_m) * self.layout.pixels_per_m)
        )
        y = top + int(
            round((0.5 * self.layout.board_height_m - y_m) * self.layout.pixels_per_m)
        )
        return x, y
