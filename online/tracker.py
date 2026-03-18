from __future__ import annotations

import math
import time

import cv2
import numpy as np

from online.apriltags import AprilTagTracker
from online.calibration import (
    CornerTagCalibrator,
    build_arena_view_layout,
    transform_points,
)
from online.camera import CameraStream
from online.config import TrackerConfig
from online.state import (
    ArenaCalibration,
    BoardState,
    ObstacleState,
    Pose2D,
    TagDetection,
    TrackingStats,
)

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


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _blend_angle(previous: float, current: float, alpha: float) -> float:
    delta = _wrap_angle(current - previous)
    return _wrap_angle(previous + alpha * delta)


class BoardTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.layout = build_arena_view_layout(config.arena, config.view)
        self.camera = CameraStream(config.camera)
        self.detector = AprilTagTracker(config)
        self.calibrator = CornerTagCalibrator(config, self.layout)
        self._previous_frame_time: float | None = None
        self._smoothed_poses: dict[int, Pose2D] = {}

    def close(self) -> None:
        self.camera.release()

    def process_next_frame(self) -> tuple[np.ndarray, BoardState]:
        frame_start = time.perf_counter()
        frame_timestamp, frame = self.camera.read()
        capture_ms = (time.perf_counter() - frame_start) * 1000.0
        loop_start = time.perf_counter()
        detector_start = time.perf_counter()
        detections = self.detector.detect(frame)
        detector_ms = (time.perf_counter() - detector_start) * 1000.0

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
            render_ms=0.0,
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

    def render_debug_views(
        self, frame: np.ndarray, board_state: BoardState
    ) -> tuple[np.ndarray, np.ndarray]:
        render_start = time.perf_counter()
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
        board_state.stats.render_ms = (time.perf_counter() - render_start) * 1000.0
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
        yaw = math.atan2(float(edge_delta[1]), float(edge_delta[0]))
        pose = Pose2D(
            x_mm=float(center[0]),
            y_mm=float(center[1]),
            yaw_rad=yaw,
            visible=True,
            last_seen_timestamp=timestamp,
        )
        return self._smooth_pose(detection.tag_id, pose)

    def _smooth_pose(self, tag_id: int, pose: Pose2D) -> Pose2D:
        previous = self._smoothed_poses.get(tag_id)
        if previous is None:
            self._smoothed_poses[tag_id] = pose
            return pose
        alpha = self.config.smoothing_alpha
        smoothed = Pose2D(
            x_mm=(1.0 - alpha) * previous.x_mm + alpha * pose.x_mm,
            y_mm=(1.0 - alpha) * previous.y_mm + alpha * pose.y_mm,
            yaw_rad=_blend_angle(previous.yaw_rad, pose.yaw_rad, alpha),
            visible=True,
            last_seen_timestamp=pose.last_seen_timestamp,
        )
        self._smoothed_poses[tag_id] = smoothed
        return smoothed

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
                    size_mm=self.config.arena.obstacle_size_mm,
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
            cv2.putText(
                frame,
                f"id={detection.tag_id}",
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                RAW_OVERLAY_COLOR,
                2,
            )

        stats = (
            f"FPS {board_state.stats.fps:.1f} | cap {board_state.stats.capture_ms:.1f}ms"
            f" | det {board_state.stats.detector_ms:.1f}ms | track {board_state.stats.tracking_ms:.1f}ms"
            f" | loop {board_state.stats.loop_ms:.1f}ms | tags {board_state.stats.visible_tags}"
        )
        cv2.putText(
            frame,
            stats,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            RAW_OVERLAY_COLOR,
            2,
        )
        calibration_text = (
            "CALIBRATED"
            if board_state.calibration.valid
            else (
                "CALIBRATING "
                f"{board_state.calibration.samples_collected}/{board_state.calibration.required_samples}"
            )
        )
        cv2.putText(
            frame,
            calibration_text,
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            CALIBRATED_COLOR if board_state.calibration.valid else CALIBRATING_COLOR,
            2,
        )

    def _draw_arena_overlay(self, frame: np.ndarray, board_state: BoardState) -> None:
        self._draw_arena_bounds(frame)
        self._draw_pose(frame, board_state.chaser, "Chaser", CHASER_COLOR)
        self._draw_pose(frame, board_state.evader, "Evader", EVADER_COLOR)
        for obstacle in board_state.obstacles:
            self._draw_obstacle(frame, obstacle)

        stats = (
            f"FPS {board_state.stats.fps:.1f} | cap {board_state.stats.capture_ms:.1f}ms"
            f" | det {board_state.stats.detector_ms:.1f}ms | track {board_state.stats.tracking_ms:.1f}ms"
            f" | draw {board_state.stats.render_ms:.1f}ms | px/mm {board_state.calibration.pixels_per_mm:.3f}"
        )
        cv2.rectangle(
            frame,
            (0, 0),
            (self.layout.frame_width, self.layout.stats_height),
            HEADER_BG_COLOR,
            thickness=-1,
        )
        cv2.putText(
            frame,
            stats,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            HEADER_TEXT_COLOR,
            2,
        )

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
            cv2.putText(
                frame,
                f"{label}: missing",
                (10, self.layout.frame_height - (50 if label == "Chaser" else 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
            return
        center = self._world_to_view(pose.x_mm, pose.y_mm)
        heading = (
            int(round(center[0] + 45.0 * math.cos(pose.yaw_rad))),
            int(round(center[1] - 45.0 * math.sin(pose.yaw_rad))),
        )
        cv2.circle(frame, center, 10, color, -1)
        cv2.line(frame, center, heading, color, 3)
        cv2.putText(
            frame,
            f"{label} ({pose.x_mm:.0f}, {pose.y_mm:.0f})mm {math.degrees(pose.yaw_rad):.0f}deg",
            (center[0] + 12, center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    def _draw_obstacle(self, frame: np.ndarray, obstacle: ObstacleState) -> None:
        center = self._world_to_view(obstacle.pose.x_mm, obstacle.pose.y_mm)
        half = int(round(0.5 * obstacle.size_mm * self.layout.pixels_per_mm))
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

    def _world_to_view(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        left = self.layout.buffer_px
        top = self.layout.stats_height + self.layout.buffer_px
        x = left + int(
            round((x_mm + 0.5 * self.layout.board_width_mm) * self.layout.pixels_per_mm)
        )
        y = top + int(
            round(
                (0.5 * self.layout.board_height_mm - y_mm) * self.layout.pixels_per_mm
            )
        )
        return x, y
