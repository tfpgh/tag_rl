from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from online.config import TrackingConfig
from online.filtering import PoseFilter
from online.types import Pose2D, TagDetection, TrackedBody, TrackedObstacle, WorldState


def _compute_yaw_from_corners(corners_world: np.ndarray) -> float:
    left = 0.5 * (corners_world[0] + corners_world[3])
    right = 0.5 * (corners_world[1] + corners_world[2])
    axis = right - left
    return math.atan2(float(axis[1]), float(axis[0]))


@dataclass(slots=True)
class TrackingBuffers:
    chaser_filter: PoseFilter
    evader_filter: PoseFilter
    obstacle_filters: dict[int, PoseFilter]
    obstacle_poses: dict[int, Pose2D]


class WorldTracker:
    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self._buffers = TrackingBuffers(
            chaser_filter=PoseFilter(config.position_alpha, config.yaw_alpha),
            evader_filter=PoseFilter(config.position_alpha, config.yaw_alpha),
            obstacle_filters={},
            obstacle_poses={},
        )

    def _pose_from_detection(
        self,
        detection: TagDetection,
        transform,
        heading_offset: float,
        timestamp: float,
    ) -> Pose2D:
        center_world = transform(np.asarray([detection.center_px], dtype=np.float32))[0]
        corners_world = transform(np.asarray(detection.corners_px, dtype=np.float32))
        yaw = _compute_yaw_from_corners(corners_world) + heading_offset
        return Pose2D(
            x=float(center_world[0]),
            y=float(center_world[1]),
            yaw=float(yaw),
            timestamp=timestamp,
        )

    def _tracked_body(
        self,
        tag_id: int,
        label: str,
        filter_state: PoseFilter,
        detection: TagDetection | None,
        transform,
        heading_offset: float,
        timeout_s: float,
        timestamp: float,
    ) -> TrackedBody:
        raw_pose = None
        filtered_pose = filter_state.pose
        visible = detection is not None
        if detection is not None:
            raw_pose = self._pose_from_detection(
                detection, transform, heading_offset, timestamp
            )
            filtered_pose = filter_state.update(raw_pose)
        age_s = (
            float("inf")
            if filtered_pose is None
            else max(0.0, timestamp - filtered_pose.timestamp)
        )
        stale = filtered_pose is None or age_s > timeout_s
        return TrackedBody(
            tag_id=tag_id,
            label=label,
            visible=visible,
            stale=stale,
            age_s=age_s,
            raw_pose=raw_pose,
            filtered_pose=filtered_pose,
        )

    def update(
        self,
        detections: list[TagDetection],
        *,
        transform,
        timestamp: float,
        frame_id: int,
        calibration_ready: bool,
    ) -> WorldState:
        detections_by_id = {det.tag_id: det for det in detections}
        chaser = self._tracked_body(
            self.config.chaser_tag_id,
            "chaser",
            self._buffers.chaser_filter,
            detections_by_id.get(self.config.chaser_tag_id),
            transform,
            self.config.robot_heading_offset_rad,
            self.config.robot_pose_timeout_s,
            timestamp,
        )
        evader = self._tracked_body(
            self.config.evader_tag_id,
            "evader",
            self._buffers.evader_filter,
            detections_by_id.get(self.config.evader_tag_id),
            transform,
            self.config.robot_heading_offset_rad,
            self.config.robot_pose_timeout_s,
            timestamp,
        )

        visible_obstacles: dict[int, Pose2D] = {}
        for detection in detections:
            if detection.tag_id < self.config.obstacle_tag_min_id:
                continue
            pose = self._pose_from_detection(
                detection,
                transform,
                self.config.obstacle_heading_offset_rad,
                timestamp,
            )
            filter_state = self._buffers.obstacle_filters.setdefault(
                detection.tag_id,
                PoseFilter(self.config.position_alpha, self.config.yaw_alpha),
            )
            visible_obstacles[detection.tag_id] = filter_state.update(pose)

        self._buffers.obstacle_poses.update(visible_obstacles)
        obstacles: list[TrackedObstacle] = []
        for tag_id, pose in list(self._buffers.obstacle_poses.items()):
            visible = tag_id in visible_obstacles
            age_s = max(0.0, timestamp - pose.timestamp)
            stale = age_s > self.config.obstacle_hold_timeout_s
            if stale:
                self._buffers.obstacle_poses.pop(tag_id, None)
                self._buffers.obstacle_filters.pop(tag_id, None)
                continue
            obstacles.append(
                TrackedObstacle(
                    tag_id=tag_id,
                    visible=visible,
                    stale=False,
                    age_s=age_s,
                    pose=pose,
                    size_m=self.config.obstacle_size_m,
                )
            )
        obstacles.sort(key=lambda obstacle: obstacle.tag_id)

        ready = calibration_ready and not chaser.stale and not evader.stale
        return WorldState(
            timestamp=timestamp,
            ready=ready,
            frame_id=frame_id,
            chaser=chaser,
            evader=evader,
            obstacles=obstacles,
        )
