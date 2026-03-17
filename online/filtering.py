from __future__ import annotations

import math
from dataclasses import dataclass

from online.types import Pose2D


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _blend_angle(current: float, target: float, alpha: float) -> float:
    delta = _wrap_angle(target - current)
    return _wrap_angle(current + alpha * delta)


@dataclass(slots=True)
class PoseFilter:
    position_alpha: float
    yaw_alpha: float
    pose: Pose2D | None = None

    def update(self, measurement: Pose2D) -> Pose2D:
        if self.pose is None:
            self.pose = measurement
            return measurement
        self.pose = Pose2D(
            x=self.pose.x + self.position_alpha * (measurement.x - self.pose.x),
            y=self.pose.y + self.position_alpha * (measurement.y - self.pose.y),
            yaw=_blend_angle(self.pose.yaw, measurement.yaw, self.yaw_alpha),
            timestamp=measurement.timestamp,
        )
        return self.pose

    def clear(self) -> None:
        self.pose = None
