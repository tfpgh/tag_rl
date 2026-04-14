from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

from online.game.config import RobotEndpointConfig

MAX_INT16 = 32767


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True, slots=True)
class RobotCommand:
    left: float
    right: float

    @property
    def left_int16(self) -> int:
        return int(_clamp(self.left, -1.0, 1.0) * MAX_INT16)

    @property
    def right_int16(self) -> int:
        return int(_clamp(self.right, -1.0, 1.0) * MAX_INT16)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "left": self.left,
            "right": self.right,
            "left_int16": self.left_int16,
            "right_int16": self.right_int16,
        }


class DualRobotController:
    def __init__(
        self, chaser: RobotEndpointConfig, evader: RobotEndpointConfig, enabled: bool
    ) -> None:
        self._enabled = enabled
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._chaser = chaser
        self._evader = evader

    def close(self) -> None:
        self.zero_all()
        self._socket.close()

    def zero_all(self) -> None:
        self.send(RobotCommand(0.0, 0.0), RobotCommand(0.0, 0.0))

    def send(self, chaser: RobotCommand, evader: RobotCommand) -> None:
        if not self._enabled:
            return
        self._send_one(self._chaser, chaser)
        self._send_one(self._evader, evader)

    def _send_one(self, endpoint: RobotEndpointConfig, command: RobotCommand) -> None:
        payload = struct.pack("<hh", command.left_int16, command.right_int16)
        self._socket.sendto(payload, (endpoint.robot_ip, endpoint.udp_port))
