from __future__ import annotations

import socket
import struct
import time

from online.config import RobotEndpoint
from online.types import RobotCommandState

MAX_INT16 = 32767


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


class RobotClient:
    def __init__(self, endpoint: RobotEndpoint) -> None:
        self.endpoint = endpoint
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command = RobotCommandState(name=endpoint.name)

    def send(
        self, left: float, right: float, *, watchdog_stop: bool = False
    ) -> RobotCommandState:
        left = _clamp(left)
        right = _clamp(right)
        payload = struct.pack("<hh", int(left * MAX_INT16), int(right * MAX_INT16))
        self.sock.sendto(payload, (self.endpoint.ip, self.endpoint.port))
        self.command.timestamp = time.time()
        self.command.left = left
        self.command.right = right
        self.command.packets_sent += 1
        self.command.watchdog_stop = watchdog_stop
        return self.command

    def stop(self) -> RobotCommandState:
        return self.send(0.0, 0.0, watchdog_stop=True)

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.sock.close()
