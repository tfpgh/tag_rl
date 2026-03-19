from __future__ import annotations

import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass

from online.core.config import TeleopConfig

MAX_INT16 = 32767


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(slots=True)
class CommandState:
    monotonic_time: float
    linear: float
    angular: float
    left: float
    right: float
    left_int16: int
    right_int16: int
    packets_sent: int


@dataclass(slots=True)
class CommandEvent:
    monotonic_time: float
    left_int16: int
    right_int16: int
    packets_sent: int


class TeleopController:
    def __init__(self, config: TeleopConfig) -> None:
        self.config = config
        self.linear = 0.0
        self.angular = 0.0
        self.packets_sent = 0
        self.running = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._events: deque[CommandEvent] = deque()

    @property
    def left(self) -> float:
        return clamp(self.linear - self.angular, -1.0, 1.0)

    @property
    def right(self) -> float:
        return clamp(self.linear + self.angular, -1.0, 1.0)

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._send_packet(0, 0)
        self._socket.close()

    def handle_key(self, key: str) -> bool:
        with self._lock:
            if key == "w":
                self.linear = clamp(
                    self.linear + self.config.linear_step,
                    -self.config.linear_max,
                    self.config.linear_max,
                )
            elif key == "s":
                self.linear = clamp(
                    self.linear - self.config.linear_step,
                    -self.config.linear_max,
                    self.config.linear_max,
                )
            elif key == "a":
                self.angular = clamp(
                    self.angular + self.config.angular_step,
                    -self.config.angular_max,
                    self.config.angular_max,
                )
            elif key == "d":
                self.angular = clamp(
                    self.angular - self.config.angular_step,
                    -self.config.angular_max,
                    self.config.angular_max,
                )
            elif key == " ":
                self.linear = 0.0
                self.angular = 0.0
            elif key in ("q", "\x1b"):
                return False
        return True

    def set_command(self, linear: float, angular: float) -> None:
        with self._lock:
            self.linear = clamp(linear, -self.config.linear_max, self.config.linear_max)
            self.angular = clamp(
                angular, -self.config.angular_max, self.config.angular_max
            )

    def snapshot(self) -> CommandState:
        with self._lock:
            linear = self.linear
            angular = self.angular
            left = clamp(self.linear - self.angular, -1.0, 1.0)
            right = clamp(self.linear + self.angular, -1.0, 1.0)
            left_int16 = int(left * MAX_INT16)
            right_int16 = int(right * MAX_INT16)
            packets_sent = self.packets_sent
        return CommandState(
            monotonic_time=time.monotonic(),
            linear=linear,
            angular=angular,
            left=left,
            right=right,
            left_int16=left_int16,
            right_int16=right_int16,
            packets_sent=packets_sent,
        )

    def pop_events(self) -> list[CommandEvent]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def _send_loop(self) -> None:
        interval = 1.0 / self.config.send_hz
        while self.running:
            with self._lock:
                left_int16 = int(self.left * MAX_INT16)
                right_int16 = int(self.right * MAX_INT16)
            self._send_packet(left_int16, right_int16)
            time.sleep(interval)

    def _send_packet(self, left_int16: int, right_int16: int) -> None:
        self._socket.sendto(
            struct.pack("<hh", left_int16, right_int16),
            (self.config.robot_ip, self.config.udp_port),
        )
        with self._lock:
            self.packets_sent += 1
            self._events.append(
                CommandEvent(
                    monotonic_time=time.monotonic(),
                    left_int16=left_int16,
                    right_int16=right_int16,
                    packets_sent=self.packets_sent,
                )
            )
