import socket
import struct
import sys
import termios
import threading
import time
import tty

ROBOT_IP = "192.168.1.3"
UDP_PORT = 8888
SEND_HZ = 20

LINEAR_STEP = 0.15
ANGULAR_STEP = 0.015
LINEAR_MAX = 1.0
ANGULAR_MAX = 1.0
MAX_INT16 = 32767


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class Teleop:
    def __init__(self) -> None:
        self.lin = 0.0
        self.ang = 0.0
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.packets_sent = 0

    @property
    def left(self) -> float:
        return clamp(self.lin - self.ang, -1.0, 1.0)

    @property
    def right(self) -> float:
        return clamp(self.lin + self.ang, -1.0, 1.0)

    def handle_key(self, ch: str) -> None:
        if ch == "w":
            self.lin = clamp(self.lin + LINEAR_STEP, -LINEAR_MAX, LINEAR_MAX)
        elif ch == "s":
            self.lin = clamp(self.lin - LINEAR_STEP, -LINEAR_MAX, LINEAR_MAX)
        elif ch == "a":
            self.ang = clamp(self.ang + ANGULAR_STEP, -ANGULAR_MAX, ANGULAR_MAX)
        elif ch == "d":
            self.ang = clamp(self.ang - ANGULAR_STEP, -ANGULAR_MAX, ANGULAR_MAX)
        elif ch == " ":
            self.lin = 0.0
            self.ang = 0.0
        elif ch in ("q", "\x1b"):  # q or Esc
            self.running = False

    def send_loop(self) -> None:
        dt = 1.0 / SEND_HZ
        while self.running:
            l16 = int(self.left * MAX_INT16)
            r16 = int(self.right * MAX_INT16)
            self.sock.sendto(struct.pack("<hh", l16, r16), (ROBOT_IP, UDP_PORT))
            self.packets_sent += 1
            time.sleep(dt)
        # send stop on exit
        self.sock.sendto(struct.pack("<hh", 0, 0), (ROBOT_IP, UDP_PORT))
        self.sock.close()

    def display(self) -> None:
        l16 = int(self.left * MAX_INT16)
        r16 = int(self.right * MAX_INT16)
        # move cursor to top-left and draw over previous frame
        sys.stdout.write("\033[H")
        lines = [
            "  Differential Drive Teleop",
            "  ─────────────────────────────────",
            f"  linear:  {self.lin:+.2f}   angular: {self.ang:+.2f}",
            f"  L motor: {l16:+6d}    R motor: {r16:+6d}",
            "  ─────────────────────────────────",
            "  W/S linear | A/D angular | Space stop | Q quit",
            f"  -> {ROBOT_IP}:{UDP_PORT}  {SEND_HZ}Hz  pkts:{self.packets_sent}",
        ]
        for line in lines:
            sys.stdout.write(f"{line:<50}\n")
        sys.stdout.flush()

    def run(self) -> None:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            # clear screen once
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()

            sender = threading.Thread(target=self.send_loop, daemon=True)
            sender.start()

            display_thread = threading.Thread(target=self._display_loop, daemon=True)
            display_thread.start()

            while self.running:
                ch = sys.stdin.read(1)
                if ch:
                    self.handle_key(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            self.running = False
            # reset terminal
            sys.stdout.write("\033[?25h\n")
            sys.stdout.flush()

    def _display_loop(self) -> None:
        while self.running:
            self.display()
            time.sleep(1.0 / SEND_HZ)


if __name__ == "__main__":
    Teleop().run()
