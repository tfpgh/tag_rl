import sys
import termios
import threading
import time
import tty

from online.config import TeleopConfig
from online.teleop import TeleopController


class Teleop:
    def __init__(self) -> None:
        self.controller = TeleopController(TeleopConfig())
        self.running = True

    def handle_key(self, ch: str) -> None:
        if not self.controller.handle_key(ch):
            self.running = False

    def display(self) -> None:
        state = self.controller.snapshot()
        sys.stdout.write("\033[H")
        lines = [
            "  Differential Drive Teleop",
            "  ─────────────────────────────────",
            f"  linear:  {state.linear:+.2f}   angular: {state.angular:+.2f}",
            f"  L motor: {state.left_int16:+6d}    R motor: {state.right_int16:+6d}",
            "  ─────────────────────────────────",
            "  W/S linear | A/D angular | Space stop | Q quit",
            f"  -> {self.controller.config.robot_ip}:{self.controller.config.udp_port}"
            f"  {self.controller.config.send_hz:.0f}Hz  pkts:{state.packets_sent}",
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

            self.controller.start()
            display_thread = threading.Thread(target=self._display_loop, daemon=True)
            display_thread.start()

            while self.running:
                ch = sys.stdin.read(1)
                if ch:
                    self.handle_key(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            self.running = False
            self.controller.stop()
            # reset terminal
            sys.stdout.write("\033[?25h\n")
            sys.stdout.flush()

    def _display_loop(self) -> None:
        while self.running:
            self.display()
            time.sleep(1.0 / self.controller.config.send_hz)


if __name__ == "__main__":
    Teleop().run()
