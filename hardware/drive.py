import socket
import struct
import threading
import time

from pynput import keyboard
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROBOT_IP = "192.168.1.3"
UDP_PORT = 8888
SEND_HZ = 20

LINEAR_STEP = 0.1
ANGULAR_STEP = 0.01
LINEAR_MAX = 1.0
ANGULAR_MAX = 1.0
MAX_INT16 = 32767


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Teleop:
    def __init__(self):
        self.lin = 0.0
        self.ang = 0.0
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.packets_sent = 0

    @property
    def left(self):
        return clamp(self.lin - self.ang, -1.0, 1.0)

    @property
    def right(self):
        return clamp(self.lin + self.ang, -1.0, 1.0)

    def on_press(self, key):
        try:
            c = key.char
        except AttributeError:
            if key == keyboard.Key.space:
                self.lin = 0.0
                self.ang = 0.0
            elif key == keyboard.Key.esc:
                self.running = False
            return

        if c == "w":
            self.lin = clamp(self.lin + LINEAR_STEP, -LINEAR_MAX, LINEAR_MAX)
        elif c == "s":
            self.lin = clamp(self.lin - LINEAR_STEP, -LINEAR_MAX, LINEAR_MAX)
        elif c == "a":
            self.ang = clamp(self.ang + ANGULAR_STEP, -ANGULAR_MAX, ANGULAR_MAX)
        elif c == "d":
            self.ang = clamp(self.ang - ANGULAR_STEP, -ANGULAR_MAX, ANGULAR_MAX)
        elif c == "q":
            self.running = False

    def send_loop(self):
        dt = 1.0 / SEND_HZ
        while self.running:
            l16 = int(self.left * MAX_INT16)
            r16 = int(self.right * MAX_INT16)
            self.sock.sendto(struct.pack("<hh", l16, r16), (ROBOT_IP, UDP_PORT))
            self.packets_sent += 1
            time.sleep(dt)
        self.sock.sendto(struct.pack("<hh", 0, 0), (ROBOT_IP, UDP_PORT))
        self.sock.close()

    def make_bar(self, value, width=30):
        mid = width // 2
        filled = int(abs(value) * mid)
        bar = [" "] * width
        if value > 0:
            for i in range(mid, mid + filled):
                bar[i] = "█"
        elif value < 0:
            for i in range(mid - filled, mid):
                bar[i] = "█"
        bar[mid] = "│"
        color = (
            "green" if abs(value) < 0.5 else ("yellow" if abs(value) < 0.8 else "red")
        )
        return Text("".join(bar), style=f"bold {color}")

    def render(self):
        l16 = int(self.left * MAX_INT16)
        r16 = int(self.right * MAX_INT16)

        tbl = Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column(width=10, justify="right")
        tbl.add_column(width=8, justify="right")
        tbl.add_column(width=32)

        tbl.add_row(
            Text("linear", style="bold cyan"),
            Text(f"{self.lin:+.2f}", style="bold white"),
            self.make_bar(self.lin),
        )
        tbl.add_row(
            Text("angular", style="bold magenta"),
            Text(f"{self.ang:+.2f}", style="bold white"),
            self.make_bar(self.ang),
        )
        tbl.add_row(Text(""), Text(""), Text(""))
        tbl.add_row(
            Text("L motor", style="dim"),
            Text(f"{l16:+6d}", style="bold white"),
            self.make_bar(self.left),
        )
        tbl.add_row(
            Text("R motor", style="dim"),
            Text(f"{r16:+6d}", style="bold white"),
            self.make_bar(self.right),
        )

        keys = Text.assemble(
            ("  W/S ", "bold cyan"),
            ("linear  ", "dim"),
            ("  A/D ", "bold magenta"),
            ("angular  ", "dim"),
            ("  Space ", "bold yellow"),
            ("stop  ", "dim"),
            ("  Q/Esc ", "bold red"),
            ("quit", "dim"),
        )

        status = Text.assemble(
            ("  ● ", "bold green"),
            (f"{ROBOT_IP}:{UDP_PORT}  ", "dim"),
            (f"  {SEND_HZ}Hz  ", "dim"),
            (f"  pkts: {self.packets_sent}", "dim"),
        )

        layout = Layout()
        layout.split_column(
            Layout(Align.center(keys), size=1),
            Layout(Text("")),
            Layout(tbl, size=7),
            Layout(Text("")),
            Layout(Align.center(status), size=1),
        )

        return Panel(
            layout,
            title="[bold white]🤖 Differential Drive Teleop[/]",
            border_style="bright_blue",
            height=14,
            width=60,
        )

    def run(self):
        sender = threading.Thread(target=self.send_loop, daemon=True)
        sender.start()

        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        console = Console()
        with Live(
            self.render(), console=console, refresh_per_second=SEND_HZ, screen=True
        ) as live:
            while self.running:
                live.update(self.render())
                time.sleep(1.0 / SEND_HZ)

        listener.stop()
        sender.join(timeout=1)


if __name__ == "__main__":
    Teleop().run()
