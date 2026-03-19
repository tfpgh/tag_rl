from __future__ import annotations

import select
import sys
import termios
import time
import tty

import cv2

from online.control.teleop import TeleopController
from online.core.config import TrackerConfig
from online.data.logger import RunLogger
from online.tracking.tracker import ARENA_WINDOW_NAME, RAW_WINDOW_NAME, BoardTracker


def _print_status(logger: RunLogger, frame_index: int, packets_sent: int) -> None:
    sys.stdout.write("\033[H")
    lines = [
        "  Real Robot Data Collection",
        "  --------------------------",
        f"  run dir: {logger.run_dir}",
        f"  frames: {frame_index}",
        f"  packets sent: {packets_sent}",
        "  W/S linear | A/D angular | Space stop | Q quit",
    ]
    for line in lines:
        sys.stdout.write(f"{line}\n")
    sys.stdout.flush()


def main() -> None:
    config = TrackerConfig()
    tracker = BoardTracker(config)
    teleop = TeleopController(config.teleop)
    logger = RunLogger(config)

    show_raw = config.display.show_raw_window
    show_arena = config.display.show_arena_window
    preview_interval_s = 1.0 / config.display.preview_fps
    next_preview_time = 0.0
    frame_index = 0

    if show_raw:
        cv2.namedWindow(RAW_WINDOW_NAME, cv2.WINDOW_NORMAL)
    if show_arena:
        cv2.namedWindow(ARENA_WINDOW_NAME, cv2.WINDOW_NORMAL)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    teleop.start()

    try:
        tty.setcbreak(fd)
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

        running = True
        while running:
            while select.select([sys.stdin], [], [], 0.0)[0]:
                key = sys.stdin.read(1)
                if key and not teleop.handle_key(key):
                    running = False
                    break

            frame, board_state = tracker.process_next_frame()
            command_state = teleop.snapshot()
            logger.log_sample(frame_index, time.monotonic(), board_state, command_state)

            for event in teleop.pop_events():
                logger.log_command(event)

            now = time.perf_counter()
            if now >= next_preview_time:
                raw_view, arena_view = tracker.render_debug_views(frame, board_state)
                if show_raw:
                    cv2.imshow(RAW_WINDOW_NAME, raw_view)
                if show_arena:
                    cv2.imshow(ARENA_WINDOW_NAME, arena_view)
                next_preview_time = now + preview_interval_s

            _print_status(logger, frame_index, command_state.packets_sent)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                running = False

            frame_index += 1
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        teleop.stop()
        logger.close()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
