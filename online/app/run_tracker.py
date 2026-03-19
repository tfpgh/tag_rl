from __future__ import annotations

import time

import cv2

from online.core.config import TrackerConfig
from online.tracking.tracker import ARENA_WINDOW_NAME, RAW_WINDOW_NAME, BoardTracker


def main() -> None:
    config = TrackerConfig()
    config.display.show_raw_window = True
    config.display.show_arena_window = True
    tracker = BoardTracker(config)

    show_raw = config.display.show_raw_window
    show_arena = config.display.show_arena_window
    preview_interval_s = 1.0 / config.display.preview_fps
    next_preview_time = 0.0

    if show_raw:
        cv2.namedWindow(RAW_WINDOW_NAME, cv2.WINDOW_NORMAL)
    if show_arena:
        cv2.namedWindow(ARENA_WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("Press q or esc to quit")

    try:
        while True:
            frame, board_state = tracker.process_next_frame()

            now = time.perf_counter()
            if now >= next_preview_time:
                raw_view, arena_view = tracker.render_debug_views(frame, board_state)
                if show_raw:
                    cv2.imshow(RAW_WINDOW_NAME, raw_view)
                if show_arena:
                    cv2.imshow(ARENA_WINDOW_NAME, arena_view)
                next_preview_time = now + preview_interval_s

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
