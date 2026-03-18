from __future__ import annotations

import time

import cv2

from online.config import TrackerConfig
from online.tracker import ARENA_WINDOW_NAME, RAW_WINDOW_NAME, BoardTracker


def main() -> None:
    tracker = BoardTracker(TrackerConfig())
    cv2.namedWindow(RAW_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(ARENA_WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame, board_state = tracker.process_next_frame()
            raw_view, arena_view = tracker.render_debug_views(frame, board_state)
            show_start = time.perf_counter()
            cv2.imshow(RAW_WINDOW_NAME, raw_view)
            cv2.imshow(ARENA_WINDOW_NAME, arena_view)
            board_state.stats.imshow_ms = (time.perf_counter() - show_start) * 1000.0
            waitkey_start = time.perf_counter()
            key = cv2.waitKey(1) & 0xFF
            board_state.stats.waitkey_ms = (
                time.perf_counter() - waitkey_start
            ) * 1000.0
            board_state.stats.display_ms = (
                board_state.stats.render_ms
                + board_state.stats.imshow_ms
                + board_state.stats.waitkey_ms
            )
            if key in (ord("q"), 27):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
