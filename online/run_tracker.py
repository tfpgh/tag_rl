from __future__ import annotations

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
            cv2.imshow(RAW_WINDOW_NAME, raw_view)
            cv2.imshow(ARENA_WINDOW_NAME, arena_view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
