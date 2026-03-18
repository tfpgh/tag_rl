from __future__ import annotations

import cv2

from online.state import TrackerConfig
from online.tracker import BoardTracker


def main() -> None:
    tracker = BoardTracker(TrackerConfig())
    cv2.namedWindow("Tracker Raw", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tracker Arena", cv2.WINDOW_NORMAL)
    try:
        while True:
            frame, board_state = tracker.process_next_frame()
            raw_view, arena_view = tracker.render_debug_views(frame, board_state)
            cv2.imshow("Tracker Raw", raw_view)
            cv2.imshow("Tracker Arena", arena_view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
