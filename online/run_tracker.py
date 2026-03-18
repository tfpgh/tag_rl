from __future__ import annotations

import time
from dataclasses import dataclass

import cv2

from online.config import TrackerConfig
from online.tracker import ARENA_WINDOW_NAME, RAW_WINDOW_NAME, BoardTracker


@dataclass(slots=True)
class CompletedFrameMetrics:
    fps: float = 0.0
    render_ms: float = 0.0
    imshow_ms: float = 0.0
    waitkey_ms: float = 0.0
    display_ms: float = 0.0
    end_to_end_ms: float = 0.0


def _set_window_visible(name: str, visible: bool, open_windows: set[str]) -> None:
    if visible and name not in open_windows:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        open_windows.add(name)
    elif not visible and name in open_windows:
        cv2.destroyWindow(name)
        open_windows.remove(name)


def main() -> None:
    config = TrackerConfig()
    tracker = BoardTracker(config)
    show_raw = config.display.show_raw_window
    show_arena = config.display.show_arena_window
    open_windows: set[str] = set()
    previous_metrics = CompletedFrameMetrics()

    print("Controls: q/esc quit, 1 toggle raw, 2 toggle arena, 0 headless, 3 both")

    try:
        while True:
            frame_loop_start = time.perf_counter()
            _set_window_visible(RAW_WINDOW_NAME, show_raw, open_windows)
            _set_window_visible(ARENA_WINDOW_NAME, show_arena, open_windows)

            frame, board_state = tracker.process_next_frame()
            board_state.stats.fps = previous_metrics.fps
            board_state.stats.render_ms = previous_metrics.render_ms
            board_state.stats.imshow_ms = previous_metrics.imshow_ms
            board_state.stats.waitkey_ms = previous_metrics.waitkey_ms
            board_state.stats.display_ms = previous_metrics.display_ms
            board_state.stats.end_to_end_ms = previous_metrics.end_to_end_ms

            raw_view, arena_view, render_ms = tracker.render_debug_views(
                frame, board_state
            )

            show_start = time.perf_counter()
            if show_raw:
                cv2.imshow(RAW_WINDOW_NAME, raw_view)
            if show_arena:
                cv2.imshow(ARENA_WINDOW_NAME, arena_view)
            imshow_ms = (time.perf_counter() - show_start) * 1000.0

            waitkey_start = time.perf_counter()
            key = cv2.waitKey(1) & 0xFF
            waitkey_ms = (time.perf_counter() - waitkey_start) * 1000.0

            end_to_end_ms = (time.perf_counter() - frame_loop_start) * 1000.0
            display_ms = render_ms + imshow_ms + waitkey_ms
            fps = 1000.0 / end_to_end_ms if end_to_end_ms > 0.0 else 0.0
            previous_metrics = CompletedFrameMetrics(
                fps=fps,
                render_ms=render_ms,
                imshow_ms=imshow_ms,
                waitkey_ms=waitkey_ms,
                display_ms=display_ms,
                end_to_end_ms=end_to_end_ms,
            )

            if key in (ord("q"), 27):
                break
            if key == ord("1"):
                show_raw = not show_raw
            elif key == ord("2"):
                show_arena = not show_arena
            elif key == ord("0"):
                show_raw = False
                show_arena = False
            elif key == ord("3"):
                show_raw = True
                show_arena = True
    finally:
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
