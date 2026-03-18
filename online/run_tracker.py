from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import cast

import cv2
import numpy as np

from online.config import TrackerConfig
from online.state import BoardState
from online.tracker import ARENA_WINDOW_NAME, RAW_WINDOW_NAME, BoardTracker


@dataclass(slots=True)
class PreviewMetrics:
    render_ms: float = 0.0
    imshow_ms: float = 0.0
    waitkey_ms: float = 0.0
    display_ms: float = 0.0
    fps: float = 0.0


@dataclass(slots=True)
class PreviewFrame:
    frame: np.ndarray
    board_state: BoardState


class PreviewLoop:
    def __init__(
        self, tracker: BoardTracker, config: TrackerConfig, stop_event: Event
    ) -> None:
        self._tracker = tracker
        self._config = config
        self._stop_event = stop_event
        self._lock = Lock()
        self._latest: PreviewFrame | None = None
        self._metrics = PreviewMetrics()
        self._show_raw = config.display.show_raw_window
        self._show_arena = config.display.show_arena_window

    def publish(self, frame: np.ndarray, board_state: BoardState) -> None:
        with self._lock:
            self._latest = PreviewFrame(frame=frame, board_state=board_state)

    def metrics(self) -> PreviewMetrics:
        with self._lock:
            return PreviewMetrics(
                render_ms=self._metrics.render_ms,
                imshow_ms=self._metrics.imshow_ms,
                waitkey_ms=self._metrics.waitkey_ms,
                display_ms=self._metrics.display_ms,
                fps=self._metrics.fps,
            )

    def run(self) -> None:
        open_windows: set[str] = set()
        min_interval = 1.0 / self._config.display.preview_fps

        print("Controls: q/esc quit, 1 toggle raw, 2 toggle arena, 0 headless, 3 both")

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            self._set_window_visible(RAW_WINDOW_NAME, self._show_raw, open_windows)
            self._set_window_visible(ARENA_WINDOW_NAME, self._show_arena, open_windows)

            with self._lock:
                latest = self._latest

            render_ms = 0.0
            imshow_ms = 0.0
            waitkey_ms = 0.0

            if latest is not None and (self._show_raw or self._show_arena):
                frame = cast(np.ndarray, latest.frame)
                raw_view, arena_view, render_ms = self._tracker.render_debug_views(
                    frame, latest.board_state
                )
                show_start = time.perf_counter()
                if self._show_raw:
                    cv2.imshow(RAW_WINDOW_NAME, raw_view)
                if self._show_arena:
                    cv2.imshow(ARENA_WINDOW_NAME, arena_view)
                imshow_ms = (time.perf_counter() - show_start) * 1000.0

            waitkey_start = time.perf_counter()
            key = cv2.waitKey(1) & 0xFF
            waitkey_ms = (time.perf_counter() - waitkey_start) * 1000.0

            total_ms = (time.perf_counter() - loop_start) * 1000.0
            display_ms = render_ms + imshow_ms + waitkey_ms
            fps = 1000.0 / total_ms if total_ms > 0.0 else 0.0

            with self._lock:
                self._metrics = PreviewMetrics(
                    render_ms=render_ms,
                    imshow_ms=imshow_ms,
                    waitkey_ms=waitkey_ms,
                    display_ms=display_ms,
                    fps=fps,
                )

            if key in (ord("q"), 27):
                self._stop_event.set()
            elif key == ord("1"):
                self._show_raw = not self._show_raw
            elif key == ord("2"):
                self._show_arena = not self._show_arena
            elif key == ord("0"):
                self._show_raw = False
                self._show_arena = False
            elif key == ord("3"):
                self._show_raw = True
                self._show_arena = True

            sleep_s = min_interval - (time.perf_counter() - loop_start)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        cv2.destroyAllWindows()

    @staticmethod
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
    stop_event = Event()
    preview = PreviewLoop(tracker, config, stop_event)
    preview_thread = Thread(target=preview.run, daemon=True)
    preview_thread.start()

    previous_frame_end: float | None = None

    try:
        while not stop_event.is_set():
            frame_start = time.perf_counter()
            frame, board_state = tracker.process_next_frame()

            preview_metrics = preview.metrics()
            board_state.stats.render_ms = preview_metrics.render_ms
            board_state.stats.imshow_ms = preview_metrics.imshow_ms
            board_state.stats.waitkey_ms = preview_metrics.waitkey_ms
            board_state.stats.display_ms = preview_metrics.display_ms

            if previous_frame_end is not None:
                tracker_total_ms = (frame_start - previous_frame_end) * 1000.0
                board_state.stats.end_to_end_ms = tracker_total_ms
                board_state.stats.fps = (
                    1000.0 / tracker_total_ms if tracker_total_ms > 0.0 else 0.0
                )

            preview.publish(frame.copy(), board_state)
            previous_frame_end = time.perf_counter()
    finally:
        stop_event.set()
        preview_thread.join(timeout=1.0)
        tracker.close()


if __name__ == "__main__":
    main()
