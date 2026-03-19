from __future__ import annotations

import time

import cv2

from online.control.teleop import TeleopController
from online.core.config import TrackerConfig
from online.data.logger import RunLogger
from online.data.protocols import default_protocols
from online.tracking.tracker import ARENA_WINDOW_NAME, RAW_WINDOW_NAME, BoardTracker


def _print_status(
    run_dir: str, protocol_name: str, repeat_index: int, segment_label: str
) -> None:
    print(
        f"\rcollecting {protocol_name} repeat {repeat_index} segment {segment_label} -> {run_dir}",
        end="",
        flush=True,
    )


def main() -> None:
    config = TrackerConfig()
    protocols = default_protocols()
    tracker = BoardTracker(config)
    teleop = TeleopController(config.teleop)

    show_raw = config.display.show_raw_window
    show_arena = config.display.show_arena_window
    preview_interval_s = 1.0 / config.display.preview_fps
    next_preview_time = 0.0

    if show_raw:
        cv2.namedWindow(RAW_WINDOW_NAME, cv2.WINDOW_NORMAL)
    if show_arena:
        cv2.namedWindow(ARENA_WINDOW_NAME, cv2.WINDOW_NORMAL)

    teleop.start()

    try:
        for protocol in protocols:
            for repeat_index in range(config.collection.scripted_repeat_count):
                config.teleop.run_label = f"{protocol.name}_{repeat_index + 1}"
                logger = RunLogger(config)
                frame_index = 0
                try:
                    for segment in protocol.segments:
                        teleop.set_command(segment.linear, segment.angular)
                        segment_end = time.monotonic() + segment.duration_s
                        while time.monotonic() < segment_end:
                            frame, board_state = tracker.process_next_frame()
                            command_state = teleop.snapshot()
                            logger.log_sample(
                                frame_index,
                                time.monotonic(),
                                board_state,
                                command_state,
                            )
                            for event in teleop.pop_events():
                                logger.log_command(event)

                            now = time.perf_counter()
                            if now >= next_preview_time:
                                raw_view, arena_view = tracker.render_debug_views(
                                    frame, board_state
                                )
                                if show_raw:
                                    cv2.imshow(RAW_WINDOW_NAME, raw_view)
                                if show_arena:
                                    cv2.imshow(ARENA_WINDOW_NAME, arena_view)
                                next_preview_time = now + preview_interval_s

                            _print_status(
                                str(logger.run_dir),
                                protocol.name,
                                repeat_index + 1,
                                segment.label,
                            )
                            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                                return
                            frame_index += 1

                    teleop.set_command(0.0, 0.0)
                    settle_end = time.monotonic() + config.collection.settle_time_s
                    while time.monotonic() < settle_end:
                        frame, board_state = tracker.process_next_frame()
                        command_state = teleop.snapshot()
                        logger.log_sample(
                            frame_index, time.monotonic(), board_state, command_state
                        )
                        for event in teleop.pop_events():
                            logger.log_command(event)
                        frame_index += 1
                finally:
                    logger.close()
        print()
    finally:
        teleop.stop()
        tracker.close()
        cv2.destroyAllWindows()
