from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from pupil_apriltags import Detector

from online.calibration import ArenaCalibrator
from online.config import DemoConfig
from online.runtime_state import RuntimeState
from online.tracking import WorldTracker
from online.types import TagDetection


def _annotate_frame(
    frame: np.ndarray,
    detections: list[TagDetection],
) -> np.ndarray:
    annotated = frame.copy()
    for detection in detections:
        corners = np.asarray(detection.corners_px, dtype=np.int32)
        cv2.polylines(
            annotated, [corners], isClosed=True, color=(120, 255, 160), thickness=2
        )
        center = tuple(int(v) for v in detection.center_px)
        cv2.circle(annotated, center, 4, (0, 0, 255), -1)
    return annotated


def _draw_game_border(frame: np.ndarray, border_points: np.ndarray) -> np.ndarray:
    outline = border_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [outline], isClosed=True, color=(0, 0, 255), thickness=3)
    return frame


class DetectionWorker(threading.Thread):
    def __init__(self, config: DemoConfig, state: RuntimeState) -> None:
        super().__init__(name="detection-worker", daemon=True)
        self.config = config
        self.state = state
        self.detector = Detector(
            families=config.detection.families,
            nthreads=config.detection.nthreads,
            quad_decimate=config.detection.quad_decimate,
            quad_sigma=config.detection.quad_sigma,
            refine_edges=config.detection.refine_edges,
            decode_sharpening=config.detection.decode_sharpening,
        )
        self.calibrator = ArenaCalibrator(
            config.calibration,
            config.env.arena_width,
            config.env.arena_height,
        )
        self.tracker = WorldTracker(config.tracking)
        self._last_detection_ts = 0.0
        self._roi: tuple[int, int, int, int] | None = None
        self._last_frame_id = -1

    def _detect(self, frame: np.ndarray) -> list[TagDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x0 = 0
        y0 = 0
        gray_roi = gray
        if self.config.detection.use_dynamic_roi and self._roi is not None:
            x0, y0, x1, y1 = self._roi
            gray_roi = gray[y0:y1, x0:x1]
        detections = self.detector.detect(gray_roi)
        result: list[TagDetection] = []
        min_x = frame.shape[1]
        min_y = frame.shape[0]
        max_x = 0
        max_y = 0
        for detection in detections:
            corners = detection.corners.astype(np.float32)
            corners[:, 0] += x0
            corners[:, 1] += y0
            center = np.asarray(detection.center, dtype=np.float32)
            center[0] += x0
            center[1] += y0
            min_x = int(min(min_x, corners[:, 0].min()))
            min_y = int(min(min_y, corners[:, 1].min()))
            max_x = int(max(max_x, corners[:, 0].max()))
            max_y = int(max(max_y, corners[:, 1].max()))
            result.append(
                TagDetection(
                    tag_id=int(detection.tag_id),
                    center_px=(float(center[0]), float(center[1])),
                    corners_px=[(float(x), float(y)) for x, y in corners.tolist()],
                    decision_margin=float(detection.decision_margin),
                    hamming=int(detection.hamming),
                )
            )
        if result and self.config.detection.use_dynamic_roi:
            pad = self.config.detection.roi_padding_px
            self._roi = (
                max(0, min_x - pad),
                max(0, min_y - pad),
                min(frame.shape[1], max_x + pad),
                min(frame.shape[0], max_y + pad),
            )
        elif not result:
            self._roi = None
        return result

    def run(self) -> None:
        try:
            while not self.state.stop_event().is_set():
                snap = self.state.snapshot()
                if snap.operator.pause_detection:
                    time.sleep(0.05)
                    continue
                frame, frame_id, timestamp = self.state.get_raw_frame()
                if frame is None or frame_id == 0:
                    time.sleep(0.01)
                    continue
                if frame_id == self._last_frame_id:
                    time.sleep(self.config.detection.latest_only_sleep_s)
                    continue
                self._last_frame_id = frame_id
                loop_start = time.time()
                detections = self._detect(frame)
                calibration = self.calibrator.update(detections)
                now = time.time()
                fps = (
                    0.0
                    if self._last_detection_ts == 0.0
                    else 1.0 / max(1e-6, now - self._last_detection_ts)
                )
                self._last_detection_ts = now
                homography = calibration.homography
                if homography is not None:
                    transform = lambda points: self.calibrator.image_to_world(
                        homography, points
                    )
                    world = self.tracker.update(
                        detections,
                        transform=transform,
                        timestamp=timestamp,
                        frame_id=frame_id,
                        calibration_ready=calibration.state.status
                        in {"ready", "degraded"},
                    )
                else:
                    world = snap.world
                    world.ready = False
                    world.frame_id = frame_id
                    world.timestamp = timestamp
                annotated = _annotate_frame(
                    frame,
                    detections,
                )
                if calibration.display_homography is not None:
                    annotated = self.calibrator.warp_to_board(
                        annotated,
                        calibration.display_homography,
                        calibration.display_size,
                    )
                    annotated = _draw_game_border(
                        annotated,
                        calibration.game_border_points,
                    )
                ok, encoded = cv2.imencode(
                    ".jpg",
                    annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.config.gui.jpeg_quality],
                )
                jpeg_bytes = encoded.tobytes() if ok else None
                self.state.set_annotated_frame(annotated, jpeg=jpeg_bytes)

                def update(snapshot) -> None:  # type: ignore[no-untyped-def]
                    snapshot.detections = detections
                    snapshot.calibration = calibration.state
                    snapshot.world = world
                    snapshot.stats.detection_ms = (time.time() - loop_start) * 1000.0
                    snapshot.stats.detection_fps = fps
                    snapshot.stats.detections = len(detections)
                    snapshot.stats.detection_error = ""

                self.state.mutate_snapshot(update)
                time.sleep(self.config.detection.latest_only_sleep_s)
        except Exception as exc:
            self.state.mutate_snapshot(
                lambda snapshot: setattr(
                    snapshot.stats, "detection_error", f"detection error: {exc}"
                )
            )
