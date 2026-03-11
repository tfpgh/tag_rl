import time

import cv2
import numpy as np
from pupil_apriltags import Detector

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# MJPG format
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # pyright: ignore[reportAttributeAccessIssue]
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Manual exposure
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, 17)

detector = Detector(families="tagStandard41h12", nthreads=7, quad_decimate=2.0)

# Mat dimensions (mm)
TAG_CENTER_W = 2338.4
TAG_CENTER_H = 1119.2
TAG_SIZE = 100.0
MAT_W = TAG_CENTER_W + TAG_SIZE  # full outer edge
MAT_H = TAG_CENTER_H + TAG_SIZE

# Output: mat fills frame with 1% buffer, stats bar on top
FRAME_W = 1920
STATS_H = 35
BUFFER_PX = int(0.01 * FRAME_W)
mat_w_px = FRAME_W - 2 * BUFFER_PX
px_per_mm = mat_w_px / MAT_W
mat_h_px = int(MAT_H * px_per_mm)
FRAME_H = STATS_H + BUFFER_PX + mat_h_px + BUFFER_PX

# Tag centers are inset 50mm (half a tag) from the mat edge
inset = int(TAG_SIZE / 2 * px_per_mm)
x0 = BUFFER_PX + inset
y0 = STATS_H + BUFFER_PX + inset
x1 = BUFFER_PX + mat_w_px - inset
y1 = STATS_H + BUFFER_PX + mat_h_px - inset

DST_POINTS = np.array(
    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
    dtype=np.float32,
)

CALIBRATION_FRAMES = 100

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

prev_time = time.time()
warp_matrix = None
calibration_samples: dict[int, list[np.ndarray]] = {i: [] for i in range(4)}


COLOR = (180, 255, 180)

# Direction from tag center toward mat center for each corner tag
INWARD_DIR = {0: (1, 1), 1: (-1, 1), 2: (-1, -1), 3: (1, -1)}


def draw_detections(frame: np.ndarray, detections: list) -> None:
    inside_pts: dict[int, tuple[int, int]] = {}

    for det in detections:
        corners = det.corners.astype(int)
        for i in range(4):
            cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), COLOR, 2)
        cx, cy = det.center[0], det.center[1]
        cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        if det.tag_id in INWARD_DIR:
            half = TAG_SIZE / 2 * px_per_mm
            dx, dy = INWARD_DIR[det.tag_id]
            inside_pts[det.tag_id] = (int(cx + dx * half), int(cy + dy * half))

    # Draw red rectangle between inside corners of the 4 tags
    if len(inside_pts) == 4:
        pts = [inside_pts[i] for i in range(4)]
        for i in range(4):
            cv2.line(frame, pts[i], pts[(i + 1) % 4], (0, 0, 255), 2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
    prev_time = now

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    draw_detections(frame, detections)

    h, w = frame.shape[:2]

    # Calibration: collect corner tag positions
    if warp_matrix is None:
        for det in detections:
            if (
                det.tag_id in calibration_samples
                and len(calibration_samples[det.tag_id]) < CALIBRATION_FRAMES
            ):
                calibration_samples[det.tag_id].append(
                    np.array(det.center, dtype=np.float32)
                )
        min_count = min(len(s) for s in calibration_samples.values())
        cv2.putText(
            frame,
            f"Calibrating: {min_count}/{CALIBRATION_FRAMES}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (180, 255, 180),
            2,
        )
        if min_count >= CALIBRATION_FRAMES:
            src = np.array(
                [np.mean(calibration_samples[i], axis=0) for i in range(4)],
                dtype=np.float32,
            )
            warp_matrix = cv2.getPerspectiveTransform(src, DST_POINTS)

    if warp_matrix is not None:
        frame = cv2.warpPerspective(frame, warp_matrix, (FRAME_W, FRAME_H))

    # Stats on top (drawn after warp so text isn't distorted)
    stats = (
        f"FPS: {fps:.1f}  Tags: {len(detections)}  {w}x{h}  {1 / px_per_mm:.2f}mm/px"
    )
    cv2.putText(
        frame, stats, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 180), 2
    )

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
