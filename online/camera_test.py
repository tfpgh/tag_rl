import math
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
cap.set(cv2.CAP_PROP_EXPOSURE, 10)

detector = Detector(
    families="tagStandard41h12",
    nthreads=7,
    quad_decimate=2.0,
)

# Mat dimensions (tag center to tag center, mm)
MAT_W_MM = 2338.4
MAT_H_MM = 1119.2

# Output 1920x1080, fit mat inside with 15% margin so surrounding area is visible
FRAME_W, FRAME_H = 1920, 1080
MARGIN = 0.15
usable_w = FRAME_W * (1 - 2 * MARGIN)
usable_h = FRAME_H * (1 - 2 * MARGIN)
scale = min(usable_w / MAT_W_MM, usable_h / MAT_H_MM)
mat_w_px = MAT_W_MM * scale
mat_h_px = MAT_H_MM * scale
x0 = (FRAME_W - mat_w_px) / 2
y0 = (FRAME_H - mat_h_px) / 2

# Destination points for the 4 corner tags (TL=0, TR=1, BR=2, BL=3)
DST_POINTS = np.array(
    [
        [x0, y0],
        [x0 + mat_w_px, y0],
        [x0 + mat_w_px, y0 + mat_h_px],
        [x0, y0 + mat_h_px],
    ],
    dtype=np.float32,
)

CORNER_TAG_IDS = {0, 1, 2, 3}
CALIBRATION_FRAMES = 10

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Warped", cv2.WINDOW_NORMAL)

prev_time = time.time()
warp_matrix = None
calibration_samples: dict[int, list[np.ndarray]] = {i: [] for i in range(4)}
calibrated = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
    prev_time = now

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Only run detection during calibration
    if not calibrated:
        detections = detector.detect(gray)

        for det in detections:
            corners = det.corners.astype(int)
            for i in range(4):
                cv2.line(
                    frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2
                )

            cx, cy = int(det.center[0]), int(det.center[1])
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            diag1 = np.linalg.norm(corners[2] - corners[0])
            diag2 = np.linalg.norm(corners[3] - corners[1])
            size = (diag1 + diag2) / 2.0

            H = det.homography
            angle = math.degrees(math.atan2(H[1, 0], H[0, 0]))

            label = f"id={det.tag_id} {angle:.0f}deg {size:.0f}px"
            cv2.putText(
                frame,
                label,
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            if det.tag_id in CORNER_TAG_IDS and len(calibration_samples[det.tag_id]) < CALIBRATION_FRAMES:
                calibration_samples[det.tag_id].append(np.array(det.center, dtype=np.float32))

        # Check if we have enough samples for all 4 corners
        counts = {tid: len(samples) for tid, samples in calibration_samples.items()}
        min_count = min(counts.values())

        if min_count >= CALIBRATION_FRAMES:
            src_points = np.array(
                [np.mean(calibration_samples[tid], axis=0) for tid in range(4)],
                dtype=np.float32,
            )
            warp_matrix = cv2.getPerspectiveTransform(src_points, DST_POINTS)
            calibrated = True

        # Show calibration progress
        progress = f"Calibrating: {min_count}/{CALIBRATION_FRAMES}"
    else:
        progress = "Calibrated"

    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}  {progress}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (180, 255, 180),
        2,
    )
    cv2.putText(
        frame, f"{w}x{h}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 255, 180), 2
    )

    cv2.imshow("Camera", frame)

    if warp_matrix is not None:
        warped = cv2.warpPerspective(frame, warp_matrix, (1920, 1080))
        cv2.imshow("Warped", warped)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
