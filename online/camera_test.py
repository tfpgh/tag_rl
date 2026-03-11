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
    nthreads=6,
)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
    prev_time = now

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    for det in detections:
        corners = det.corners.astype(int)

        # Draw quad outline
        for i in range(4):
            cv2.line(
                frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2
            )

        # Center
        cx, cy = int(det.center[0]), int(det.center[1])
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Size (average of two diagonal lengths in pixels)
        diag1 = np.linalg.norm(corners[2] - corners[0])
        diag2 = np.linalg.norm(corners[3] - corners[1])
        size = (diag1 + diag2) / 2.0

        # 2D rotation from homography
        H = det.homography
        angle = math.degrees(math.atan2(H[1, 0], H[0, 0]))

        # Label
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

    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}  Tags: {len(detections)}",
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
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
