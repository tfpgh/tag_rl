import time

import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# MJPG format
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Manual exposure
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, 10)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
    prev_time = now

    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
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
