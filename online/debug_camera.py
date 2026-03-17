from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug camera capture outside the web app"
    )
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--buffersize", type=int, default=1)
    parser.add_argument("--exposure", type=int, default=17)
    parser.add_argument(
        "--backend",
        choices=["auto", "v4l2", "avfoundation"],
        default="v4l2" if sys.platform.startswith("linux") else "auto",
        help="OpenCV camera backend",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("debug_camera_frame.jpg"),
        help="Where to save a sample frame",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="How many frames to attempt before exiting",
    )
    return parser


def resolve_backend(name: str) -> int | None:
    if name == "auto":
        return None
    if name == "v4l2":
        return getattr(cv2, "CAP_V4L2", None)
    if name == "avfoundation":
        return getattr(cv2, "CAP_AVFOUNDATION", None)
    return None


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    backend = resolve_backend(args.backend)
    if backend is None:
        cap = cv2.VideoCapture(args.device)
    else:
        cap = cv2.VideoCapture(args.device, backend)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera device {args.device} with backend {args.backend}"
        )
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, args.buffersize)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
    return cap


def print_capture_properties(cap: cv2.VideoCapture) -> None:
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    actual_backend = int(cap.get(cv2.CAP_PROP_BACKEND))
    print(f"backend={actual_backend}")
    print(f"size={actual_width}x{actual_height}")
    print(f"fps={actual_fps:.2f}")
    print(f"exposure={actual_exposure:.2f}")


def main() -> None:
    args = build_parser().parse_args()
    cap = open_capture(args)
    print_capture_properties(cap)

    saved = False
    read_count = 0
    start = time.time()
    last = start
    try:
        for index in range(args.frames):
            ok, frame = cap.read()
            now = time.time()
            dt = now - last
            last = now
            if not ok:
                print(f"frame {index}: read failed")
                time.sleep(0.05)
                continue
            read_count += 1
            fps = 0.0 if dt <= 0 else 1.0 / dt
            print(
                f"frame {index}: ok shape={frame.shape[1]}x{frame.shape[0]} read_fps={fps:.1f}"
            )
            if not saved:
                ok_write = cv2.imwrite(str(args.output), frame)
                print(f"saved_sample={ok_write} path={args.output}")
                saved = ok_write
        total = time.time() - start
        print(f"reads_ok={read_count}/{args.frames} elapsed={total:.2f}s")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
