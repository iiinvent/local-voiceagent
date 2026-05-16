"""
Background webcam capture for the voice agent (OpenCV).

Env: VISION_CAMERA_DEVICE, VISION_CAPTURE_INTERVAL_SEC, VISION_MAX_WIDTH,
VISION_JPEG_QUALITY, VISION_STARTUP_SEC, VISION_CAMERA_WARMUP_FRAMES.
"""

from __future__ import annotations

import base64
import os
import sys
import threading
import time
from typing import Optional

_lock = threading.Lock()
_latest_jpeg: Optional[bytes] = None
_capture_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_camera_ok = False
_camera_error: Optional[str] = None

VISION_CAMERA_DEVICE = int(os.environ.get("VISION_CAMERA_DEVICE", "0"))
VISION_CAPTURE_INTERVAL_SEC = max(
    0.1, float(os.environ.get("VISION_CAPTURE_INTERVAL_SEC", "0.5"))
)
VISION_MAX_WIDTH = max(160, int(os.environ.get("VISION_MAX_WIDTH", "640")))
VISION_JPEG_QUALITY = max(30, min(95, int(os.environ.get("VISION_JPEG_QUALITY", "75"))))
VISION_STARTUP_SEC = max(1.0, float(os.environ.get("VISION_STARTUP_SEC", "8")))
VISION_CAMERA_WARMUP_FRAMES = max(0, int(os.environ.get("VISION_CAMERA_WARMUP_FRAMES", "8")))


def _opencv_backend() -> int:
    import cv2

    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def _resize_and_encode(frame) -> Optional[bytes]:
    import cv2

    h, w = frame.shape[:2]
    if w > VISION_MAX_WIDTH:
        scale = VISION_MAX_WIDTH / float(w)
        frame = cv2.resize(
            frame,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), VISION_JPEG_QUALITY]
    )
    if not ok:
        return None
    return buf.tobytes()


def _open_capture():
    import cv2

    backend = _opencv_backend()
    cap = cv2.VideoCapture(VISION_CAMERA_DEVICE, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(VISION_CAMERA_DEVICE)
    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    return cap


def _capture_loop() -> None:
    global _latest_jpeg, _camera_ok, _camera_error
    cap = None
    try:
        cap = _open_capture()
        if cap is None or not cap.isOpened():
            _camera_error = (
                f"could not open camera index {VISION_CAMERA_DEVICE} "
                f"(try VISION_CAMERA_DEVICE=1 or grant Camera access to Terminal)"
            )
            _camera_ok = False
            return

        _camera_ok = True
        _camera_error = None

        for _ in range(VISION_CAMERA_WARMUP_FRAMES):
            if _stop_event.is_set():
                return
            cap.read()

        while not _stop_event.is_set():
            ok, frame = cap.read()
            if ok and frame is not None:
                jpeg = _resize_and_encode(frame)
                if jpeg:
                    with _lock:
                        _latest_jpeg = jpeg
            _stop_event.wait(VISION_CAPTURE_INTERVAL_SEC)
    except Exception as exc:
        _camera_error = str(exc)
        _camera_ok = False
    finally:
        if cap is not None:
            cap.release()


def _wait_for_first_frame() -> bool:
    deadline = time.monotonic() + VISION_STARTUP_SEC
    while time.monotonic() < deadline:
        if _camera_error:
            return False
        with _lock:
            if _latest_jpeg:
                return True
        time.sleep(0.05)
    with _lock:
        return _latest_jpeg is not None


def start_camera() -> bool:
    """Start background capture thread. Returns True once a frame is available."""
    global _capture_thread, _stop_event, _latest_jpeg, _camera_ok, _camera_error

    if _capture_thread is not None and _capture_thread.is_alive():
        if camera_ready():
            return True
        return _wait_for_first_frame()

    _stop_event = threading.Event()
    _latest_jpeg = None
    _camera_ok = False
    _camera_error = None
    _capture_thread = threading.Thread(target=_capture_loop, daemon=True)
    _capture_thread.start()
    return _wait_for_first_frame()


def ensure_camera() -> bool:
    """Retry camera start (e.g. after startup failure or when user asks to see)."""
    if camera_ready():
        return True
    stop_camera()
    time.sleep(0.2)
    return start_camera()


def stop_camera() -> None:
    """Stop background capture."""
    global _capture_thread, _camera_ok
    _stop_event.set()
    if _capture_thread is not None:
        _capture_thread.join(timeout=5.0)
        _capture_thread = None
    _camera_ok = False


def camera_ready() -> bool:
    with _lock:
        return _latest_jpeg is not None


def camera_error() -> Optional[str]:
    return _camera_error


def capture_snapshot() -> Optional[str]:
    """Return base64 JPEG of the latest frame, or None if unavailable."""
    with _lock:
        data = _latest_jpeg
    if not data:
        return None
    return base64.standard_b64encode(data).decode("ascii")
