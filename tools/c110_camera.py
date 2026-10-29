"""Independent C110 photo capture and optional detection operations.

Importing this module never connects to either the camera or detection service.
"""
from pathlib import Path
import sys


def capture_photo(path, camera_folder, timeout=20):
    """Capture one fresh frame and save it; never call a detection service."""
    import cv2

    folder = str(Path(camera_folder).resolve())
    if folder not in sys.path:
        sys.path.insert(0, folder)
    from camera_frame_capture import C110FrameCapture

    frames = C110FrameCapture().capture_frames(count=1, timeout=timeout)
    if len(frames) != 1:
        raise RuntimeError("Camera did not return exactly one frame")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frames[0]):
        raise OSError(f"Could not save captured frame to {path}")
    return path


def detect_photo(path, detection_url, timeout=60):
    """Explicitly submit an already saved photo and return its positions mapping."""
    import requests

    path = Path(path)
    with path.open("rb") as stream:
        response = requests.post(
            detection_url, files={"image": (path.name, stream, "image/jpeg")},
            timeout=timeout,
        )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("positions"), dict):
        raise ValueError("Detection response has no positions mapping")
    return data["positions"]
