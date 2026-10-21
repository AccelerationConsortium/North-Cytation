import json
import sys
from pathlib import Path
import cv2
import requests

#Pathing for camera control and output folders + Detection URL

CAMERA_FOLDER = Path(
    r"C:\Users\Imaging Controller\Desktop\camera-control"
)
OUTPUT_FOLDER = Path(
    r"C:\Users\Imaging Controller\Desktop\C110_test_frames"
)
DETECTION_URL = "http://sdl4-not-for-gaming:8001/api/detect"
sys.path.insert(0, str(CAMERA_FOLDER))

from camera_frame_capture import C110FrameCapture

camera = C110FrameCapture()
frames = camera.capture_frames(count=1, timeout=20)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
results = {}

for index, frame in enumerate(frames, start=1):
    frame_path = OUTPUT_FOLDER / f"frame_{index}.jpg"
    if not cv2.imwrite(str(frame_path), frame):
        raise OSError(f"Could not save captured frame to {frame_path}")
    print(f"Saved frame {index}: shape={frame.shape}, path={frame_path}")

    with frame_path.open("rb") as image_file:
        response = requests.post(
            DETECTION_URL,
            files={"image": (frame_path.name, image_file, "image/jpeg")},
            timeout=60,
        )
    response.raise_for_status()
    response_data = response.json()
    if (
        not isinstance(response_data, dict)
        or not isinstance(response_data.get("positions"), dict)
    ):
        raise ValueError(
            f"Detection response for {frame_path.name} has no positions mapping"
        )
    results[frame_path.name] = response_data["positions"]
    print(f"Analyzed frame {index}")

results_path = OUTPUT_FOLDER / "detection_results.json"
results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"Saved detection results: {results_path}")