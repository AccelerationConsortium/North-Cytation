import json
import sys
from pathlib import Path
import argparse

#Pathing for camera control and output folders + Detection URL

CAMERA_FOLDER = Path(
    r"C:\Users\Imaging Controller\Desktop\camera-control"
)
OUTPUT_FOLDER = Path(
    r"C:\Users\Imaging Controller\Desktop\C110_test_frames"
)
DETECTION_URL = "http://sdl4-not-for-gaming:8001/api/detect"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.c110_camera import capture_photo, detect_photo

def main():
    """Capture only by default; --detect explicitly enables the separate action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-folder", type=Path, default=CAMERA_FOLDER)
    parser.add_argument("--output-folder", type=Path, default=OUTPUT_FOLDER)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--detection-url", default=DETECTION_URL)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    results = {}
    for index in range(1, args.count + 1):
        path = capture_photo(args.output_folder / f"frame_{index}.jpg", args.camera_folder)
        print(f"Saved frame: {path}")
        if args.detect:
            results[path.name] = detect_photo(path, args.detection_url)
    if args.detect:
        (args.output_folder / "detection_results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8",
        )


if __name__ == "__main__":
    main()
