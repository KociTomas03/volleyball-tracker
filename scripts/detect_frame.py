"""Run player + ball detection on a single frame.

Usage:
    python scripts/detect_frame.py --image data/frames/online_match_01/frame_0003.jpg [--save-annotated]
"""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

PLAYER_MODEL_PATH = "models/yolov8n.pt"
BALL_MODEL_PATH = "models/ball_yolov8n_v4.pt"
PLAYER_CONF = 0.4
BALL_CONF = 0.25
COCO_PERSON_CLASS = 0

# Ultralytics' `model()` defaults to an internal conf=0.25 cutoff during inference/NMS,
# applied before our own min_conf filtering ever sees the results. Without passing conf=
# explicitly, that hidden default silently discards anything below 0.25 - including the
# 0.1-0.25 band detect_video.py's --min-conf cache floor (and ByteTrack's hardcoded 0.1
# low-confidence second-association tier) both depend on. Keep this well below 0.1 so
# nothing in that band is clipped before it reaches our filtering/caching.
DETECT_INTERNAL_CONF = 0.05

# Ball is small (~15-20px at full 1280x720 res) - the default imgsz=640 halves source
# resolution and shrinks it further. Only bump this for the ball model; players are
# already large/easy targets and don't need the extra inference cost.
BALL_IMGSZ = 1280

_player_model = None
_ball_model = None


def _get_player_model() -> YOLO:
    global _player_model
    if _player_model is None:
        _player_model = YOLO(PLAYER_MODEL_PATH)
    return _player_model


def _get_ball_model() -> YOLO:
    global _ball_model
    if _ball_model is None:
        _ball_model = YOLO(BALL_MODEL_PATH)
    return _ball_model


def detect_players(image_path: str, min_conf: float = PLAYER_CONF) -> list[dict]:
    model = _get_player_model()
    results = model(image_path, conf=DETECT_INTERNAL_CONF, verbose=False)[0]
    out = []
    for box in results.boxes:
        if int(box.cls[0]) != COCO_PERSON_CLASS:
            continue
        conf = float(box.conf[0])
        if conf < min_conf:
            continue
        out.append({
            "class": "player",
            "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
            "confidence": round(conf, 3),
        })
    return out


def detect_ball(image_path: str, min_conf: float = BALL_CONF) -> list[dict]:
    model = _get_ball_model()
    results = model(image_path, conf=DETECT_INTERNAL_CONF, imgsz=BALL_IMGSZ, verbose=False)[0]
    out = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < min_conf:
            continue
        out.append({
            "class": "ball",
            "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
            "confidence": round(conf, 3),
        })
    return out


def detect(image_path: str) -> list[dict]:
    return detect_players(image_path) + detect_ball(image_path)


def save_annotated(image_path: str, detections: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(image_path)
    colors = {"player": (0, 200, 0), "ball": (0, 0, 255)}
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        color = colors.get(d["class"], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class']} {d['confidence']:.2f}"
        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    image_path_obj = Path(image_path)
    out_path = out_dir / f"{image_path_obj.parent.name}_{image_path_obj.name}"
    cv2.imwrite(str(out_path), img)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--out-dir", default="data/annotations/detect_review")
    args = parser.parse_args()

    detections = detect(args.image)
    print(json.dumps(detections, indent=2))

    if args.save_annotated:
        out_path = save_annotated(args.image, detections, Path(args.out_dir))
        print(f"annotated image saved to {out_path}")


if __name__ == "__main__":
    main()
