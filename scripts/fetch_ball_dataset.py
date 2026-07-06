"""Discover and evaluate public Roboflow volleyball-ball detection projects.

Usage:
    python scripts/fetch_ball_dataset.py --probe
        List candidate projects and whether each has a hosted model.

    python scripts/fetch_ball_dataset.py --zero-shot workspace/project/version --frames data/frames
        Run a candidate's hosted model against our frames, save annotated previews.

    python scripts/fetch_ball_dataset.py --download workspace/project/version
        Download a dataset (for local fine-tuning) into data/roboflow_ball/.
"""

import argparse
import os
from pathlib import Path

import cv2
from dotenv import load_dotenv
from roboflow import Roboflow

CANDIDATES = [
    ("primaryws", "volleyball_ball_object_detection_dataset"),
    ("aivolleyballref", "volleyball_detection"),
    ("volleyballtracking-gouow", "volleyball-tracking-7ovg1"),
]


def get_client() -> Roboflow:
    load_dotenv()
    return Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])


def probe():
    rf = get_client()
    for workspace, project in CANDIDATES:
        try:
            proj = rf.workspace(workspace).project(project)
            versions = proj.versions()
            latest = versions[0] if versions else None
            has_model = bool(latest and getattr(latest, "model", None))
            print(f"{workspace}/{project}: OK, versions={len(versions)}, "
                  f"latest={latest.version if latest else None}, hosted_model={has_model}")
        except Exception as e:
            print(f"{workspace}/{project}: FAILED ({e})")


def zero_shot(slug: str, frames_dir: Path, out_dir: Path, confidence: int = 40, overlap: int = 30):
    workspace, project, version = slug.split("/")
    rf = get_client()
    ver = rf.workspace(workspace).project(project).version(int(version))
    if not getattr(ver, "model", None):
        print(f"{slug}: no hosted model available for zero-shot inference")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(Path(frames_dir).glob("*/frame_*.jpg"))
    total_hits = 0
    for f in frames:
        preds = ver.model.predict(str(f), confidence=confidence, overlap=overlap).json()
        boxes = preds.get("predictions", [])
        total_hits += len(boxes)

        img = cv2.imread(str(f))
        for b in boxes:
            x1 = int(b["x"] - b["width"] / 2)
            y1 = int(b["y"] - b["height"] / 2)
            x2 = int(b["x"] + b["width"] / 2)
            y2 = int(b["y"] + b["height"] / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{b['class']} {b['confidence']:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        safe_slug = slug.replace("/", "_")
        cv2.imwrite(str(out_dir / f"{safe_slug}_{f.parent.name}_{f.name}"), img)

    print(f"{slug}: {total_hits} ball boxes across {len(frames)} frames -> previews in {out_dir}")


def download(slug: str, out_dir: Path, fmt: str = "yolov8"):
    workspace, project, version = slug.split("/")
    rf = get_client()
    ver = rf.workspace(workspace).project(project).version(int(version))
    ver.download(fmt, location=str(out_dir / slug.replace("/", "_")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--zero-shot", metavar="WORKSPACE/PROJECT/VERSION")
    parser.add_argument("--download", metavar="WORKSPACE/PROJECT/VERSION")
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/annotations/ball_zero_shot"))
    args = parser.parse_args()

    if args.probe:
        probe()
    elif args.zero_shot:
        zero_shot(args.zero_shot, args.frames, args.out_dir)
    elif args.download:
        download(args.download, Path("data/roboflow_ball"))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
