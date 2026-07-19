"""Run the ball detector across a dense frame set and log every detection.

Unlike run_sense_check.py, this does NOT save an annotated image per frame
(too slow/large at hundreds-thousands of frames) - it just records which
frames got a hit, for later targeted review.

Usage:
    python scripts/scan_ball_dense.py --frames-dir data/frames_dense
    python scripts/scan_ball_dense.py --frames-dir data/frames_dense/LANSKROUN1
    python scripts/scan_ball_dense.py --model runs/detect/runs/ball_finetune/v2_self_labeled/weights/best.pt \\
        --exclude-images-dir data/self_labeled --out-csv data/annotations/ball_dense_scan_v2.csv
"""

import argparse
import csv
from pathlib import Path

from detect_frame import BALL_CONF, BALL_MODEL_PATH
from ultralytics import YOLO


def load_excluded_names(exclude_images_dir: Path) -> set[str]:
    """Frame basenames already used as training/val images - excluded from the
    scan so re-evaluation stays out-of-sample (avoids data leakage)."""
    if not exclude_images_dir:
        return set()
    return {p.name for p in exclude_images_dir.glob("*/images/*.jpg")}


def iter_frames(frames_dir: Path) -> list[tuple[str, Path]]:
    """Yields (clip_name, path) pairs. Supports both a single clip directory
    (frame_*.jpg directly inside, e.g. data/frames_dense/LANSKROUN1) and a parent
    directory holding multiple clip subdirectories (the original/default layout,
    e.g. data/frames_dense itself)."""
    direct = sorted(frames_dir.glob("frame_*.jpg"))
    if direct:
        return [(frames_dir.name, f) for f in direct]
    return [(f.parent.name, f) for f in sorted(frames_dir.glob("*/frame_*.jpg"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, default=Path("data/frames_dense"))
    parser.add_argument("--out-csv", type=Path, default=Path("data/annotations/ball_dense_scan.csv"))
    parser.add_argument("--model", type=str, default=BALL_MODEL_PATH,
                         help="Ball model weights to use (default: current production model)")
    parser.add_argument("--exclude-images-dir", type=Path, default=None,
                         help="Skip frames whose basename appears under <dir>/*/images/ (e.g. data/self_labeled)")
    args = parser.parse_args()

    print(f"loading ball model: {args.model}")
    model = YOLO(args.model)

    excluded = load_excluded_names(args.exclude_images_dir) if args.exclude_images_dir else set()
    if excluded:
        print(f"excluding {len(excluded)} frames already used for training/val")

    all_entries = iter_frames(args.frames_dir)
    entries = [(clip, f) for clip, f in all_entries if f"{clip}_{f.name}" not in excluded]
    print(f"scanning {len(entries)} frames (of {len(all_entries)} total)...")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    hit_count = 0
    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_id", "image_path", "confidence", "x1", "y1", "x2", "y2"])
        for i, (clip, f) in enumerate(entries):
            results = model(str(f), verbose=False)[0]
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < BALL_CONF:
                    continue
                x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
                writer.writerow([f"{clip}/{f.name}", str(f), round(conf, 3), x1, y1, x2, y2])
                hit_count += 1
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(entries)} frames scanned, {hit_count} detections so far")

    print(f"done. {hit_count} total detections across {len(entries)} frames -> {args.out_csv}")


if __name__ == "__main__":
    main()
