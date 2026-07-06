"""Run detect_frame over a set of frames and build a manual precision/recall
sense-check worksheet.

Usage:
    python scripts/run_sense_check.py
        Runs detection over all data/frames/*/frame_*.jpg, saves annotated
        previews to data/annotations/detect_review/, and writes
        data/annotations/phase2_sense_check.csv pre-filled with predicted
        box counts (TP/FP/FN columns left blank for manual review).

    python scripts/run_sense_check.py --summarize
        Reads the (hand-filled) CSV and prints precision/recall for both
        classes.
"""

import argparse
import csv
from pathlib import Path

from detect_frame import detect, save_annotated

CSV_PATH = Path("data/annotations/phase2_sense_check.csv")
REVIEW_DIR = Path("data/annotations/detect_review")
FIELDS = [
    "frame_id", "image_path",
    "player_boxes_predicted", "ball_boxes_predicted",
    "player_TP", "player_FP", "player_FN",
    "ball_TP", "ball_FP", "ball_FN",
    "notes",
]


def run(frames_dir: Path):
    frames = sorted(frames_dir.glob("*/frame_*.jpg"))
    rows = []
    for f in frames:
        detections = detect(str(f))
        save_annotated(str(f), detections, REVIEW_DIR)
        player_count = sum(1 for d in detections if d["class"] == "player")
        ball_count = sum(1 for d in detections if d["class"] == "ball")
        rows.append({
            "frame_id": f"{f.parent.name}/{f.name}",
            "image_path": str(f),
            "player_boxes_predicted": player_count,
            "ball_boxes_predicted": ball_count,
            "player_TP": "", "player_FP": "", "player_FN": "",
            "ball_TP": "", "ball_FP": "", "ball_FN": "",
            "notes": "",
        })

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} frames processed. Annotated previews in {REVIEW_DIR}/")
    print(f"Sense-check worksheet written to {CSV_PATH}")
    print("Open each annotated image, compare to ground truth by eye, and fill in the TP/FP/FN columns.")


def summarize():
    with open(CSV_PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))

    def totals(prefix):
        tp = sum(int(r[f"{prefix}_TP"] or 0) for r in rows)
        fp = sum(int(r[f"{prefix}_FP"] or 0) for r in rows)
        fn = sum(int(r[f"{prefix}_FN"] or 0) for r in rows)
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        return tp, fp, fn, precision, recall

    for cls in ("player", "ball"):
        tp, fp, fn, precision, recall = totals(cls)
        print(f"{cls}: TP={tp} FP={fp} FN={fn}  precision={precision:.3f}  recall={recall:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, default=Path("data/frames"))
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    if args.summarize:
        summarize()
    else:
        run(args.frames_dir)


if __name__ == "__main__":
    main()
