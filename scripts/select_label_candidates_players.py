"""Round-1 player-label candidate selection: rank frames by player-box overlap/
clustering density instead of generic pixel-diff or motion scoring.

Rationale (see the Phase 5 replan's Phase E): a diagnostic against real
touch-attribution ground truth found the player detector/tracker (currently
untouched pretrained `models/yolov8n.pt`) has two concrete failure patterns
- (1) the true toucher sometimes isn't tracked at all at the contact frame,
and (2) general box quality/separation matters most exactly where several
players cluster close together (net play, defensive scrambles) - the same
situations that also drive touch-attribution errors, so cleaner player data
has value there too even though box-vs-hand representation is the primary
fix for that specific problem (see Phase F). Both failure patterns cluster
in frames with several raw player detections close together, so that's the
targeting signal here - reuses the same schema/tooling (`label_ball.py` with
`--dataset-root`, `select_positives`) as the ball candidate selectors.

Usage:
    python scripts/select_label_candidates_players.py \
        --video data/real_fotage/SLAP_SVIT_1z_upr.mp4 \
        --detections-csv data/annotations/video_detections/SLAP_SVIT_1z_upr_validation5min_segv3.csv
"""

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import cv2

from select_label_candidates import select_positives
from track_video import PLAYER_CONF, load_detections_csv

SEED = 0


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clustering_scores(per_frame: dict[int, dict[str, list]], frame_range: range,
                       min_conf: float = PLAYER_CONF) -> dict[int, float]:
    """Sum of pairwise IoU among confident player boxes in each frame - high
    where several players' boxes overlap/crowd together (net play, scrambles,
    the situations behind both diagnosed player-tracking failure patterns),
    near zero for frames with well-separated players."""
    scores: dict[int, float] = {}
    for f in frame_range:
        boxes = [b[:4] for b in per_frame.get(f, {"player": []})["player"] if b[4] >= min_conf]
        if len(boxes) < 2:
            scores[f] = 0.0
            continue
        total = 0.0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                total += iou(boxes[i], boxes[j])
        scores[f] = total
    return scores


def select_low_score_negatives(scores: dict[int, float], quota: int, min_spacing: int,
                                exclude: set[int], rng: random.Random) -> list[int]:
    frames = sorted(scores)
    ranked = sorted(frames, key=lambda f: scores[f])
    low_half = [f for f in ranked[: max(quota * 3, len(ranked) // 2)] if f not in exclude]
    rng.shuffle(low_half)
    chosen: list[int] = []
    for f in low_half:
        if all(abs(f - g) >= min_spacing for g in chosen):
            chosen.append(f)
        if len(chosen) >= quota:
            break
    return sorted(chosen)


def extract_frame(video_path: Path, frame_idx: int, out_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx} from {video_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--detections-csv", required=True, type=Path)
    parser.add_argument("--clip-name", default=None, help="Defaults to --video's stem")
    parser.add_argument("--frame-start", type=int, default=12000)
    parser.add_argument("--frame-count", type=int, default=9000)
    parser.add_argument("--positives", type=int, default=300)
    parser.add_argument("--random-negatives", type=int, default=60)
    parser.add_argument("--min-spacing", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--frames-out-dir", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    clip = args.clip_name or args.video.stem
    frames_out_dir = args.frames_out_dir or Path("data/frames_dense") / f"{clip}_players1"
    out_csv = args.out_csv or Path("data/annotations") / f"label_candidates_{clip}_players1.csv"

    per_frame = load_detections_csv(args.detections_csv)
    frame_range = range(args.frame_start, args.frame_start + args.frame_count)

    scores = clustering_scores(per_frame, frame_range)
    print(f"{sum(1 for v in scores.values() if v > 0)} frames with overlapping player boxes")

    frames_sorted = sorted(scores)
    score_list = [scores[f] for f in frames_sorted]
    pos_local_idx = select_positives(score_list, args.positives, args.min_spacing)
    pos_frames = [frames_sorted[i] for i in pos_local_idx]
    print(f"{len(pos_frames)} positive (high-clustering) candidates selected "
          f"(score range {scores[pos_frames[-1]]:.2f}-{scores[pos_frames[0]]:.2f})")

    rng = random.Random(SEED)
    neg_frames = select_low_score_negatives(scores, args.random_negatives, args.min_spacing,
                                             set(pos_frames), rng)
    print(f"{len(neg_frames)} random (low-clustering) negative candidates selected")

    rows = []
    for f in pos_frames:
        rows.append({"frame_id": f"{clip}/frame_{f:05d}.jpg", "frame_idx": f, "clip": clip,
                     "motion_score": round(scores[f], 3), "selection_reason": "clustering_candidate"})
    for f in neg_frames:
        rows.append({"frame_id": f"{clip}/frame_{f:05d}.jpg", "frame_idx": f, "clip": clip,
                     "motion_score": round(scores[f], 3), "selection_reason": "random_negative"})

    print(f"extracting {len(rows)} frames to {frames_out_dir}...")
    for i, r in enumerate(rows):
        image_path = frames_out_dir / f"frame_{r['frame_idx']:05d}.jpg"
        extract_frame(args.video, r["frame_idx"], image_path)
        r["image_path"] = str(image_path)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}")

    rng2 = random.Random(SEED)
    by_reason: dict[str, list[dict]] = {}
    for r in rows:
        by_reason.setdefault(r["selection_reason"], []).append(r)
    for grows in by_reason.values():
        rng2.shuffle(grows)
        n_val = max(1, round(len(grows) * args.val_frac))
        for i, r in enumerate(grows):
            r["split"] = "val" if i < n_val else "train"

    final_rows = by_reason.get("clustering_candidate", []) + by_reason.get("random_negative", [])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "image_path", "clip", "motion_score",
                                                "selection_reason", "split"])
        writer.writeheader()
        for r in final_rows:
            writer.writerow({k: r[k] for k in ["frame_id", "image_path", "clip", "motion_score",
                                                "selection_reason", "split"]})

    print(f"\n{len(final_rows)} candidates written to {out_csv}")
    print("by reason:", Counter(r["selection_reason"] for r in final_rows))
    print("by split:", Counter(r["split"] for r in final_rows))


if __name__ == "__main__":
    main()
