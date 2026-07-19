"""Round 4 ball-label candidate selection: rank frames by ball motion-vector
discontinuity (the raw finite-difference velocity-vector delta on the same
smoothed x/y trajectory find_ball_contacts uses), instead of the generic
pixel-diff scoring select_label_candidates.py's motion_scores() uses.

Rationale (see the Phase 5 replan's "round 4" plan section): 29% of the
current pipeline's flagged "touches" are top-of-arc false positives, and a
Phase A validation found a real (if noisy) separation between real touches
(median delta-v 8.9 px/frame) and false positives (4.0-5.1 px/frame) using
this exact metric. Targeting labelling effort at high-delta-v frames biases
the new training data toward the hardest, highest-value moments (near-
contact, fast/blurry motion) instead of "the ball is mostly just flying,"
matching this project's 3 prior self-labelling rounds' pattern of targeting
a specific diagnosed failure mode rather than sampling generically.

Reuses track_video.py's already-proven tracking/interpolation and
derive_stats.py's smoothing, plus select_label_candidates.py's
select_positives() (a generic list-of-scores/greedy-with-spacing selector,
no clip-specific logic to duplicate) for the min-spacing candidate picking.

Usage:
    python scripts/select_label_candidates_motion_vector.py \
        --video data/real_fotage/SLAP_SVIT_1z_upr.mp4 \
        --detections-csv data/annotations/video_detections/SLAP_SVIT_1z_upr_validation5min_segv3.csv
"""

import argparse
import csv
import math
import random
from collections import Counter
from pathlib import Path

import cv2

from derive_stats import smooth_ball_trajectory
from select_label_candidates import select_positives
from track_video import (
    MAX_BALL_GAP_FRAMES,
    interpolate_gaps_ballistic,
    load_detections_csv,
    track_ball_states,
)

SEED = 0
HALF_WINDOW = 2  # frames either side averaged before/after, matching the validated Phase A metric


def delta_v_scores(ball_by_frame: dict[int, tuple[float, float, bool]],
                    half_window: int = HALF_WINDOW) -> dict[int, float]:
    """Per-frame |v_after - v_before| where v_before/after are the average
    frame-to-frame velocity over half_window frames on each side of the
    frame - smooths single-frame jitter while still localizing to the
    contact frame. Same formula validated in Phase A's vector_change()."""
    xs = smooth_ball_trajectory({f: v[0] for f, v in ball_by_frame.items()})
    ys = smooth_ball_trajectory({f: v[1] for f, v in ball_by_frame.items()})
    frames = sorted(set(xs) & set(ys))

    vel: dict[int, tuple[float, float]] = {}
    for a, b in zip(frames, frames[1:]):
        if b - a != 1:
            continue
        vel[b] = (xs[b] - xs[a], ys[b] - ys[a])

    scores: dict[int, float] = {}
    for f in frames:
        before = [vel[g] for g in range(f - half_window, f) if g in vel]
        after = [vel[g] for g in range(f, f + half_window) if g in vel]
        if not before or not after:
            continue
        vb = (sum(v[0] for v in before) / len(before), sum(v[1] for v in before) / len(before))
        va = (sum(v[0] for v in after) / len(after), sum(v[1] for v in after) / len(after))
        scores[f] = math.hypot(va[0] - vb[0], va[1] - vb[1])
    return scores


def select_low_score_negatives(scores: dict[int, float], quota: int, min_spacing: int,
                                exclude: set[int], rng: random.Random) -> list[int]:
    """Random sample from the lower half of scored frames (calm ball flight),
    for training-set balance against the high-delta-v positives - mirrors
    select_label_candidates.py's select_random_negatives, adapted to work
    directly off this module's frame->score dict instead of a dense-frame
    file list."""
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
    frames_out_dir = args.frames_out_dir or Path("data/frames_dense") / f"{clip}_round4"
    out_csv = args.out_csv or Path("data/annotations") / f"label_candidates_{clip}_round4.csv"

    per_frame = load_detections_csv(args.detections_csv)
    frame_range = range(args.frame_start, args.frame_start + args.frame_count)

    print("running track_ball_states()...")
    states = track_ball_states(per_frame, frame_range)
    ball_by_frame = interpolate_gaps_ballistic(states, MAX_BALL_GAP_FRAMES)
    print(f"{len(ball_by_frame)} ball positions (real + interpolated) in range")

    scores = delta_v_scores(ball_by_frame)
    print(f"{len(scores)} frames scored")

    frames_sorted = sorted(scores)
    score_list = [scores[f] for f in frames_sorted]
    pos_local_idx = select_positives(score_list, args.positives, args.min_spacing)
    pos_frames = [frames_sorted[i] for i in pos_local_idx]
    print(f"{len(pos_frames)} positive candidates selected "
          f"(score range {scores[pos_frames[-1]]:.1f}-{scores[pos_frames[0]]:.1f})")

    rng = random.Random(SEED)
    neg_frames = select_low_score_negatives(scores, args.random_negatives, args.min_spacing,
                                             set(pos_frames), rng)
    print(f"{len(neg_frames)} random negative candidates selected")

    rows = []
    for f in pos_frames:
        rows.append({"frame_id": f"{clip}/frame_{f:05d}.jpg", "frame_idx": f, "clip": clip,
                     "motion_score": round(scores[f], 3), "selection_reason": "motion_vector_candidate"})
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

    # stratified 85/15 train/val split, per selection_reason group - same convention
    # as select_label_candidates.py/select_label_candidates_single.py
    rng2 = random.Random(SEED)
    by_reason: dict[str, list[dict]] = {}
    for r in rows:
        by_reason.setdefault(r["selection_reason"], []).append(r)
    for grows in by_reason.values():
        rng2.shuffle(grows)
        n_val = max(1, round(len(grows) * args.val_frac))
        for i, r in enumerate(grows):
            r["split"] = "val" if i < n_val else "train"

    final_rows = by_reason.get("motion_vector_candidate", []) + by_reason.get("random_negative", [])

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
