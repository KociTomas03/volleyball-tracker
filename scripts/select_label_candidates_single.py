"""Rank dense frames from a single clip for ball-labeling - a single-clip counterpart
to select_label_candidates.py, reusing its motion-scoring/FP-hotspot/random-negative
selection functions instead of duplicating them. select_label_candidates.py's own
CLIPS-keyed stratification/interleaving is built for balancing several clips at fixed
per-clip quotas; that doesn't apply when selecting from just one new clip, so this is a
plain single-clip CLI instead of bending that one to fit.

Usage:
    python scripts/select_label_candidates_single.py \\
        --frames-dir data/frames_dense/SLAP_SVIT_1z_upr \\
        --scan-csv data/annotations/ball_dense_scan_slap.csv
"""

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from select_label_candidates import (
    list_frames,
    motion_scores,
    select_known_fp_negatives,
    select_positives,
    select_random_negatives,
)

SEED = 0
MIN_SPACING = 3  # frames apart in the dense pool


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-dir", required=True, type=Path,
                         help="Directory of dense frames for one clip, e.g. data/frames_dense/SLAP_SVIT_1z_upr")
    parser.add_argument("--clip-name", default=None, help="Defaults to --frames-dir's basename")
    parser.add_argument("--scan-csv", required=True, type=Path,
                         help="Output of scan_ball_dense.py for this clip (used for FP-hotspot negatives)")
    parser.add_argument("--positives", type=int, default=150)
    parser.add_argument("--fp-negatives", type=int, default=80)
    parser.add_argument("--random-negatives", type=int, default=70)
    parser.add_argument("--min-spacing", type=int, default=MIN_SPACING)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    clip = args.clip_name or args.frames_dir.name
    out_csv = args.out_csv or Path("data/annotations") / f"label_candidates_{clip}.csv"

    frames = list_frames(args.frames_dir.parent, clip)
    print(f"{len(frames)} dense frames for {clip}")
    scores = motion_scores(frames)
    name_to_idx = {f.name: i for i, f in enumerate(frames)}

    pos_idx = select_positives(scores, args.positives, args.min_spacing)
    pos_frame_ids = {f"{clip}/{frames[i].name}" for i in pos_idx}

    fp_neg_ids = select_known_fp_negatives(args.scan_csv, clip, args.fp_negatives)
    fp_neg_ids = [fid for fid in fp_neg_ids if fid not in pos_frame_ids][: args.fp_negatives]

    exclude_idx = set(pos_idx)
    for fid in fp_neg_ids:
        name = fid.split("/")[1]
        if name in name_to_idx:
            exclude_idx.add(name_to_idx[name])

    rng = random.Random(SEED)
    rand_neg_idx = select_random_negatives(frames, scores, args.random_negatives, exclude_idx, rng)
    rand_neg_ids = [f"{clip}/{frames[i].name}" for i in rand_neg_idx]

    rows = []
    for i in pos_idx:
        rows.append({"frame_id": f"{clip}/{frames[i].name}", "image_path": str(frames[i]),
                      "clip": clip, "motion_score": round(scores[i], 3), "selection_reason": "motion_candidate"})
    for fid in fp_neg_ids:
        rows.append({"frame_id": fid, "image_path": str(args.frames_dir / fid.split("/")[1]),
                      "clip": clip, "motion_score": "", "selection_reason": "known_fp_negative"})
    for fid in rand_neg_ids:
        idx = name_to_idx[fid.split("/")[1]]
        rows.append({"frame_id": fid, "image_path": str(args.frames_dir / fid.split("/")[1]),
                      "clip": clip, "motion_score": round(scores[idx], 3), "selection_reason": "random_negative"})

    # stratified split per selection_reason, same convention as select_label_candidates.py
    rng2 = random.Random(SEED)
    by_reason: dict[str, list[dict]] = {}
    for r in rows:
        by_reason.setdefault(r["selection_reason"], []).append(r)
    for grows in by_reason.values():
        rng2.shuffle(grows)
        n_val = max(1, round(len(grows) * args.val_frac))
        for i, r in enumerate(grows):
            r["split"] = "val" if i < n_val else "train"

    final_rows = by_reason.get("motion_candidate", []) + by_reason.get("known_fp_negative", []) + \
        by_reason.get("random_negative", [])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "image_path", "clip", "motion_score", "selection_reason", "split"])
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"{len(final_rows)} candidates written to {out_csv}")
    print("by reason:", Counter(r["selection_reason"] for r in final_rows))
    print("by split:", Counter(r["split"] for r in final_rows))


if __name__ == "__main__":
    main()
