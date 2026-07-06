"""Rank dense frames for manual ball-labeling, using motion scoring for
likely rally moments plus known false-positive locations (from a prior
ball_dense_scan.csv) as targeted negatives.

Usage:
    python scripts/select_label_candidates.py
"""

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

CLIPS = ["online_match_01", "online_match_02", "online_match_03", "online_match_04"]
ROI_Y_FRAC = 0.36  # exclude the crowd/stands band that dominated false positives
MIN_SPACING = 3  # frames apart (~1.1s at this dense sampling rate)
SEED = 0

POSITIVES_PER_CLIP = {"online_match_01": 88, "online_match_02": 88, "online_match_03": 87, "online_match_04": 87}
KNOWN_FP_NEG_PER_CLIP = {"online_match_01": 40, "online_match_02": 14, "online_match_03": 13, "online_match_04": 13}
RANDOM_NEG_PER_CLIP = {"online_match_01": 18, "online_match_02": 18, "online_match_03": 17, "online_match_04": 17}

# --- round 2: scoreboard-change detection (targets genuine mid-rally frames) ---
SCOREBOARD_CROP = (100, 35, 460, 115)  # x1, y1, x2, y2 - covers score digits + SET POINT/TIME OUT banner
EVENT_TOP_K_PER_CLIP = 12
EVENT_MIN_SPACING = 10  # frames (~3.7s), so one graphic transition isn't picked twice
PRE_EVENT_OFFSETS = [1, 3, 5, 7]  # frames before each detected change (~0.4-3s before)
ROUND2_QUOTA_PER_CLIP = {"online_match_01": 38, "online_match_02": 35, "online_match_03": 35, "online_match_04": 35}


def list_frames(frames_dir: Path, clip: str) -> list[Path]:
    return sorted((frames_dir / clip).glob("frame_*.jpg"))


def motion_scores(frames: list[Path]) -> list[float]:
    raw = [0.0] * len(frames)
    prev = None
    for i, f in enumerate(frames):
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        roi = img[int(h * ROI_Y_FRAC):, :]
        small = cv2.resize(roi, (320, max(1, int(320 * roi.shape[0] / roi.shape[1]))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            raw[i] = float(np.mean(np.abs(gray - prev)))
        prev = gray

    scores = [0.0] * len(raw)
    for i in range(len(raw)):
        lo, hi = max(0, i - 1), min(len(raw), i + 2)
        scores[i] = sum(raw[lo:hi]) / (hi - lo)
    return scores


def select_positives(scores: list[float], quota: int, min_spacing: int) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    selected = []
    for i in order:
        if all(abs(i - j) >= min_spacing for j in selected):
            selected.append(i)
        if len(selected) >= quota:
            break
    return sorted(selected)


def load_fp_hotspots(scan_csv: Path, clip: str):
    rows = [r for r in csv.DictReader(open(scan_csv)) if r["frame_id"].startswith(clip + "/")]
    bucket_counter = Counter()
    bucket_frames = defaultdict(set)
    for r in rows:
        cx = (float(r["x1"]) + float(r["x2"])) / 2
        cy = (float(r["y1"]) + float(r["y2"])) / 2
        key = (round(cx / 40) * 40, round(cy / 40) * 40)
        bucket_counter[key] += 1
        bucket_frames[key].add(r["frame_id"])
    return bucket_counter, bucket_frames


def select_known_fp_negatives(scan_csv: Path, clip: str, quota: int) -> list[str]:
    bucket_counter, bucket_frames = load_fp_hotspots(scan_csv, clip)
    chosen, seen = [], set()
    for bucket, _ in bucket_counter.most_common():
        for fid in sorted(bucket_frames[bucket]):
            if fid in seen:
                continue
            seen.add(fid)
            chosen.append(fid)
            if len(chosen) >= quota:
                return chosen
    return chosen


def select_random_negatives(frames, scores, quota, exclude_idx, rng) -> list[int]:
    candidates = [i for i in range(len(frames)) if i not in exclude_idx]
    candidates.sort(key=lambda i: scores[i])
    low_half = candidates[: max(quota * 3, len(candidates) // 2)]
    rng.shuffle(low_half)
    return low_half[:quota]


def scoreboard_change_scores(frames: list[Path]) -> list[float]:
    x1, y1, x2, y2 = SCOREBOARD_CROP
    raw = [0.0] * len(frames)
    prev = None
    for i, f in enumerate(frames):
        img = cv2.imread(str(f))
        crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            raw[i] = float(np.mean(np.abs(gray - prev)))
        prev = gray
    return raw


def select_top_k_ranked(scores: list[float], quota: int, min_spacing: int) -> list[int]:
    """Like select_positives, but returns indices in descending-score priority order."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    selected = []
    for i in order:
        if all(abs(i - j) >= min_spacing for j in selected):
            selected.append(i)
        if len(selected) >= quota:
            break
    return selected


def probe_events(frames_dir: Path, top_n: int = 4):
    """Print the top-N scoreboard-change events per clip, for a sanity check
    before generating the full round-2 manifest."""
    for clip in CLIPS:
        frames = list_frames(frames_dir, clip)
        scores = scoreboard_change_scores(frames)
        event_idx = select_top_k_ranked(scores, top_n, EVENT_MIN_SPACING)
        print(f"--- {clip} top {top_n} scoreboard-change events ---")
        for idx in event_idx:
            print(f"  frame_{idx:04d}.jpg  score={scores[idx]:.2f}  "
                  f"(preceding: frame_{max(0, idx - 1):04d}.jpg)")


def build_round2(frames_dir: Path, existing_manifest: Path, out_csv: Path):
    existing_ids = set()
    if existing_manifest.exists():
        existing_ids = {r["frame_id"] for r in csv.DictReader(open(existing_manifest))}

    rng = random.Random(SEED)
    rows = []
    for clip in CLIPS:
        frames = list_frames(frames_dir, clip)
        scores = scoreboard_change_scores(frames)
        event_idx = select_top_k_ranked(scores, EVENT_TOP_K_PER_CLIP, EVENT_MIN_SPACING)

        quota = ROUND2_QUOTA_PER_CLIP[clip]
        chosen_idx, seen = [], set()
        for k in event_idx:
            if len(chosen_idx) >= quota:
                break
            for off in PRE_EVENT_OFFSETS:
                idx = k - off
                if idx < 0 or idx in seen:
                    continue
                fid = f"{clip}/{frames[idx].name}"
                if fid in existing_ids:
                    continue
                seen.add(idx)
                chosen_idx.append(idx)
                if len(chosen_idx) >= quota:
                    break

        for idx in chosen_idx:
            rows.append({
                "frame_id": f"{clip}/{frames[idx].name}", "image_path": str(frames[idx]),
                "clip": clip, "motion_score": "", "selection_reason": "pre_score_change",
            })

    by_clip = defaultdict(list)
    for r in rows:
        by_clip[r["clip"]].append(r)
    for grows in by_clip.values():
        rng.shuffle(grows)
        n_val = max(1, round(len(grows) * 0.15))
        for i, r in enumerate(grows):
            r["split"] = "val" if i < n_val else "train"

    final_rows = []
    by_clip_ordered = defaultdict(list)
    for r in rows:
        by_clip_ordered[r["clip"]].append(r)
    while any(by_clip_ordered.values()):
        for clip in CLIPS:
            if by_clip_ordered[clip]:
                final_rows.append(by_clip_ordered[clip].pop(0))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "image_path", "clip", "motion_score", "selection_reason", "split"])
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"{len(final_rows)} round-2 candidates written to {out_csv}")
    print("by clip:", Counter(r["clip"] for r in final_rows))
    print("by split:", Counter(r["split"] for r in final_rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, default=Path("data/frames_dense"))
    parser.add_argument("--scan-csv", type=Path, default=Path("data/annotations/ball_dense_scan.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("data/annotations/label_candidates.csv"))
    parser.add_argument("--probe-events", action="store_true", help="Print top scoreboard-change events per clip and exit")
    parser.add_argument("--round2", action="store_true", help="Build the pre-score-change round-2 manifest instead")
    parser.add_argument("--round2-out-csv", type=Path, default=Path("data/annotations/label_candidates_round2.csv"))
    args = parser.parse_args()

    if args.probe_events:
        probe_events(args.frames_dir)
        return

    if args.round2:
        build_round2(args.frames_dir, args.out_csv, args.round2_out_csv)
        return

    rng = random.Random(SEED)
    rows = []

    for clip in CLIPS:
        frames = list_frames(args.frames_dir, clip)
        scores = motion_scores(frames)
        name_to_idx = {f.name: i for i, f in enumerate(frames)}

        pos_idx = select_positives(scores, POSITIVES_PER_CLIP[clip], MIN_SPACING)
        pos_frame_ids = {f"{clip}/{frames[i].name}" for i in pos_idx}

        fp_neg_ids = select_known_fp_negatives(args.scan_csv, clip, KNOWN_FP_NEG_PER_CLIP[clip])
        fp_neg_ids = [fid for fid in fp_neg_ids if fid not in pos_frame_ids][: KNOWN_FP_NEG_PER_CLIP[clip]]

        exclude_idx = set(pos_idx)
        for fid in fp_neg_ids:
            name = fid.split("/")[1]
            if name in name_to_idx:
                exclude_idx.add(name_to_idx[name])

        rand_neg_idx = select_random_negatives(frames, scores, RANDOM_NEG_PER_CLIP[clip], exclude_idx, rng)
        rand_neg_ids = [f"{clip}/{frames[i].name}" for i in rand_neg_idx]

        for i in pos_idx:
            rows.append({
                "frame_id": f"{clip}/{frames[i].name}", "image_path": str(frames[i]),
                "clip": clip, "motion_score": round(scores[i], 3), "selection_reason": "motion_candidate",
            })
        for fid in fp_neg_ids:
            rows.append({
                "frame_id": fid, "image_path": str(args.frames_dir / fid),
                "clip": clip, "motion_score": "", "selection_reason": "known_fp_negative",
            })
        for fid in rand_neg_ids:
            idx = name_to_idx[fid.split("/")[1]]
            rows.append({
                "frame_id": fid, "image_path": str(args.frames_dir / fid),
                "clip": clip, "motion_score": round(scores[idx], 3), "selection_reason": "random_negative",
            })

    # stratified 85/15 train/val split, per (clip, reason) group
    by_group = defaultdict(list)
    for r in rows:
        by_group[(r["clip"], r["selection_reason"])].append(r)
    for grows in by_group.values():
        rng.shuffle(grows)
        n_val = max(1, round(len(grows) * 0.15))
        for i, r in enumerate(grows):
            r["split"] = "val" if i < n_val else "train"

    def emit_order(reason):
        by_clip = defaultdict(list)
        for r in rows:
            if r["selection_reason"] == reason:
                by_clip[r["clip"]].append(r)
        interleaved = []
        while any(by_clip.values()):
            for clip in CLIPS:
                if by_clip[clip]:
                    interleaved.append(by_clip[clip].pop(0))
        return interleaved

    final_rows = emit_order("motion_candidate") + emit_order("known_fp_negative") + emit_order("random_negative")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "image_path", "clip", "motion_score", "selection_reason", "split"])
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"{len(final_rows)} candidates written to {args.out_csv}")
    print("by reason:", Counter(r["selection_reason"] for r in final_rows))
    print("by split:", Counter(r["split"] for r in final_rows))


if __name__ == "__main__":
    main()
