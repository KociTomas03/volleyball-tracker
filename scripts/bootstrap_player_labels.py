"""Bootstrap player labels for a candidate manifest with minimal manual effort.

Targets the low-clustering "negative" candidates from
select_label_candidates_players.py - these are exactly the frames where players are
well-separated, i.e. where the stock COCO player detector is already known to be
reliable. For each candidate frame, in order:
  1. If the local player detector finds at least one high-confidence box, accept the
     full set of high-confidence boxes directly - zero human/API cost. (A lower-
     confidence long tail of junk detections - crowd/stands clutter on this wide-shot
     footage - is common even on otherwise-clean frames and is not itself a signal of
     a missed player, so it isn't used to force escalation the way it might be on a
     tighter/closer shot.)
  2. Otherwise (no confident detections at all) ask Gemini
     (gemini_verify.verify_player_frames) to independently count and box every
     player, and accept its answer if confident and internally consistent.
  3. Anything still unresolved is left unlabeled - label_ball.py's own `todo`
     filtering already picks these up for a manual pass.

The dense clustering-positive candidates are NOT good bootstrap targets (that's the
whole point of selecting them - they're where the detector is weakest): a frame can
easily have one or more confident boxes AND still be missing a clustered/occluded
player the detector didn't find at all, so "has a confident box" does not imply "has
every player." By default this script only processes rows with
selection_reason=random_negative for exactly that reason - clustering candidates are
left untouched (still zero) for manual review in label_ball.py. Pass
--include-clustering-candidates to override this (not recommended - only for cases
where you've separately confirmed the local detector's recall is trustworthy on that
manifest).

Output lands in the same {dataset-root}/{split}/{images,labels}/ layout label_ball.py
writes to (reuses its path/save helpers directly), so train_player.py picks it up with
no changes.

Usage:
    python scripts/bootstrap_player_labels.py \
        --manifest data/annotations/label_candidates_SLAP_SVIT_1z_upr_players1.csv \
        --dataset-root data/self_labeled_players
"""

import argparse
import csv
from pathlib import Path

import cv2

from detect_frame import detect_players
from gemini_verify import DEFAULT_MODEL, verify_player_frames
from label_ball import label_path_for, save_frame

PLAYER_DATASET_ROOT = Path("data/self_labeled_players")
SAFE_SELECTION_REASONS = {"random_negative"}

# Stricter than production PLAYER_CONF (0.4): this box set becomes a ground-truth
# training label with no human review, so it needs to be a set we'd trust
# unconditionally, not just one that clears the normal inference bar.
LOCAL_HIGH_CONF = 0.6
GEMINI_TRUSTED_CONFIDENCE = {"high", "medium"}


def image_size(path: str) -> tuple[int, int]:
    img = cv2.imread(path)
    h, w = img.shape[:2]
    return w, h


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=PLAYER_DATASET_ROOT)
    parser.add_argument("--local-high-conf", type=float, default=LOCAL_HIGH_CONF)
    parser.add_argument("--gemini-concurrency", type=int, default=8)
    parser.add_argument("--gemini-model", default=None, help="Passed through to gemini_verify.py (its own default if unset)")
    parser.add_argument("--skip-gemini", action="store_true",
                         help="Skip tier 2 entirely - useful when the Gemini key/quota is known to be unavailable")
    parser.add_argument("--include-clustering-candidates", action="store_true",
                         help="Also bootstrap selection_reason=clustering_candidate rows (NOT recommended - "
                              "see module docstring for why this risks baking incomplete ground truth into "
                              "exactly the frames this dataset most needs to be correct on)")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))
    if not args.include_clustering_candidates:
        skipped = [r for r in rows if r["selection_reason"] not in SAFE_SELECTION_REASONS]
        if skipped:
            print(f"skipping {len(skipped)} clustering-candidate rows (bootstrap only handles "
                  f"selection_reason={sorted(SAFE_SELECTION_REASONS)} by default) - label these "
                  f"manually via label_ball.py instead")
        rows = [r for r in rows if r["selection_reason"] in SAFE_SELECTION_REASONS]
    todo = [r for r in rows if not label_path_for(r, args.dataset_root).exists()]
    print(f"{len(rows)} eligible candidates, {len(rows) - len(todo)} already labeled, {len(todo)} remaining")

    tier1, needs_gemini = [], []
    for row in todo:
        confident = detect_players(row["image_path"], min_conf=args.local_high_conf)
        if confident:
            tier1.append((row, [tuple(h["bbox"]) for h in confident]))
        else:
            needs_gemini.append(row)
    print(f"tier 1 (local detector, high confidence): {len(tier1)}")

    for row, boxes in tier1:
        w, h = image_size(row["image_path"])
        save_frame(row, boxes, w, h, args.dataset_root)

    tier2_count = tier3 = 0
    if needs_gemini and args.skip_gemini:
        tier3 = len(needs_gemini)
    elif needs_gemini:
        model = args.gemini_model or DEFAULT_MODEL
        print(f"asking Gemini ({model}) about {len(needs_gemini)} frames the local detector had borderline boxes on...")
        try:
            results = verify_player_frames([r["image_path"] for r in needs_gemini], model=model,
                                            concurrency=args.gemini_concurrency)
        except Exception as exc:
            # A batch-wide failure (e.g. quota exhaustion) shouldn't lose the tier-1
            # work already saved above - degrade to "needs manual review" instead of
            # crashing, and surface the error so it's obvious this needs attention.
            print(f"Gemini batch failed ({exc!r}) - leaving all {len(needs_gemini)} of these frames "
                  f"for manual review in label_ball.py instead.")
            results = {}
        for row in needs_gemini:
            result = results.get(row["image_path"])
            confidence = result.get("confidence") if result else None
            boxes = result.get("boxes") if result else None
            count = result.get("player_count") if result else None
            if not result or confidence not in GEMINI_TRUSTED_CONFIDENCE or boxes is None or count != len(boxes):
                tier3 += 1
                continue
            w, h = image_size(row["image_path"])
            save_frame(row, [tuple(b) for b in boxes], w, h, args.dataset_root)
            tier2_count += 1

    print(f"tier 2 (Gemini, confident and consistent): {tier2_count}")
    print(f"tier 3 (unresolved, needs manual review in label_ball.py): {tier3}")
    print(f"total auto-labeled this run: {len(tier1) + tier2_count}")


if __name__ == "__main__":
    main()
