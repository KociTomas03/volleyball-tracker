"""Bootstrap ball labels for a candidate manifest with minimal manual effort.

For each candidate frame, in order:
  1. If the current local ball detector already finds it with high confidence,
     accept that box directly - zero human/API cost, and consistent with what the
     model already "knows" (see LOCAL_HIGH_CONF below for why this bar is stricter
     than production BALL_CONF).
  2. Otherwise ask Gemini (gemini_verify.py, already built for exactly this "local
     detector unsure" case) and accept its answer if it's confident either way.
  3. Anything still unresolved (Gemini confidence "low", or an error) is left
     unlabeled - label_ball.py's own `todo` filtering already picks up exactly these
     frames for a manual pass, no changes needed there.

Output lands in the same data/self_labeled/{split}/{images,labels}/ layout
label_ball.py writes to (reuses its path/save helpers directly), so train_ball.py
picks it up with no changes.

Usage:
    python scripts/bootstrap_ball_labels.py --manifest data/annotations/label_candidates_SLAP_SVIT_1z_upr.csv
"""

import argparse
import csv
from pathlib import Path

import cv2

from detect_frame import detect_ball
from gemini_verify import DEFAULT_MODEL, verify_ball_frames
from label_ball import label_path_for, save_frame

# Stricter than production BALL_CONF (0.25): this box becomes a ground-truth training
# label with no human review, so it needs to be a box we'd trust unconditionally, not
# just one that clears the normal inference bar.
LOCAL_HIGH_CONF = 0.5
GEMINI_TRUSTED_CONFIDENCE = {"high", "medium"}


def image_size(path: str) -> tuple[int, int]:
    img = cv2.imread(path)
    h, w = img.shape[:2]
    return w, h


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--local-high-conf", type=float, default=LOCAL_HIGH_CONF)
    parser.add_argument("--gemini-concurrency", type=int, default=8)
    parser.add_argument("--gemini-model", default=None, help="Passed through to gemini_verify.py (its own default if unset)")
    parser.add_argument("--skip-gemini", action="store_true",
                         help="Skip tier 2 entirely - useful when the Gemini key/quota is known to be unavailable")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))
    todo = [r for r in rows if not label_path_for(r).exists()]
    print(f"{len(rows)} total candidates, {len(rows) - len(todo)} already labeled, {len(todo)} remaining")

    tier1, needs_gemini = [], []
    for row in todo:
        hits = detect_ball(row["image_path"], min_conf=args.local_high_conf)
        if hits:
            best = max(hits, key=lambda d: d["confidence"])
            tier1.append((row, [tuple(best["bbox"])]))
        else:
            needs_gemini.append(row)
    print(f"tier 1 (local detector, high confidence): {len(tier1)}")

    for row, boxes in tier1:
        w, h = image_size(row["image_path"])
        save_frame(row, boxes, w, h)

    tier2_positive = tier2_negative = tier3 = 0
    if needs_gemini and args.skip_gemini:
        tier3 = len(needs_gemini)
    elif needs_gemini:
        model = args.gemini_model or DEFAULT_MODEL
        print(f"asking Gemini ({model}) about {len(needs_gemini)} frames the local detector wasn't confident on...")
        try:
            results = verify_ball_frames([r["image_path"] for r in needs_gemini], model=model,
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
            if not result or confidence not in GEMINI_TRUSTED_CONFIDENCE:
                tier3 += 1
                continue
            w, h = image_size(row["image_path"])
            if result.get("ball_visible") and result.get("bbox"):
                save_frame(row, [tuple(result["bbox"])], w, h)
                tier2_positive += 1
            elif not result.get("ball_visible"):
                save_frame(row, [], w, h)
                tier2_negative += 1
            else:
                tier3 += 1  # ball_visible=true but no usable bbox - can't trust without one

    print(f"tier 2 (Gemini, confident): {tier2_positive} ball, {tier2_negative} no-ball")
    print(f"tier 3 (unresolved, needs manual review in label_ball.py): {tier3}")
    print(f"total auto-labeled this run: {len(tier1) + tier2_positive + tier2_negative}")


if __name__ == "__main__":
    main()
