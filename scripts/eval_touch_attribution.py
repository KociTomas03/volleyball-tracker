"""Score scripts/derive_stats.py's touch detection/attribution against a
manually-reviewed ground-truth CSV (see data/annotations/touch_ground_truth_<clip>.csv).

Ground truth so far is built by reviewing the pipeline's OWN candidate touch list
frame-by-frame (see the Phase 5 replan's Phase 1c) - full frame-level context
(ball + all player boxes) rendered per touch, eyeballed against the real video
action. This validates contact precision (is each pipeline-flagged "contact"
real?) and attribution accuracy (is the credited player right, among contacts
the pipeline actually flagged?) with real confidence.

It does NOT yet fully validate contact recall for touches the pipeline never
flagged as a candidate at all: a real contact with zero row in the pipeline's
own touches list can still be caught here (see contacts_false_negative below,
computed from ground-truth rows missing from the pipeline's current output),
but the ground-truth CSV itself was built by reviewing the pipeline's candidate
list, not by independently re-scrubbing each rally's raw trajectory for
contacts find_ball_contacts structurally never considered. A genuinely
complete recall check would need that independent pass. Treat
contacts_false_negative as a lower bound on missed contacts, not a ceiling.

Usage:
    python scripts/eval_touch_attribution.py \
        --stats data/stats/SLAP_SVIT_1z_upr_validation5min_v3_stats.json \
        --ground-truth data/annotations/touch_ground_truth_SLAP_SVIT_1z_upr.csv
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GroundTruthRow:
    frame_idx: int
    is_real_touch: bool
    correct_track_id: int | None


def load_ground_truth(path: Path) -> dict[int, GroundTruthRow]:
    rows: dict[int, GroundTruthRow] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            frame_idx = int(row["frame_idx"])
            correct_track_id = row["correct_track_id"].strip()
            rows[frame_idx] = GroundTruthRow(
                frame_idx=frame_idx,
                is_real_touch=row["is_real_touch"].strip().lower() == "true",
                correct_track_id=int(correct_track_id) if correct_track_id else None,
            )
    return rows


@dataclass
class EvalResult:
    contacts_true_positive: int  # ground truth real, pipeline flagged a touch there
    contacts_false_positive: int  # ground truth not-real, pipeline flagged a touch there anyway
    contacts_false_negative: int  # ground truth real, pipeline has no touch there at all
    contacts_out_of_scope: int  # pipeline touch frame not covered by ground truth - skipped, not scored
    attribution_correct: int
    attribution_wrong: int
    attribution_none: int  # contact correctly flagged, but pipeline attributed no player
    attribution_unknown: int  # contact correctly flagged, but ground truth has no pinned correct_track_id to check against

    @property
    def contact_precision(self) -> float | None:
        total = self.contacts_true_positive + self.contacts_false_positive
        return self.contacts_true_positive / total if total else None

    @property
    def contact_recall(self) -> float | None:
        total = self.contacts_true_positive + self.contacts_false_negative
        return self.contacts_true_positive / total if total else None

    @property
    def attribution_accuracy(self) -> float | None:
        """Among true-positive contacts with a known correct answer."""
        scored = self.attribution_correct + self.attribution_wrong + self.attribution_none
        return self.attribution_correct / scored if scored else None


def compare_touches(pipeline_touches: list[dict], ground_truth: dict[int, GroundTruthRow]) -> EvalResult:
    """Scores a pipeline run's touches (list of dicts with "frame_idx" and
    "track_id" keys, matching derive_stats.py's Touch/JSON shape) against
    ground_truth rows. Only frames ground_truth actually covers are scored -
    pipeline touches outside the ground-truth-reviewed rallies are silently
    skipped (contacts_out_of_scope), since they were never reviewed and
    counting them either way would be a guess, not a measurement."""
    tp = fp = out_of_scope = 0
    correct = wrong = none_attributed = unknown = 0
    scored_frames: set[int] = set()

    for touch in pipeline_touches:
        frame_idx = touch["frame_idx"]
        gt = ground_truth.get(frame_idx)
        if gt is None:
            out_of_scope += 1
            continue
        scored_frames.add(frame_idx)

        if not gt.is_real_touch:
            fp += 1
            continue

        tp += 1
        track_id = touch.get("track_id")
        if track_id is None:
            none_attributed += 1
        elif gt.correct_track_id is None:
            unknown += 1
        elif track_id == gt.correct_track_id:
            correct += 1
        else:
            wrong += 1

    fn = sum(
        1 for frame_idx, gt in ground_truth.items()
        if gt.is_real_touch and frame_idx not in scored_frames
    )

    return EvalResult(
        contacts_true_positive=tp,
        contacts_false_positive=fp,
        contacts_false_negative=fn,
        contacts_out_of_scope=out_of_scope,
        attribution_correct=correct,
        attribution_wrong=wrong,
        attribution_none=none_attributed,
        attribution_unknown=unknown,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    args = parser.parse_args()

    stats = json.loads(args.stats.read_text())
    ground_truth = load_ground_truth(args.ground_truth)
    result = compare_touches(stats["touches"], ground_truth)

    print(f"contacts: TP={result.contacts_true_positive} FP={result.contacts_false_positive} "
          f"FN={result.contacts_false_negative} out_of_scope={result.contacts_out_of_scope}")
    precision = result.contact_precision
    recall = result.contact_recall
    print(f"contact precision: {precision:.1%}" if precision is not None else "contact precision: n/a")
    print(f"contact recall (lower bound - see module docstring): {recall:.1%}" if recall is not None else "contact recall: n/a")

    print(f"attribution: correct={result.attribution_correct} wrong={result.attribution_wrong} "
          f"none={result.attribution_none} unknown={result.attribution_unknown}")
    accuracy = result.attribution_accuracy
    print(f"attribution accuracy: {accuracy:.1%}" if accuracy is not None else "attribution accuracy: n/a")


if __name__ == "__main__":
    main()
