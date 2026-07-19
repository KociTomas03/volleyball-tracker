"""Score touch attribution against ground truth by POSITION instead of raw track_id.

eval_touch_attribution.py compares `touch["track_id"] == ground_truth.correct_track_id`
directly - but track IDs are per-run bookkeeping, not a stable per-player identifier.
Found 2026-07-19: re-evaluating touch attribution after retraining the player detector
came back 0% attribution accuracy, which looked like a severe regression until a
control run (old player model, unchanged otherwise) also scored 0% - proving the cause
was scripts/track_video.py's tracker upgrade (BoT-SORT+Re-ID replacing ByteTrack,
commit f51b3ec) making current track IDs incomparable to the IDs the ground truth CSV
was built against, regardless of detector quality.

This script fixes the comparison to be tracker-version-agnostic: it reconstructs the
ORIGINAL run's per-frame player boxes using legacy_bytetrack_player_tracker.py (a
frozen snapshot of the pre-f51b3ec tracker) against the same historical detections CSV
the ground truth was built from, looks up each ground-truth row's correct_track_id in
that reconstruction to get a real pixel-space reference box (not an ID), then compares
the CURRENT run's attributed track's box (reconstructed the same way, using whichever
tracker scripts/track_video.py currently ships) against that reference via IoU. Two
different tracker versions/detector models can now be compared fairly - IN PRINCIPLE.

KNOWN LIMITATION (found 2026-07-19, unresolved): reconstructing the legacy tracker
against the true historical detections (from the actual frame-0 start the original run
used - frame-range matters a lot for ByteTrack's ID-assignment continuity) still does
not reproduce the original run's track IDs - it now creates far MORE tracks (700+ by
frame 12089) than the original apparently had (~31), the opposite direction from the
too-few-tracks result of starting reconstruction mid-stream. The `ByteTrack` class in
the installed `supervision` version is flagged deprecated, so this is most likely
library-version drift in ByteTrack's own internals since the ground truth was built,
not something fixable by adjusting this script's frame range or parameters. Don't trust
reference_box output from this script until that's independently resolved (e.g. by
pinning the historical supervision version, or building fresh ground truth whose
reference is captured as a position directly rather than a track_id, which sidesteps
this problem entirely). This script's position-based COMPARISON logic is still sound
and worth keeping for that future case.

Usage:
    python scripts/eval_touch_attribution_by_position.py \
        --stats <path to a derive_stats.py output json> \
        --ground-truth data/annotations/touch_review_SLAP_SVIT_1z_upr.csv \
        --current-detections-csv data/annotations/video_detections/SLAP_SVIT_1z_upr_newplayer_v1.csv \
        --current-video data/real_fotage/SLAP_SVIT_1z_upr.mp4
"""

import argparse
import json
from pathlib import Path

import cv2

from eval_touch_attribution import load_ground_truth
from legacy_bytetrack_player_tracker import track_players_bytetrack
from track_video import load_detections_csv, track_players

DEFAULT_HISTORICAL_DETECTIONS_CSV = Path("data/annotations/video_detections/SLAP_SVIT_1z_upr_validation5min_segv3.csv")
DEFAULT_HISTORICAL_VIDEO = Path("data/real_fotage/SLAP_SVIT_1z_upr.mp4")
DEFAULT_IOU_THRESH = 0.3


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


def box_for_track(frame_players: dict, frame_idx: int, track_id: int) -> tuple[float, float, float, float] | None:
    dets = frame_players.get(frame_idx)
    if dets is None:
        return None
    for box, tid in zip(dets.xyxy, dets.tracker_id):
        if int(tid) == track_id:
            return tuple(float(v) for v in box)
    return None


def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def frame_range_of(per_frame: dict) -> range:
    """Track from each source's OWN earliest cached frame, not the ground truth's
    earliest touch frame - a tracker's ID assignment depends on every frame since
    its actual start, so starting mid-stream (skipping real detections the source
    cache does have, just before the ground truth's window) reassigns every ID
    from a different, incomparable baseline. Found by getting a near-zero match
    rate reconstructing from the ground truth's min touch frame instead of this."""
    frames = sorted(per_frame)
    return range(frames[0], frames[-1] + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--current-detections-csv", required=True, type=Path)
    parser.add_argument("--current-video", required=True, type=Path)
    parser.add_argument("--historical-detections-csv", type=Path, default=DEFAULT_HISTORICAL_DETECTIONS_CSV)
    parser.add_argument("--historical-video", type=Path, default=DEFAULT_HISTORICAL_VIDEO)
    parser.add_argument("--iou-thresh", type=float, default=DEFAULT_IOU_THRESH)
    args = parser.parse_args()

    stats = json.loads(args.stats.read_text())
    ground_truth = load_ground_truth(args.ground_truth)
    gt_frames = [f for f, row in ground_truth.items() if row.is_real_touch and row.correct_track_id is not None]

    historical_fps = get_fps(args.historical_video)
    historical_per_frame = load_detections_csv(args.historical_detections_csv)
    historical_frame_range = frame_range_of(historical_per_frame)
    print(f"reconstructing historical (ByteTrack) positions over frames "
          f"{historical_frame_range.start}-{historical_frame_range.stop - 1} (the cache's own full span)...")
    historical_frame_players = track_players_bytetrack(historical_per_frame, historical_frame_range, historical_fps)

    reference_box: dict[int, tuple[float, float, float, float]] = {}
    missing_reference = 0
    for frame_idx in gt_frames:
        box = box_for_track(historical_frame_players, frame_idx, ground_truth[frame_idx].correct_track_id)
        if box is None:
            missing_reference += 1
            continue
        reference_box[frame_idx] = box
    print(f"{len(reference_box)}/{len(gt_frames)} ground-truth rows got a reconstructed reference box "
          f"({missing_reference} missing - historical reconstruction didn't find that track_id at that frame)")

    current_fps = get_fps(args.current_video)
    current_per_frame = load_detections_csv(args.current_detections_csv)
    current_frame_range = frame_range_of(current_per_frame)
    print(f"reconstructing current (BoT-SORT) positions over frames "
          f"{current_frame_range.start}-{current_frame_range.stop - 1} (the cache's own full span)...")
    current_frame_players = track_players(current_per_frame, current_frame_range, current_fps, args.current_video)

    correct = wrong = none_attributed = unknown = 0
    tp = fp = 0
    scored_frames: set[int] = set()
    for touch in stats["touches"]:
        frame_idx = touch["frame_idx"]
        gt = ground_truth.get(frame_idx)
        if gt is None:
            continue
        scored_frames.add(frame_idx)
        if not gt.is_real_touch:
            fp += 1
            continue
        tp += 1

        track_id = touch.get("track_id")
        if track_id is None:
            none_attributed += 1
            continue
        ref = reference_box.get(frame_idx)
        if ref is None:
            unknown += 1
            continue
        attributed_box = box_for_track(current_frame_players, frame_idx, track_id)
        if attributed_box is None:
            wrong += 1
            continue
        if iou(attributed_box, ref) >= args.iou_thresh:
            correct += 1
        else:
            wrong += 1

    fn = sum(1 for frame_idx, gt in ground_truth.items() if gt.is_real_touch and frame_idx not in scored_frames)

    scored = correct + wrong + none_attributed
    print(f"contacts scored: TP={tp} FP={fp} FN={fn}")
    print(f"attribution (position-based, IoU>={args.iou_thresh}): correct={correct} wrong={wrong} "
          f"none={none_attributed} unknown={unknown}")
    print(f"attribution accuracy: {correct / scored:.1%}" if scored else "attribution accuracy: n/a")


if __name__ == "__main__":
    main()
