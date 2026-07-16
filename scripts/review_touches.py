"""Interactive per-touch review: step through the pipeline's touches for a clip,
watch a short looping clip around each contact with full player/ball overlay, and
flag whether it's real, a false positive, correctly attributed, or click the real
toucher's box if the pipeline got it wrong.

Exists because the earlier Phase 5 ground-truth pass (data/annotations/
touch_ground_truth_<clip>.csv) was built by eyeballing single still frames, which
can't show what motion reveals - e.g. a player mid-dive who never actually reaches
the ball only looks wrong once you watch the ball's path continue unchanged through
the "contact" frame. This tool plays the actual motion instead, and lets the person
who knows what really happened (not a frame-by-frame guess) do the judging.

Output is a CSV in the same schema as touch_ground_truth_<clip>.csv, written
incrementally (one row per decision) so progress survives a quit or crash, and
already-reviewed frames are skipped on a re-run - re-launch anytime to continue
where you left off. To redo a frame, delete its row from --out and re-run.

Usage:
    python scripts/review_touches.py \
        --stats data/stats/SLAP_SVIT_1z_upr_validation5min_v3_stats.json \
        --video data/real_fotage/SLAP_SVIT_1z_upr.mp4 \
        --detections-csv data/annotations/video_detections/SLAP_SVIT_1z_upr_validation5min_segv3.csv \
        --out data/annotations/touch_review_SLAP_SVIT_1z_upr.csv

    # only rally 3 (see the stats JSON's "rallies" list for indices)
    python scripts/review_touches.py --stats ... --video ... --detections-csv ... --out ... --rally 3

Controls (also printed at startup):
    SPACE  - correct: the pipeline's attribution (or its lack of one) is right
    x      - false positive: not a real touch at all
    click  - real touch, WRONG player credited - click the player who really touched it
    u      - real touch, but no player can be reasonably credited even by you
    r      - replay the loop from the start
    p / n  - previous / next touch, without recording a decision (revisit later)
    q      - quit, saving progress so far
"""

import argparse
import csv
import json
from pathlib import Path

import cv2

from track_video import (
    MAX_BALL_GAP_FRAMES,
    interpolate_gaps_ballistic,
    load_detections_csv,
    track_ball_states,
    track_players,
)

REVIEW_WINDOW_HALF = 20  # ~0.7s at ~30fps either side of the contact frame - enough to see the real motion
PLAYBACK_DELAY_MS = 70  # loop playback speed - slower than real-time (~30fps) for easier judgment

FIELDNAMES = [
    "frame_idx", "rally_index", "is_real_touch", "pipeline_track_id", "pipeline_dist_px",
    "attribution_verdict", "correct_track_id", "confidence", "notes",
]


def rally_index_for_frame(frame_idx: int, rallies: list[dict]) -> int | None:
    for i, r in enumerate(rallies):
        if r["start_frame"] <= frame_idx <= r["end_frame"]:
            return i
    return None


def load_reviewed_frames(out_path: Path) -> set[int]:
    if not out_path.exists():
        return set()
    with out_path.open(newline="") as f:
        return {int(row["frame_idx"]) for row in csv.DictReader(f)}


def append_row(out_path: Path, row: dict) -> None:
    is_new = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def box_contains(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def find_clicked_track(point: tuple[float, float],
                        boxes: dict[int, tuple[float, float, float, float]]) -> int | None:
    """Track id of whichever box the click landed inside, preferring the smallest
    (most specific) box when several overlap. None if the click hit no one."""
    hits = [tid for tid, box in boxes.items() if box_contains(point, box)]
    if not hits:
        return None
    return min(hits, key=lambda tid: (boxes[tid][2] - boxes[tid][0]) * (boxes[tid][3] - boxes[tid][1]))


def boxes_for_frame(frame_players: dict, frame_idx: int) -> dict[int, tuple[float, float, float, float]]:
    dets = frame_players.get(frame_idx)
    if dets is None or len(dets) == 0:
        return {}
    return {int(tid): tuple(float(v) for v in box) for box, tid in zip(dets.xyxy, dets.tracker_id)}


def build_row(touch: dict, rally_idx: int | None, verdict: str, correct_track_id: int | None) -> dict:
    is_real = verdict != "false_positive"
    return {
        "frame_idx": touch["frame_idx"],
        "rally_index": rally_idx if rally_idx is not None else "",
        "is_real_touch": "true" if is_real else "false",
        "pipeline_track_id": touch["track_id"] if touch["track_id"] is not None else "",
        "pipeline_dist_px": touch["distance_px"] if touch["distance_px"] is not None else "",
        "attribution_verdict": {"correct": "correct", "false_positive": "na",
                                 "wrong": "wrong", "unattributable": "missed"}[verdict],
        "correct_track_id": correct_track_id if correct_track_id is not None else "",
        "confidence": "high",
        "notes": "user-reviewed (scripts/review_touches.py)",
    }


def draw_frame(frame, frame_idx: int, frame_players: dict, ball_by_frame: dict, touch: dict):
    display = frame.copy()
    boxes = boxes_for_frame(frame_players, frame_idx)
    for tid, box in boxes.items():
        x1, y1, x2, y2 = map(int, box)
        is_credited = tid == touch.get("track_id")
        color = (0, 255, 255) if is_credited else (0, 200, 0)
        thickness = 4 if is_credited else 1
        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(display, str(tid), (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if frame_idx in ball_by_frame:
        cx, cy, _ = ball_by_frame[frame_idx]
        cv2.circle(display, (int(cx), int(cy)), 10, (0, 0, 255), 3)
    dist = touch["distance_px"]
    label = (f"frame {frame_idx}  pipeline: track={touch['track_id']} "
             f"dist={dist:.0f}px" if dist is not None else f"frame {frame_idx}  pipeline: track={touch['track_id']}")
    cv2.putText(display, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return display


def review_loop(video_path: Path, out_path: Path, todo: list[dict], all_touches_count: int,
                 rally_filter: int | None, rallies: list[dict], start_frame: int, end_frame: int,
                 frame_players: dict, ball_by_frame: dict) -> None:
    cap = cv2.VideoCapture(str(video_path))
    window = "review touches (SPACE=correct x=false click=wrong u=unattributable r=replay p/n=skip q=quit)"
    cv2.namedWindow(window)
    state = {"cur_frame": None, "decision": None}

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and state["cur_frame"] is not None:
            boxes = boxes_for_frame(frame_players, state["cur_frame"])
            tid = find_clicked_track((x, y), boxes)
            if tid is not None:
                state["decision"] = ("wrong", tid)

    cv2.setMouseCallback(window, on_click)

    print(f"{len(todo)} touches to review (of {all_touches_count} total"
          + (f" in rally {rally_filter}" if rally_filter is not None else "") + ")")
    print("SPACE=correct  x=false positive  click=wrong (pick correct player)  u=real but unattributable")
    print("r=replay  p/n=prev/next (skip without deciding)  q=quit and save")

    idx = 0
    while 0 <= idx < len(todo):
        touch = todo[idx]
        frame_idx = touch["frame_idx"]
        lo = max(start_frame, frame_idx - REVIEW_WINDOW_HALF)
        hi = min(end_frame, frame_idx + REVIEW_WINDOW_HALF)
        cached = []
        for f in range(lo, hi + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if ok:
                cached.append((f, frame))

        rally_idx = rally_index_for_frame(frame_idx, rallies)
        print(f"[{idx + 1}/{len(todo)}] frame {frame_idx}  t={frame_idx / 29.97:.1f}s  rally={rally_idx}  "
              f"pipeline track={touch['track_id']} dist={touch['distance_px']}")

        state["decision"] = None
        action = None
        while action is None:
            for f, frame in cached:
                state["cur_frame"] = f
                cv2.imshow(window, draw_frame(frame, f, frame_players, ball_by_frame, touch))
                key = cv2.waitKey(PLAYBACK_DELAY_MS) & 0xFF
                if state["decision"] is not None:
                    action = state["decision"]
                    break
                if key == ord(" "):
                    action = ("correct", touch["track_id"])
                elif key == ord("x"):
                    action = ("false_positive", None)
                elif key == ord("u"):
                    action = ("unattributable", None)
                elif key == ord("r"):
                    pass  # falls through, replays from the top of the cached window
                elif key == ord("p"):
                    action = ("skip_prev", None)
                elif key == ord("n"):
                    action = ("skip_next", None)
                elif key == ord("q"):
                    action = ("quit", None)
                if action is not None:
                    break

        if action[0] == "quit":
            break
        if action[0] == "skip_next":
            idx += 1
            continue
        if action[0] == "skip_prev":
            idx = max(0, idx - 1)
            continue

        verdict, correct_tid = action
        row = build_row(touch, rally_idx, verdict, correct_tid)
        append_row(out_path, row)
        print(f"  -> {row['attribution_verdict']} "
              f"(correct_track_id={row['correct_track_id'] if row['correct_track_id'] != '' else 'n/a'})")
        idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"saved progress to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--detections-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rally", type=int, default=None, help="only review touches within this rally index")
    args = parser.parse_args()

    stats = json.loads(args.stats.read_text())
    touches = stats["touches"]
    rallies = stats["rallies"]
    start_frame, end_frame = stats["frame_range"]
    frame_range = range(start_frame, end_frame + 1)

    scoped_touches = touches
    if args.rally is not None:
        r = rallies[args.rally]
        scoped_touches = [t for t in touches if r["start_frame"] <= t["frame_idx"] <= r["end_frame"]]

    already_reviewed = load_reviewed_frames(args.out)
    todo = [t for t in scoped_touches if t["frame_idx"] not in already_reviewed]
    if not todo:
        print("nothing left to review" + (f" for rally {args.rally}" if args.rally is not None else ""))
        return

    print(f"loading detections + tracking for frames {start_frame}-{end_frame} (this can take a few minutes)...")
    per_frame = load_detections_csv(args.detections_csv)
    frame_players = track_players(per_frame, frame_range, stats["fps"], args.video)
    ball_states = track_ball_states(per_frame, frame_range)
    ball_by_frame = interpolate_gaps_ballistic(ball_states, MAX_BALL_GAP_FRAMES)

    review_loop(args.video, args.out, todo, len(scoped_touches), args.rally, rallies,
                start_frame, end_frame, frame_players, ball_by_frame)


if __name__ == "__main__":
    main()
