"""Track players and the ball across a clip and render an annotated output video.

Reads cached per-frame detections (from detect_video.py, generated automatically
if missing), tracks players with ByteTrack, tracks the ball with a lightweight
single-object tracker (see track_ball - ByteTrack's multi-identity confirmation
logic doesn't suit a single intermittently-detected object), linearly interpolates
short ball-detection gaps, and renders boxes/IDs/a fading ball trail to an output
video.

Usage:
    python scripts/track_video.py --video data/raw/online_match_01.mp4 --max-frames 500
    python scripts/track_video.py --video data/raw/online_match_01.mp4
"""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from ball_kalman import GATE_CHI2, SIGMA_MIN_PX, BallKalmanFilter, innovation, mahalanobis_sq
from detect_frame import BALL_CONF, PLAYER_CONF
from detect_video import detect_video

MAX_BALL_GAP_FRAMES = 15  # interpolate real-detection gaps up to this long; longer gaps stay untracked
BALL_MARKER_RADIUS = 8
BALL_MARKER_COLOR = (0, 165, 255)  # BGR orange
BALL_TRACK_ID = 0  # constant - track_ball() is single-object, there's no per-segment identity to preserve

# Empirically derived from consecutive-frame displacement of the highest-confidence ball
# candidate per frame on a 500-frame slice of online_match_01 (1280x720): p85 was ~80px/frame
# (normal in-play motion), rising to ~220px/frame at p90 and ~740px/frame at p100. The top end
# is a full-frame teleport in 1/30s (~85+ m/s) - physically impossible for a volleyball, and
# traced to best_box() latching onto a false-positive hit on a player's head (rounder/more
# visible since the imgsz=1280 bump) instead of the real ball. But a false head-hit near the
# net during an attack often sits at a *plausible* ball speed too, so a speed cap alone can't
# fully separate the two - see SOFT/HARD split and confirmation logic in track_ball().
BALL_SOFT_SPEED_PX_PER_FRAME = 100  # accepted immediately - normal in-play motion (just above p85)
BALL_HARD_SPEED_PX_PER_FRAME = 220  # accepted only if a nearby detection confirms it next frame
BALL_JUMP_CONFIRM_WINDOW = 2  # frames to look ahead for confirmation of a soft-hard jump
BALL_JUMP_CONFIRM_RADIUS = 60  # px - how close a follow-up detection must land to confirm

# A position-based veto (excluding candidates near a player's head/feet, where the v2 ball
# model kept mistaking heads and bright shoe accents for the ball) was tried and reverted.
# It worked for v2, but the ball detector was retrained on a much larger, more diverse
# dataset (see scripts/train_ball.py) specifically to fix that false-positive pattern at the
# source rather than paper over it in tracking - and the veto turned out to have a much
# larger cost against the retrained (v3) model's detections: volleyball is played with
# hands near head height very often (sets, blocks, attacks near the net), and vetoing that
# region rejected genuine high-confidence ball detections far more often than it caught
# remaining false positives, dropping measured coverage on a 500-frame test slice from 98%
# to 59-66%. The residual false-positive rate after the v3 retrain is low enough that the
# motion/confirmation gate below is a better tradeoff than a positional veto.


def load_detections_csv(csv_path: Path) -> dict[int, dict[str, list[tuple[float, float, float, float, float]]]]:
    per_frame: dict[int, dict[str, list]] = {}
    for row in csv.DictReader(open(csv_path)):
        idx = int(row["frame_idx"])
        entry = per_frame.setdefault(idx, {"player": [], "ball": []})
        entry[row["class"]].append((
            float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]), float(row["confidence"]),
        ))
    return per_frame


def to_sv_detections(boxes: list[tuple[float, float, float, float, float]]) -> sv.Detections:
    if not boxes:
        return sv.Detections.empty()
    xyxy = np.array([b[:4] for b in boxes], dtype=np.float32)
    confidence = np.array([b[4] for b in boxes], dtype=np.float32)
    class_id = np.zeros(len(boxes), dtype=int)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


def best_box(boxes: list[tuple]) -> tuple | None:
    """Only one physical ball is ever in play - collapse to the single highest-confidence
    candidate per frame (occasionally the detector also fires on a spare/practice ball)."""
    return max(boxes, key=lambda b: b[4]) if boxes else None


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass
class BallEstimate:
    pos: tuple[float, float]
    vel: tuple[float, float]
    acc: tuple[float, float]
    is_contact: bool  # True if this frame's position came from a contact re-anchor, not a normal update


def track_ball_states(per_frame: dict[int, dict[str, list]], frame_indices: range,
                       high_conf: float = BALL_CONF, low_conf: float = 0.1,
                       max_extend_gap: int = MAX_BALL_GAP_FRAMES,
                       contact_bound: float = BALL_HARD_SPEED_PX_PER_FRAME,
                       confirm_window: int = BALL_JUMP_CONFIRM_WINDOW,
                       confirm_radius: float = BALL_JUMP_CONFIRM_RADIUS,
                       gate_chi2: float = GATE_CHI2) -> dict[int, BallEstimate]:
    """Kalman-filter-based single-object ball tracker (see ball_kalman.py for the pure
    math). Two states - SEARCHING (no filter) and TRACKING (filter anchored) - plus one
    internal event (CONTACT, a velocity re-seed) rather than a separate state, since a
    contact doesn't need any bookkeeping beyond resetting the estimate.

    SEARCHING: only a high_conf detection seeds a new filter (nothing else to judge
    plausibility against yet) - same bar as track_ball_heuristic's seed step.

    TRACKING, per frame: predict the state forward by the elapsed gap, then gate each
    raw candidate by Mahalanobis distance against the predicted position using a FIXED
    baseline measurement noise (not confidence-scaled - a candidate can't buy its way
    through the gate by being labeled low-confidence). Among candidates that pass the
    gate, pick the one minimizing `d^2 - 2*ln(conf)` (trajectory-consistency first,
    confidence as tiebreaker) and commit it with the Kalman update (which *does* use
    confidence-scaled measurement noise, so a low-confidence pick still nudges the
    filter gently). This is what lets a near, lower-confidence, trajectory-consistent
    detection beat a distant higher-confidence false positive - the false positive
    simply never passes the gate.

    If nothing passes the gate: a candidate within the looser contact_bound (the old
    heuristic's hard_speed, playing the same "how far could a real jump go" role) that
    gets corroborated by a follow-up detection within confirm_window frames (same
    look-ahead idea as track_ball_heuristic) is treated as a genuine contact - the
    ball's velocity actually changed (serve/spike/set/dig), so the stale KF velocity
    estimate is discarded and re-seeded from the jump rather than trusted. Anything
    else (beyond contact_bound, or unconfirmed) is rejected - left as a gap. contact_bound
    is applied as an absolute pre-filter to every candidate (not just the no-match
    fallback) before either gate runs - without this, a large predicted covariance (e.g.
    right after a fresh seed, before any real velocity is established) let the *relative*
    Mahalanobis gate alone accept candidates hundreds of pixels away, since
    "trajectory-consistent" is meaningless when the trajectory itself is still unknown.
    Verified against real detection data (online_match_01): this fixed several 500-900px
    single-frame jumps traced to low-confidence noise shortly after a seed.

    Known residual limitation (not chased further - see plan discussion on not
    deep-tuning against disposable placeholder footage): a false positive that happens to
    be locally self-consistent across 2+ frames (e.g. a bright spot in the crowd stands
    that doesn't move much) can still occasionally seed or extend a short-lived incorrect
    track, since a self-consistent false trajectory is - by construction - indistinguishable
    from a real one using motion alone. This is the same category of failure as the
    player-head/shoe false positives found earlier in this project's ball-detector work;
    a position-based veto was tried and rejected for that problem (it cost far more real
    coverage than it saved - see the removed-veto comment near the top of this file), and
    the same tradeoff applies here.

    A track with no accepted frame for more than max_extend_gap drops back to
    SEARCHING; the next high_conf detection anywhere seeds a fresh filter."""
    frames = list(frame_indices)
    states: dict[int, BallEstimate] = {}
    kf: BallKalmanFilter | None = None
    last_frame = None
    last_pos = None

    def candidates_at(frame_idx: int) -> list[tuple]:
        return per_frame.get(frame_idx, {"player": [], "ball": []})["ball"]

    def confirmed_near(target_pos: tuple[float, float], start_idx: int) -> bool:
        for look_idx in range(start_idx + 1, min(start_idx + 1 + confirm_window, len(frames))):
            look_box = best_box(candidates_at(frames[look_idx]))
            if look_box and math.dist(box_center(look_box[:4]), target_pos) <= confirm_radius:
                return True
        return False

    def record(frame_idx: int, is_contact: bool) -> None:
        states[frame_idx] = BallEstimate(pos=kf.position(), vel=kf.velocity(),
                                          acc=kf.acceleration(), is_contact=is_contact)

    for idx, frame_idx in enumerate(frames):
        candidates = candidates_at(frame_idx)
        if not candidates:
            continue

        anchored = kf is not None and last_frame is not None and frame_idx - last_frame <= max_extend_gap
        if not anchored:
            box = best_box(candidates)
            if box[4] < high_conf:
                continue
            kf = BallKalmanFilter()
            seed_pos = box_center(box[:4])
            kf.init_state(seed_pos)
            last_frame, last_pos = frame_idx, seed_pos
            record(frame_idx, is_contact=False)
            continue

        gap = frame_idx - last_frame
        x_pred, P_pred = kf.predict(dt=gap)

        # contact_bound is an absolute physical outer bound (independent of the filter's
        # own uncertainty) applied before either gate below - without this, a large
        # predicted covariance (e.g. right after a fresh seed, before any real velocity
        # is established) makes the *relative* Mahalanobis gate alone accept candidates
        # hundreds of pixels away, since "trajectory-consistent" is meaningless when the
        # trajectory itself is still unknown. Verified empirically against real detection
        # data: without this bound, low-confidence noise near a fresh seed produced
        # several 500-900px single-frame jumps that a plain absolute distance cap catches.
        bounded = [c for c in candidates if math.dist(box_center(c[:4]), last_pos) <= contact_bound * gap]

        R_gate = SIGMA_MIN_PX**2 * np.eye(2)
        gated = []
        for c in bounded:
            z = box_center(c[:4])
            y, S = innovation(x_pred, P_pred, z, R_gate)
            d2 = mahalanobis_sq(y, S)
            if d2 <= gate_chi2:
                gated.append((c, z, d2))

        if gated:
            box, new_pos, _ = min(gated, key=lambda item: item[2] - 2 * math.log(max(item[0][4], 1e-6)))
            kf.update(x_pred, P_pred, new_pos, box[4])
            last_frame, last_pos = frame_idx, new_pos
            record(frame_idx, is_contact=False)
            continue

        if bounded:
            box = best_box(bounded)
            new_pos = box_center(box[:4])
            if box[4] >= low_conf and confirmed_near(new_pos, idx):
                vel = ((new_pos[0] - last_pos[0]) / gap, (new_pos[1] - last_pos[1]) / gap)
                kf.reinit_after_contact(new_pos, vel)
                last_frame, last_pos = frame_idx, new_pos
                record(frame_idx, is_contact=True)
    return states


def track_ball(per_frame: dict[int, dict[str, list]], frame_indices: range, **kwargs) -> dict[int, tuple[float, float]]:
    """Thin wrapper over track_ball_states() preserving the plain frame_idx -> (cx, cy)
    contract the rest of the pipeline (interpolate_gaps, run_tracking's render loop)
    expects."""
    return {f: est.pos for f, est in track_ball_states(per_frame, frame_indices, **kwargs).items()}


def interpolate_gaps(positions: dict[int, tuple[float, float]], max_gap: int) -> dict[int, tuple[float, float, bool]]:
    """positions: frame_idx -> (cx, cy) for real detections within a single track.
    Returns frame_idx -> (cx, cy, is_interpolated), including the originals."""
    filled = {f: (x, y, False) for f, (x, y) in positions.items()}
    known = sorted(positions)
    for a, b in zip(known, known[1:]):
        gap = b - a
        if 1 < gap <= max_gap:
            (ax, ay), (bx, by) = positions[a], positions[b]
            for t in range(1, gap):
                frac = t / gap
                filled[a + t] = (ax + (bx - ax) * frac, ay + (by - ay) * frac, True)
    return filled


def interpolate_gaps_ballistic(states: dict[int, BallEstimate], max_gap: int) -> dict[int, tuple[float, float, bool]]:
    """Like interpolate_gaps(), but fills a gap with a parabola anchored at both
    endpoints using the KF's acceleration estimate at the gap start, instead of a
    straight line - a closer match to a volleyball's actual (roughly parabolic) free
    flight. p(t) = pA + (pB-pA)*frac + 0.5*acc*t*(t-gap) passes exactly through both
    endpoints and reduces to plain linear interpolation when acc=0, so it's never a
    regression on an already-straight segment.

    Falls back to the plain linear interpolate_gaps() for a gap where either endpoint
    is flagged is_contact - a contact means the ball's velocity changed somewhere
    around that endpoint, so a single parabola spanning the whole gap would be wrong."""
    positions = {f: est.pos for f, est in states.items()}
    filled = dict(interpolate_gaps(positions, max_gap))  # linear fallback, also seeds the real (non-interpolated) points

    known = sorted(states)
    for a, b in zip(known, known[1:]):
        gap = b - a
        if not (1 < gap <= max_gap):
            continue
        if states[a].is_contact or states[b].is_contact:
            continue  # keep the linear fill already in `filled`
        pax, pay = states[a].pos
        pbx, pby = states[b].pos
        acc_x, acc_y = states[a].acc
        for t in range(1, gap):
            frac = t / gap
            px = pax + (pbx - pax) * frac + 0.5 * acc_x * t * (t - gap)
            py = pay + (pby - pay) * frac + 0.5 * acc_y * t * (t - gap)
            filled[a + t] = (px, py, True)
    return filled


def track_ball_heuristic(per_frame: dict[int, dict[str, list]], frame_indices: range,
                          high_conf: float = BALL_CONF, low_conf: float = 0.1,
                          max_extend_gap: int = MAX_BALL_GAP_FRAMES,
                          soft_speed: float = BALL_SOFT_SPEED_PX_PER_FRAME,
                          hard_speed: float = BALL_HARD_SPEED_PX_PER_FRAME,
                          confirm_window: int = BALL_JUMP_CONFIRM_WINDOW,
                          confirm_radius: float = BALL_JUMP_CONFIRM_RADIUS) -> dict[int, tuple[float, float]]:
    """Lightweight single-object heuristic ball tracker (deliberately not sv.ByteTrack).

    Kept alongside the Kalman-filter-based track_ball() (see below) as a --tracker
    heuristic fallback and free A/B regression baseline - useful specifically because
    the KF tracker's constants aren't being deep-tuned against this project's current
    (disposable/placeholder) footage.

    ByteTrack requires two consecutive frames of IoU-overlapping detections before a new
    track is trusted (see supervision's STrack.activate/update) - a bar built for
    disambiguating many similar simultaneous objects. There's only ever one ball, and
    best_box() already collapses each frame to a single best candidate, so that
    confirmation delay has no upside here and empirically drops most genuine detections
    of a small, fast, intermittently-detected object.

    With no recent accepted position to anchor against, trust only a high_conf detection
    (nothing else to judge plausibility by). Once anchored, a candidate within soft_speed
    (scaled by elapsed frames) of the last accepted position is trusted immediately - normal
    in-play motion. A candidate beyond soft_speed but within hard_speed (a real but fast
    jump, e.g. a spike) is only accepted if a follow-up detection lands within
    confirm_radius of it in the next confirm_window frames. Anything beyond hard_speed, or
    an unconfirmed fast jump, is left as a gap for interpolate_gaps to bridge - a
    straight-line guess is a safer bet than an ungated (and possibly wrong) detection."""
    frames = list(frame_indices)
    positions: dict[int, tuple[float, float]] = {}
    last_frame = None
    last_pos = None

    def candidates_at(frame_idx: int) -> list[tuple]:
        return per_frame.get(frame_idx, {"player": [], "ball": []})["ball"]

    def confirmed_near(target_pos: tuple[float, float], start_idx: int) -> bool:
        for look_idx in range(start_idx + 1, min(start_idx + 1 + confirm_window, len(frames))):
            look_box = best_box(candidates_at(frames[look_idx]))
            if look_box and math.dist(box_center(look_box[:4]), target_pos) <= confirm_radius:
                return True
        return False

    for idx, frame_idx in enumerate(frames):
        candidates = candidates_at(frame_idx)
        if not candidates:
            continue

        anchored = last_frame is not None and frame_idx - last_frame <= max_extend_gap
        if not anchored:
            box = best_box(candidates)
            if box[4] < high_conf:
                continue
            positions[frame_idx] = box_center(box[:4])
            last_frame, last_pos = frame_idx, positions[frame_idx]
            continue

        gap = frame_idx - last_frame
        candidates = [c for c in candidates if math.dist(box_center(c[:4]), last_pos) <= hard_speed * gap]
        if not candidates:
            continue
        box = best_box(candidates)
        if box[4] < low_conf:
            continue

        new_pos = box_center(box[:4])
        dist = math.dist(new_pos, last_pos)
        if dist > soft_speed * gap and not confirmed_near(new_pos, idx):
            continue

        positions[frame_idx] = new_pos
        last_frame, last_pos = frame_idx, new_pos
    return positions


def run_tracking(video_path: Path, detections_csv: Path, out_video_path: Path,
                  max_frames: int | None = None, start_frame: int = 0, tracker: str = "kf"):
    per_frame = load_detections_csv(detections_csv)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    limit = min(max_frames, total - start_frame) if max_frames else total - start_frame

    # Empirically tuned on a 500-frame slice: longer buffer / looser matching kept reducing
    # player ID *count* up to about this point (57 -> 40 -> 32 -> 30 unique ids), then plateaued.
    # That original tuning only ever measured total ID count, not ID *swaps* - a loose
    # minimum_matching_threshold (0.93 = accepts matches down to ~0.07 IoU) is exactly what lets
    # two crossing players' predicted boxes both fall in the same gate and get mismatched, since
    # ByteTrack has no appearance/Re-ID signal to disambiguate them. Tightened to 0.85 (still
    # looser than the 0.8 default) to reduce swap risk while spot-checking that ID count doesn't
    # blow up; minimum_consecutive_frames=2 delays new-track activation by a frame to suppress
    # one-off flicker without touching the matching gate. Revisit with appearance-based Re-ID
    # (see plan) if swaps are still visible at close-player crossings.
    player_tracker = sv.ByteTrack(track_activation_threshold=PLAYER_CONF, lost_track_buffer=120,
                                   minimum_matching_threshold=0.85, minimum_consecutive_frames=2,
                                   frame_rate=round(fps))

    # --- pass 1: track (sequential, order matters for player ByteTrack state) ---
    frame_players: dict[int, sv.Detections] = {}

    for i in range(limit):
        frame_idx = start_frame + i
        entry = per_frame.get(frame_idx, {"player": [], "ball": []})
        p_tracked = player_tracker.update_with_detections(to_sv_detections(entry["player"]))
        frame_players[frame_idx] = p_tracked

    if tracker == "kf":
        ball_states = track_ball_states(per_frame, range(start_frame, start_frame + limit))
        ball_by_frame = interpolate_gaps_ballistic(ball_states, MAX_BALL_GAP_FRAMES)
    else:
        ball_positions = track_ball_heuristic(per_frame, range(start_frame, start_frame + limit))
        ball_by_frame = interpolate_gaps(ball_positions, MAX_BALL_GAP_FRAMES)

    n_real = sum(1 for _, _, i in ball_by_frame.values() if not i)
    n_interp = sum(1 for _, _, i in ball_by_frame.values() if i)
    print(f"ball: {n_real} real positions, {n_interp} interpolated")

    # --- pass 2: render ---
    cap.release()
    cap = cv2.VideoCapture(str(video_path))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
    trace_annotator = sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK, trace_length=20, thickness=2)

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for i in range(limit):
        frame_idx = start_frame + i
        ok, frame = cap.read()
        if not ok:
            break

        p_tracked = frame_players[frame_idx]
        frame = box_annotator.annotate(scene=frame, detections=p_tracked)
        labels = [f"#{tid}" for tid in p_tracked.tracker_id] if len(p_tracked) else []
        frame = label_annotator.annotate(scene=frame, detections=p_tracked, labels=labels)

        if frame_idx in ball_by_frame:
            cx, cy, is_interp = ball_by_frame[frame_idx]
            ball_det = sv.Detections(
                xyxy=np.array([[cx - BALL_MARKER_RADIUS, cy - BALL_MARKER_RADIUS,
                                 cx + BALL_MARKER_RADIUS, cy + BALL_MARKER_RADIUS]], dtype=np.float32),
                confidence=np.array([1.0], dtype=np.float32),
                class_id=np.array([0]),
                tracker_id=np.array([BALL_TRACK_ID]),
            )
            frame = trace_annotator.annotate(scene=frame, detections=ball_det)
            # filled circle = real detection, hollow circle = interpolated
            cv2.circle(frame, (int(cx), int(cy)), BALL_MARKER_RADIUS, BALL_MARKER_COLOR,
                       -1 if not is_interp else 2)

        writer.write(frame)
        if (i + 1) % 200 == 0:
            print(f"  rendered {i + 1}/{limit}")

    writer.release()
    cap.release()

    n_player_ids = len({tid for dets in frame_players.values() for tid in dets.tracker_id})
    print(f"players: {n_player_ids} unique track ids over {limit} frames")
    print(f"done -> {out_video_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--detections-csv", type=Path, default=None)
    parser.add_argument("--out-video", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--tracker", choices=["kf", "heuristic"], default="kf",
                         help="kf: Kalman-filter trajectory tracker (default). heuristic: the older "
                              "speed-cap/confirmation tracker, kept for A/B comparison.")
    args = parser.parse_args()

    detections_csv = args.detections_csv or Path("data/annotations/video_detections") / f"{args.video.stem}.csv"
    if not detections_csv.exists():
        print(f"no cached detections at {detections_csv}, running detect_video.py first...")
        detect_video(args.video, detections_csv, max_frames=args.max_frames, start_frame=args.start_frame)

    out_video = args.out_video or Path("data/tracked") / f"{args.video.stem}_tracked.mp4"
    run_tracking(args.video, detections_csv, out_video, max_frames=args.max_frames,
                 start_frame=args.start_frame, tracker=args.tracker)


if __name__ == "__main__":
    main()
