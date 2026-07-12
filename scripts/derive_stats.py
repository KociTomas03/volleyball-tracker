"""Derive rally boundaries, touch attribution, and court-zone occupancy from a
clip's calibrated player + ball trajectories.

Phase 5 of plan.md: explicit, inspectable heuristics over the tracking data already
produced by detect_video.py/track_video.py and the homography from calibrate.py - no
ML. Requires a saved homography for the clip (see calibrate.py); there's no pixel-only
fallback mode, since zone occupancy and net-relative rally detection are only
meaningful in calibrated court space.

Usage:
    python scripts/derive_stats.py --video data/real_fotage/SLAP_SVIT_1z_upr.mp4 --max-frames 1000
    python scripts/derive_stats.py --video data/real_fotage/SLAP_SVIT_1z_upr.mp4 \
        --homography data/calibration/SLAP_SVIT_1z_upr_homography.json
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from calibrate import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M, load_homography, pixel_to_court
from detect_video import detect_video
from track_video import (
    MAX_BALL_GAP_FRAMES,
    interpolate_gaps_ballistic,
    load_detections_csv,
    track_ball_states,
    track_players,
)

# All of the following are heuristic defaults, not yet empirically tuned against a
# full real-footage run (see plan.md's note on not deep-tuning against disposable
# footage) - revisit once a full SLAP_SVIT clip has been processed and eyeballed.
CONTACT_WINDOW_FRAMES = 5  # local-max window radius (~1/6s at ~30fps) for contact detection
CONTACT_MIN_EXCURSION_PX = 15.0  # minimum vertical excursion (px) to count as a real contact, not jitter
NET_CROSSING_X_MARGIN_M = 1.0  # how far outside the sidelines a net crossing may still register (serve/attack near the antenna)
MAX_DEAD_GAP_FRAMES = 90  # ~3s at ~30fps - longer than this without an in-bounds ball position ends a rally segment
MIN_RALLY_CROSSINGS = 2  # a real rally has the ball crossing the net at least this many times (serve + return); fewer is a toss/warm-up touch
OUT_OF_BOUNDS_MARGIN_M = 3.0  # ball beyond court+this margin is treated as "out of active play" for rally segmentation

# Regulation: the attack line sits 3m from the net, not 3m from the baseline. Kept as
# an independent constant rather than reusing calibrate.py's ATTACK_LINE_M, which is
# measured from the baseline there (used only for that file's visual overlay) - using
# it here would put the front/back zone boundary in the wrong place. Worth reconciling
# with calibrate.py at some point, but that's Phase 4 code outside this change's scope.
FRONT_ZONE_DEPTH_M = 3.0

# Verified on real footage (SLAP_SVIT_1z_upr, frames 26700-28199): a referee standing
# on the elevated stand at the net post got tracked as track_id 13 for the whole
# 1500-frame slice and moved only 59px (bbox-bottom, diagonal bounding extent) the
# entire time, vs. 300-700px for every real player track observed for a comparable
# duration in the same slice. min_frames guards against flagging a real player's
# brief, legitimately-still moment (e.g. mid-rally poised in a passing stance) as
# stationary just because a short track didn't have time to move - only a track this
# long staying this still is a reliable non-player signal. Deliberately pixel-space,
# not court-space: a court-space extent would inherit the same calibration distortion
# already documented in within_court_bounds, undermining the very separation this is
# meant to detect.
STATIONARY_MIN_FRAMES = 150  # ~5s at ~30fps
STATIONARY_MAX_EXTENT_PX = 150.0  # well above the referee's 59px, well below any real player's 300px+


def within_court_bounds(point_court: tuple[float, float], margin_m: float = OUT_OF_BOUNDS_MARGIN_M) -> bool:
    """True if a court-space point falls within the court, padded by `margin_m`.

    Used to gate both the ball (see segment_rallies) and player positions before
    they're trusted for stats. Player-position gating matters because a homography
    fit from a handful of manually-clicked points can extrapolate wildly for pixels
    far from those points (a fundamental homography property: points approaching the
    fit's vanishing line project toward infinity) - verified on real calibration data
    (SLAP_SVIT_1z_upr) where a player bbox in a plausible on-court screen position
    reprojected to court coordinates in the hundreds of meters. Clamping such a point
    into range (as zone_for_position does for genuinely-near-boundary noise) would
    silently misattribute it to a real zone instead of surfacing the bad projection -
    filtering it out here keeps zone occupancy honest (sparser) rather than
    confidently wrong. This doesn't fix the underlying calibration accuracy (that's
    Phase 4 work - see calibrate.py), just stops it from corrupting Phase 5 output."""
    x, y = point_court
    return -margin_m <= x <= COURT_WIDTH_M + margin_m and -margin_m <= y <= COURT_LENGTH_M + margin_m


def player_foot_point(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Bottom-center of a player bbox - a proxy for where the player's feet touch the
    floor, used instead of box center (which sits mid-torso, not on the ground)."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y2


def stationary_track_ids(foot_px_by_frame: dict[int, dict[int, tuple[float, float]]],
                          min_frames: int = STATIONARY_MIN_FRAMES,
                          max_extent_px: float = STATIONARY_MAX_EXTENT_PX) -> set[int]:
    """Track ids that barely move (bounding-box diagonal extent of their pixel foot
    point stays under `max_extent_px`) over at least `min_frames` observations -
    almost never a real, actively-playing volleyball player over that many frames, and
    empirically exactly how a court-side referee/official shows up (see the constants'
    docstring above). Tracks with fewer than `min_frames` observations aren't judged
    either way - not enough evidence to tell a genuinely brief, still moment apart from
    a non-player."""
    positions_by_track: dict[int, list[tuple[float, float]]] = {}
    for players in foot_px_by_frame.values():
        for track_id, pos in players.items():
            positions_by_track.setdefault(track_id, []).append(pos)

    stationary = set()
    for track_id, positions in positions_by_track.items():
        if len(positions) < min_frames:
            continue
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        extent = math.dist((min(xs), min(ys)), (max(xs), max(ys)))
        if extent <= max_extent_px:
            stationary.add(track_id)
    return stationary


def find_ball_contacts(ball_y_by_frame: dict[int, float], window: int = CONTACT_WINDOW_FRAMES,
                        min_excursion_px: float = CONTACT_MIN_EXCURSION_PX) -> list[int]:
    """Local maxima of the ball's *pixel*-y position - screen-down is physically low,
    so a local max in pixel-y is a local minimum in height, i.e. plan.md's literal
    "local minimum in the ball's height trajectory" heuristic for an approximate
    contact point. Pixel space is deliberate: the homography only maps the ground
    plane, and the ball isn't on the ground at contact height, so there's no
    calibrated "height" available to use instead.

    A frame is a contact if it's the (first, in case of a tie) maximum within a
    +/-`window`-frame neighborhood and that neighborhood has at least
    `min_excursion_px` of vertical range - both gate out jitter-driven maxima. The
    window requirement also enforces a natural minimum separation between distinct
    contacts, since a second peak within `window` frames of a taller one can't itself
    be the window-max."""
    frames = sorted(ball_y_by_frame)
    frame_set = set(frames)
    contacts: list[int] = []
    for f in frames:
        neighborhood = range(f - window, f + window + 1)
        if not all(wf in frame_set for wf in neighborhood):
            continue
        values = [ball_y_by_frame[wf] for wf in neighborhood]
        y = ball_y_by_frame[f]
        if y != max(values):
            continue
        if y - min(values) < min_excursion_px:
            continue
        if any(ball_y_by_frame[wf] == y and wf < f for wf in neighborhood):
            continue  # keep only the first frame of a flat plateau at the max
        contacts.append(f)
    return contacts


def detect_net_crossings(ball_court_positions: dict[int, tuple[float, float]], net_y: float,
                          x_bounds: tuple[float, float]) -> list[int]:
    """Frames (the later frame of each crossing pair) where the ball's court-space y
    crosses `net_y`, restricted to consecutive frame indices (a "crossing" spanning a
    tracking gap isn't a real-time net crossing) and to plausible in-court x (gated by
    `x_bounds`, wider than the court itself to allow a serve/attack near the antenna)."""
    frames = sorted(ball_court_positions)
    lo_x, hi_x = x_bounds
    crossings: list[int] = []
    for a, b in zip(frames, frames[1:]):
        if b - a != 1:
            continue
        xa, ya = ball_court_positions[a]
        xb, yb = ball_court_positions[b]
        if ya == net_y or yb == net_y:
            continue
        if (ya - net_y) * (yb - net_y) < 0 and lo_x <= xa <= hi_x and lo_x <= xb <= hi_x:
            crossings.append(b)
    return crossings


@dataclass
class Rally:
    start_frame: int
    end_frame: int
    net_crossings: int


def segment_rallies(ball_court_positions: dict[int, tuple[float, float]], net_crossings: list[int],
                     max_dead_gap_frames: int = MAX_DEAD_GAP_FRAMES, min_crossings: int = MIN_RALLY_CROSSINGS,
                     out_of_bounds_margin_m: float = OUT_OF_BOUNDS_MARGIN_M) -> list[Rally]:
    """Splits the ball's in-bounds frames into candidate segments wherever there's a
    gap longer than `max_dead_gap_frames` - either a real tracking gap, or the ball
    sitting outside the court+margin for a sustained stretch, since out-of-bounds
    frames are dropped before segmenting and so behave exactly like a gap. This
    unifies the two failure modes plan.md calls out ("long gaps OR the ball leaving
    the court area") into one mechanism. Segments are only kept as rallies if they
    contain at least `min_crossings` net crossings - filters out isolated tosses or
    warm-up touches that aren't real rallies."""
    active_frames = sorted(
        f for f, pos in ball_court_positions.items() if within_court_bounds(pos, out_of_bounds_margin_m)
    )

    segments: list[tuple[int, int]] = []
    seg_start = seg_end = None
    for f in active_frames:
        if seg_start is None:
            seg_start = seg_end = f
        elif f - seg_end <= max_dead_gap_frames:
            seg_end = f
        else:
            segments.append((seg_start, seg_end))
            seg_start = seg_end = f
    if seg_start is not None:
        segments.append((seg_start, seg_end))

    rallies = []
    for start, end in segments:
        count = sum(1 for c in net_crossings if start <= c <= end)
        if count >= min_crossings:
            rallies.append(Rally(start_frame=start, end_frame=end, net_crossings=count))
    return rallies


def frames_within_rallies(rallies: list[Rally]) -> set[int]:
    """All frame indices covered by any rally's [start_frame, end_frame] range -
    used to restrict touch attribution and zone occupancy to actual in-play periods,
    since neither should reflect dead-time noise that segment_rallies already knows
    how to exclude."""
    frames: set[int] = set()
    for r in rallies:
        frames.update(range(r.start_frame, r.end_frame + 1))
    return frames


def find_nearest_player(point: tuple[float, float],
                         candidates: dict[int, tuple[float, float]]) -> tuple[int, float] | None:
    if not candidates:
        return None
    return min(((tid, math.dist(point, pos)) for tid, pos in candidates.items()), key=lambda t: t[1])


@dataclass
class Touch:
    frame_idx: int
    track_id: int | None
    distance_m: float | None


def attribute_touches(contacts: list[int], ball_positions_court: dict[int, tuple[float, float]],
                       player_positions_court_by_frame: dict[int, dict[int, tuple[float, float]]]) -> list[Touch]:
    """For each contact frame, attribute the touch to whichever tracked player's foot
    point is nearest the ball in court space at that frame. Falls back to an
    unattributed touch (track_id=None) if no player is tracked that frame (occlusion)
    or the ball itself has no position there - a documented limitation, same shape as
    this project's other accepted tracking-limitation tradeoffs."""
    touches = []
    for frame_idx in contacts:
        ball_pos = ball_positions_court.get(frame_idx)
        if ball_pos is None:
            continue
        nearest = find_nearest_player(ball_pos, player_positions_court_by_frame.get(frame_idx, {}))
        if nearest is None:
            touches.append(Touch(frame_idx=frame_idx, track_id=None, distance_m=None))
        else:
            track_id, dist = nearest
            touches.append(Touch(frame_idx=frame_idx, track_id=track_id, distance_m=dist))
    return touches


def zone_for_position(point_court: tuple[float, float]) -> int:
    """Bucket a court-space position into a standard volleyball zone (1-6), using the
    regulation 2-row layout per court half: a 3m-deep front row (zones 4-3-2, nearest
    the net) and a 6m-deep back row (zones 5-6-1, nearest the team's own baseline),
    each split into 3 equal 3m-wide columns.

    Column left/right is defined from each team's own perspective facing the net:
    for the near team (facing +y) "left" is low x, matching calibrate.py's
    baseline_near_left=(0,0) naming convention; the far team faces -y, so their
    left/right (and therefore their column-to-zone mapping) mirrors the near team's,
    same as the real court's 180-degree-symmetric zone layout."""
    x, y = point_court
    x = min(max(x, 0.0), COURT_WIDTH_M - 1e-9)
    y = min(max(y, 0.0), COURT_LENGTH_M - 1e-9)
    col_width = COURT_WIDTH_M / 3
    col = int(x // col_width)

    if y <= NET_Y_M:
        is_front = (NET_Y_M - y) <= FRONT_ZONE_DEPTH_M
        front_ids, back_ids = {0: 4, 1: 3, 2: 2}, {0: 5, 1: 6, 2: 1}
    else:
        is_front = (y - NET_Y_M) <= FRONT_ZONE_DEPTH_M
        front_ids, back_ids = {2: 4, 1: 3, 0: 2}, {2: 5, 1: 6, 0: 1}
    return front_ids[col] if is_front else back_ids[col]


def compute_zone_occupancy(
    player_positions_court_by_frame: dict[int, dict[int, tuple[float, float]]],
) -> dict[int, dict[int, int]]:
    """Per-track-id frame counts in each zone, over the whole processed range."""
    occupancy: dict[int, dict[int, int]] = {}
    for players in player_positions_court_by_frame.values():
        for track_id, pos in players.items():
            zone = zone_for_position(pos)
            zones = occupancy.setdefault(track_id, {})
            zones[zone] = zones.get(zone, 0) + 1
    return occupancy


def derive_stats(video_path: Path, detections_csv: Path, homography_path: Path,
                  max_frames: int | None = None, start_frame: int = 0) -> dict:
    per_frame = load_detections_csv(detections_csv)
    homography = load_homography(homography_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    limit = min(max_frames, total - start_frame) if max_frames else total - start_frame
    frame_indices = range(start_frame, start_frame + limit)

    frame_players = track_players(per_frame, frame_indices, fps)
    ball_states = track_ball_states(per_frame, frame_indices)
    ball_by_frame = interpolate_gaps_ballistic(ball_states, MAX_BALL_GAP_FRAMES)

    ball_y_px_by_frame = {f: cy for f, (_cx, cy, _interp) in ball_by_frame.items()}
    ball_positions_court = {
        f: pixel_to_court((cx, cy), homography) for f, (cx, cy, _interp) in ball_by_frame.items()
    }
    foot_px_by_frame: dict[int, dict[int, tuple[float, float]]] = {}
    player_positions_court_by_frame: dict[int, dict[int, tuple[float, float]]] = {}
    for frame_idx, dets in frame_players.items():
        feet, players = {}, {}
        for box, track_id in zip(dets.xyxy, dets.tracker_id):
            track_id = int(track_id)
            foot_px = player_foot_point(tuple(float(v) for v in box))
            feet[track_id] = foot_px
            court_pos = pixel_to_court(foot_px, homography)
            if within_court_bounds(court_pos):
                players[track_id] = court_pos
        foot_px_by_frame[frame_idx] = feet
        player_positions_court_by_frame[frame_idx] = players

    stationary_ids = stationary_track_ids(foot_px_by_frame)
    player_positions_court_by_frame = {
        frame_idx: {tid: pos for tid, pos in players.items() if tid not in stationary_ids}
        for frame_idx, players in player_positions_court_by_frame.items()
    }

    contacts = find_ball_contacts(ball_y_px_by_frame)
    net_crossings = detect_net_crossings(
        ball_positions_court, NET_Y_M, x_bounds=(-NET_CROSSING_X_MARGIN_M, COURT_WIDTH_M + NET_CROSSING_X_MARGIN_M)
    )
    rallies = segment_rallies(ball_positions_court, net_crossings)

    # Touches and zone occupancy should only reflect actual in-play periods, not
    # dead-time noise - segment_rallies already knows how to tell the two apart,
    # so reuse that instead of computing these stats over the whole processed range.
    in_play_frames = frames_within_rallies(rallies)
    in_play_contacts = [f for f in contacts if f in in_play_frames]
    in_play_player_positions = {
        frame_idx: players for frame_idx, players in player_positions_court_by_frame.items()
        if frame_idx in in_play_frames
    }
    touches = attribute_touches(in_play_contacts, ball_positions_court, in_play_player_positions)
    zone_occupancy = compute_zone_occupancy(in_play_player_positions)

    return {
        "video": video_path.name,
        "fps": fps,
        "frame_range": [start_frame, start_frame + limit - 1],
        "rallies": [
            {**asdict(r), "start_s": round(r.start_frame / fps, 2), "end_s": round(r.end_frame / fps, 2)}
            for r in rallies
        ],
        "touches": [
            {**asdict(t), "time_s": round(t.frame_idx / fps, 2),
             "distance_m": round(t.distance_m, 2) if t.distance_m is not None else None}
            for t in touches
        ],
        "zone_occupancy": {str(track_id): zones for track_id, zones in zone_occupancy.items()},
        "excluded_stationary_track_ids": sorted(stationary_ids),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--detections-csv", type=Path, default=None)
    parser.add_argument("--homography", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    detections_csv = args.detections_csv or Path("data/annotations/video_detections") / f"{args.video.stem}.csv"
    if not detections_csv.exists():
        print(f"no cached detections at {detections_csv}, running detect_video.py first...")
        detect_video(args.video, detections_csv, max_frames=args.max_frames, start_frame=args.start_frame)

    homography_path = args.homography or Path("data/calibration") / f"{args.video.stem}_homography.json"
    if not homography_path.exists():
        raise FileNotFoundError(
            f"no homography at {homography_path} - Phase 5 needs calibrated court positions "
            f"(see calibrate.py); run it for this clip first."
        )

    stats = derive_stats(args.video, detections_csv, homography_path,
                          max_frames=args.max_frames, start_frame=args.start_frame)

    out_path = args.out or Path("data/stats") / f"{args.video.stem}_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"rallies: {len(stats['rallies'])}, touches: {len(stats['touches'])}, "
          f"tracked players: {len(stats['zone_occupancy'])}")
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
