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

# Verified on real footage (SLAP_SVIT_1z_upr): 29% of raw find_ball_contacts output
# (110/385) were pairs less than 10 frames apart in the same rally - implausible for
# real consecutive touches. Traced to segmentation-detector centroid jitter right around
# a real contact (e.g. frames 12086-12100: a genuine extremum at 12086, then a single-
# frame 30px jump - not physics, the ball can't teleport near its apex - followed by
# ~10 frames wobbling in a tight band, each independently clearing
# CONTACT_MIN_EXCURSION_PX and getting flagged as its own contact). No excursion
# threshold alone can fix this: close-pair excursions ranged from 1px (clear noise) to
# 789px (a real fast block/dig exchange that must NOT be merged away). A median filter
# over the raw trajectory suppresses single-frame jitter while preserving genuine
# parabolic reversals. Window chosen by prototyping against the real clip: at 7, the
# 12086-12100 jitter cluster (4 spurious contacts) collapsed to 1, while a real fast
# exchange (frames 12144/12152/12155, huge excursions) survived as 3 distinct contacts;
# close-pair count dropped 110->75 (partial, not zero - some clusters need more, and
# over-widening risks smoothing away real quick exchanges, so this isn't chased further
# by number alone - see the visual audit that followed).
CONTACT_SMOOTHING_WINDOW = 7
NET_CROSSING_X_MARGIN_M = 1.0  # how far outside the sidelines a net crossing may still register (serve/attack near the antenna)
MAX_DEAD_GAP_FRAMES = 90  # ~3s at ~30fps - longer than this without an in-bounds ball position ends a rally segment
MIN_RALLY_CROSSINGS = 2  # a real rally has the ball crossing the net at least this many times (serve + return); fewer is a toss/warm-up touch
OUT_OF_BOUNDS_MARGIN_M = 3.0  # ball beyond court+this margin is treated as "out of active play" for rally segmentation

# Pixel-space (see attribute_touches for why). Calibrated against real footage
# (SLAP_SVIT_1z_upr, 1920x1080): confirmed real touches sit at 26-137px; a spurious
# dead-ball "contact" was 128px from its nearest uninvolved player, overlapping the real-
# touch range - a pixel cutoff alone can't fully separate the two failure modes, but 150px
# keeps both verified real touches while excluding the more clearly-uninvolved players
# (250px+) seen in the untracked-toucher/off-frame cases.
MAX_TOUCH_ATTRIBUTION_DISTANCE_PX = 150.0

# User-reported wrong attributions on real footage (SLAP_SVIT_1z_upr) traced to the
# same root cause 2 of 3 times: the true toucher WAS seen by the raw YOLO detector
# (0.221-0.435 confidence, right at the ball) but never became an available track -
# either below ByteTrack's track_activation_threshold (PLAYER_CONF=0.4), or above it
# but not yet confirmed by minimum_consecutive_frames (see track_players) during the
# fast, blur-prone motion of a jump/spike. attribute_touches then fell back to
# whichever OTHER, unrelated tracked player was nearest, producing a confident wrong
# answer. 0.15 sits below both observed cases (0.221, 0.303) while staying well above
# obvious background noise.
RAW_DETECTION_VETO_MIN_CONF = 0.15

# The raw per-frame detector output is NOT deduplicated the way a single inference
# call's NMS would - the cached CSV keeps every detection down to its own low floor
# (see detect_video.py), so a single real player routinely shows up as several
# overlapping boxes at different confidences (verified on real footage: a tracked
# player at conf 0.452 had near-duplicate raw boxes at 0.189 and 0.11 covering almost
# the same region). Without deduplication, the veto above fired on these duplicates
# of the player *already, correctly* tracked - not on a genuinely different, unseen
# person - which is why it started rejecting attributions that were previously
# confirmed correct. A raw box this overlapped with an already-tracked box in the same
# frame is the same physical player, not new evidence.
RAW_DETECTION_DEDUP_IOU = 0.3

# 81% of unattributed touches (verified on real footage, SLAP_SVIT_1z_upr) had SOME raw
# player detection within 300px of the ball at the contact frame - the player was seen,
# just never confirmed into a ByteTrack track in time (track_activation_threshold or
# minimum_consecutive_frames=2, see track_players) during the fast, blur-prone motion of
# the contact itself. A track that starts a few frames before/after the contact is almost
# always the same physical player, since they can't move far in a fraction of a second.
# Prototyped against the real clip at several window sizes before choosing one: recovery
# gains taper off past 10 (37 newly-attributed touches at window=10 vs 41 at 15, 47 at
# 20), and the one window=10-boundary case spot-checked (frame 20849, offset 10) was
# still a tight, plausible match (ball fell exactly inside the matched box).
TRACK_MATCH_WINDOW_FRAMES = 10

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

# Camera/clip-specific (SLAP_SVIT_1z_upr, 1920x1080), same spirit as the constants
# above: the elevated referee stand at the net post sits at a fixed screen position -
# measured directly from three long-lived tracks that all landed there (track ids 6,
# 156, 401: x1 in [1005,1028], y1 in [253,287], x2 in [1060,1087], y2 in [353,429]
# across 800-3800 frames each), padded with margin. This exists because
# stationary_track_ids alone isn't fast enough: ByteTrack reassigns the referee a new
# track id every time it briefly loses them (18+ distinct excluded ids were observed
# in one 9000-frame slice), and each new id needs STATIONARY_MIN_FRAMES of stillness
# before that filter catches it - leaving a recurring window where a touch mid-net-
# play gets attributed to the referee instead of the real player. A fixed-position
# check has no such warm-up, since it's evaluated fresh per frame. Not used for
# stationary_track_ids itself (which is a different, camera-independent mechanism);
# only for excluding position data from touch attribution / zone occupancy.
REFEREE_ZONE_PX = (990.0, 240.0, 1100.0, 440.0)  # x1, y1, x2, y2


def in_referee_zone(point_px: tuple[float, float], zone: tuple[float, float, float, float] = REFEREE_ZONE_PX) -> bool:
    x, y = point_px
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2


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


def smooth_ball_trajectory(ball_y_by_frame: dict[int, float],
                            window: int = CONTACT_SMOOTHING_WINDOW) -> dict[int, float]:
    """Median-filter the ball's pixel-y trajectory over each frame's +/-`window//2`
    neighborhood (available frames only - a frame near a tracking gap is smoothed over
    whatever's actually present rather than skipped, unlike find_ball_contacts's stricter
    full-neighborhood requirement). Suppresses single-frame detector-centroid jitter
    (see CONTACT_SMOOTHING_WINDOW) before find_ball_contacts looks for extrema, without
    changing find_ball_contacts itself - keeps that function operating on a plain
    y-by-frame dict, pure and independently testable."""
    frames = sorted(ball_y_by_frame)
    frame_set = set(frames)
    half = window // 2
    smoothed = {}
    for f in frames:
        neighborhood = [wf for wf in range(f - half, f + half + 1) if wf in frame_set]
        values = sorted(ball_y_by_frame[wf] for wf in neighborhood)
        n = len(values)
        mid = n // 2
        smoothed[f] = values[mid] if n % 2 == 1 else (values[mid - 1] + values[mid]) / 2
    return smoothed


def find_ball_contacts(ball_y_by_frame: dict[int, float], window: int = CONTACT_WINDOW_FRAMES,
                        min_excursion_px: float = CONTACT_MIN_EXCURSION_PX) -> list[int]:
    """Local extrema (maxima OR minima) of the ball's *pixel*-y position - screen-down
    is physically low, so either direction is a real contact: a local MAX in pixel-y is
    a local minimum in height (ball near the floor reversing from falling to rising - a
    serve-receive, dig, or pass), while a local MIN in pixel-y is a local maximum in
    height (ball near the top of its arc reversing from rising to falling, or smashed
    sharply downward - a set, attack, or block). Originally only the maxima case was
    implemented; verified on real footage (SLAP_SVIT_1z_upr, plotting the trajectory
    against detected contacts) that this meant every contact near the top of the ball's
    arc - roughly half of all real touches in a rally - was structurally undetectable,
    regardless of any threshold tuning. Pixel space is deliberate: the homography only
    maps the ground plane, and the ball isn't on the ground at contact height, so
    there's no calibrated "height" available to use instead.

    A frame is a contact if it's the (first, in case of a tie) extremum within a
    +/-`window`-frame neighborhood and that neighborhood has at least
    `min_excursion_px` of vertical range - both gate out jitter-driven extrema. The
    window requirement also enforces a natural minimum separation between distinct
    contacts, since a second extremum within `window` frames of a more extreme one
    can't itself be the window's max or min."""
    frames = sorted(ball_y_by_frame)
    frame_set = set(frames)
    contacts: list[int] = []
    for f in frames:
        neighborhood = range(f - window, f + window + 1)
        if not all(wf in frame_set for wf in neighborhood):
            continue
        values = [ball_y_by_frame[wf] for wf in neighborhood]
        y = ball_y_by_frame[f]
        y_max, y_min = max(values), min(values)
        if y != y_max and y != y_min:
            continue
        if y_max - y_min < min_excursion_px:
            continue
        if any(ball_y_by_frame[wf] == y and wf < f for wf in neighborhood):
            continue  # keep only the first frame of a flat plateau at this extremum
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


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two boxes, 0.0 if they don't overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def distance_point_to_box(point: tuple[float, float], box: tuple[float, float, float, float]) -> float:
    """Euclidean distance from a point to the nearest point on a bbox - 0 if the point
    falls inside the box. Used (see find_nearest_player_by_box) instead of point-to-
    point distance for touch attribution, since a single representative point on a
    player (e.g. their foot) can be far from where they actually contact the ball."""
    px, py = point
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def find_nearest_player_by_box(point: tuple[float, float],
                                candidates: dict[int, tuple[float, float, float, float]]) -> tuple[int, float] | None:
    if not candidates:
        return None
    return min(((tid, distance_point_to_box(point, box)) for tid, box in candidates.items()), key=lambda t: t[1])


def nearby_player_boxes(player_boxes_px_by_frame: dict[int, dict[int, tuple[float, float, float, float]]],
                         frame_idx: int,
                         window: int = TRACK_MATCH_WINDOW_FRAMES) -> dict[int, tuple[float, float, float, float]]:
    """For every track id seen anywhere in `frame_idx`'s +/-`window`-frame neighborhood,
    return that track's box from whichever frame in the neighborhood is closest to
    `frame_idx` (ties keep the earlier offset). Used to give a track a fighting chance at
    attribution even if it wasn't confirmed yet - or already lost - at the exact contact
    frame (see TRACK_MATCH_WINDOW_FRAMES), on the assumption a player hasn't moved far in
    a fraction of a second."""
    best: dict[int, tuple[int, tuple[float, float, float, float]]] = {}
    for offset in range(-window, window + 1):
        boxes = player_boxes_px_by_frame.get(frame_idx + offset)
        if not boxes:
            continue
        abs_offset = abs(offset)
        for track_id, box in boxes.items():
            if track_id not in best or abs_offset < best[track_id][0]:
                best[track_id] = (abs_offset, box)
    return {track_id: box for track_id, (_, box) in best.items()}


@dataclass
class Touch:
    frame_idx: int
    track_id: int | None
    distance_px: float | None


def attribute_touches(contacts: list[int], ball_positions_px: dict[int, tuple[float, float]],
                       player_boxes_px_by_frame: dict[int, dict[int, tuple[float, float, float, float]]],
                       raw_player_boxes_px_by_frame: dict[int, list[tuple[float, float, float, float]]] | None = None,
                       max_distance_px: float = MAX_TOUCH_ATTRIBUTION_DISTANCE_PX,
                       track_match_window: int = TRACK_MATCH_WINDOW_FRAMES) -> list[Touch]:
    """For each contact frame, attribute the touch to whichever tracked player's full
    bounding box is nearest the ball in PIXEL space at that frame - but only if that
    nearest player is within `max_distance_px`.

    Deliberately the player's full box, not a single foot/center point: verified on
    real footage (SLAP_SVIT_1z_upr) that a foot-point proxy is systematically wrong
    during jumps and overhead reaches, since the actual contact happens at the hands,
    which can be a meter or more from the feet in image space (a jumping player's feet
    lift toward the ball while a standing toucher's raised arms move away from their
    planted feet) - this was the single largest source of misattribution found in a
    16-touch manual audit. Comparing against the whole box means the ball only needs
    to be near *some part* of the player, not a specific point on them.

    Deliberately pixel space, not court space: court-space nearest-player is unreliable
    near the net specifically, because the ball is airborne at contact height and the
    homography only maps the ground plane - an elevated ball's ground-plane reprojection
    can land closer to an uninvolved player than to the real toucher jumping directly
    beneath it. Pixel-space distance reflects visual proximity directly and doesn't have
    this height-reprojection distortion.

    If `raw_player_boxes_px_by_frame` is given, a tracked candidate that would otherwise
    be accepted is vetoed (downgraded to unattributed) when an untracked raw detection
    sits strictly closer to the ball than the tracked candidate. This exists because
    user-reported wrong attributions on real footage traced back to the true toucher
    being seen by the raw detector but never becoming an available *track* (below
    ByteTrack's activation threshold, or not yet confirmed) during the fast, blur-prone
    motion of a jump/spike - attribution then confidently named a different, unrelated
    tracked player instead. We can't name who the untracked detection is (no track id),
    but its mere existence closer to the ball is enough to know the tracked candidate
    probably isn't the real toucher, so unattributed is the honest answer.

    If no player is tracked close enough at the exact contact frame, falls back to
    nearby_player_boxes - a track confirmed a few frames before/after almost always
    belongs to the same physical player, since nobody moves far in a fraction of a
    second, and this is exactly the situation ByteTrack's confirmation lag
    (minimum_consecutive_frames, track_activation_threshold - see track_players)
    produces during the fast, blur-prone motion of a real contact. Verified on real
    footage (SLAP_SVIT_1z_upr): 81% of unattributed touches had SOME raw detection
    within 300px of the ball at the contact frame - the player was seen, just not
    confirmed into a track in time. Deliberately NOT applied when the exact-frame
    nearest was rejected by the veto above (rather than simply absent/too-far): that
    path already found a same-frame tracked candidate and rejected it on separate,
    specific evidence (a closer untracked detection) - re-searching a wider window
    wouldn't add new information there and would just override that judgement without
    cause. A prototype that didn't draw this distinction was verified (real footage,
    same clip) to flip 6 previously-confirmed-correct attributions to a different,
    stale-by-several-frames player for little or no distance improvement.

    Falls back to an unattributed touch (track_id=None) if no player is tracked or
    trackable-nearby that frame, the nearest one is implausibly far or vetoed, or the
    ball itself has no position there - a documented limitation, same shape as this
    project's other accepted tracking-limitation tradeoffs."""
    touches = []
    for frame_idx in contacts:
        ball_pos = ball_positions_px.get(frame_idx)
        if ball_pos is None:
            continue
        nearest = find_nearest_player_by_box(ball_pos, player_boxes_px_by_frame.get(frame_idx, {}))
        if nearest is not None and nearest[1] <= max_distance_px:
            track_id, dist = nearest
            raw_boxes = (raw_player_boxes_px_by_frame or {}).get(frame_idx, [])
            if raw_boxes and min(distance_point_to_box(ball_pos, b) for b in raw_boxes) < dist:
                touches.append(Touch(frame_idx=frame_idx, track_id=None, distance_px=dist))
            else:
                touches.append(Touch(frame_idx=frame_idx, track_id=track_id, distance_px=dist))
            continue

        rescued = find_nearest_player_by_box(
            ball_pos, nearby_player_boxes(player_boxes_px_by_frame, frame_idx, track_match_window)
        )
        if rescued is not None and rescued[1] <= max_distance_px:
            touches.append(Touch(frame_idx=frame_idx, track_id=rescued[0], distance_px=rescued[1]))
        else:
            fallback_dist = rescued[1] if rescued is not None else (nearest[1] if nearest is not None else None)
            touches.append(Touch(frame_idx=frame_idx, track_id=None, distance_px=fallback_dist))
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
    box_px_by_frame: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    player_positions_court_by_frame: dict[int, dict[int, tuple[float, float]]] = {}
    for frame_idx, dets in frame_players.items():
        feet, boxes, players = {}, {}, {}
        for raw_box, track_id in zip(dets.xyxy, dets.tracker_id):
            track_id = int(track_id)
            box = tuple(float(v) for v in raw_box)
            foot_px = player_foot_point(box)
            # Excluded here (per-frame, position-based), not just via stationary_ids
            # below - see REFEREE_ZONE_PX for why a track-id-history-based check alone
            # isn't fast enough to catch the referee after ByteTrack reassigns their id.
            if in_referee_zone(foot_px):
                continue
            feet[track_id] = foot_px
            boxes[track_id] = box
            court_pos = pixel_to_court(foot_px, homography)
            if within_court_bounds(court_pos):
                players[track_id] = court_pos
        foot_px_by_frame[frame_idx] = feet
        box_px_by_frame[frame_idx] = boxes
        player_positions_court_by_frame[frame_idx] = players

    stationary_ids = stationary_track_ids(foot_px_by_frame)
    player_positions_court_by_frame = {
        frame_idx: {tid: pos for tid, pos in players.items() if tid not in stationary_ids}
        for frame_idx, players in player_positions_court_by_frame.items()
    }
    # Pixel-space box counterpart for touch attribution (see attribute_touches) - unlike
    # the court-space dict above, this is NOT filtered by within_court_bounds, since a
    # bad homography reprojection for a given player shouldn't disqualify their
    # (perfectly valid) pixel box from being compared against the ball's pixel position.
    player_boxes_px_by_frame = {
        frame_idx: {tid: box for tid, box in boxes.items() if tid not in stationary_ids}
        for frame_idx, boxes in box_px_by_frame.items()
    }

    contacts = find_ball_contacts(smooth_ball_trajectory(ball_y_px_by_frame))
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
    in_play_player_boxes_px = {
        frame_idx: boxes for frame_idx, boxes in player_boxes_px_by_frame.items()
        if frame_idx in in_play_frames
    }
    # Raw (untracked) per-frame player detections, for attribute_touches's veto check -
    # see its docstring for why a tracked candidate isn't always trustworthy on its own.
    raw_player_boxes_px = {}
    for frame_idx in in_play_contacts:
        tracked_boxes_this_frame = list(player_boxes_px_by_frame.get(frame_idx, {}).values())
        raw_player_boxes_px[frame_idx] = [
            box[:4] for box in per_frame.get(frame_idx, {"player": []})["player"]
            if box[4] >= RAW_DETECTION_VETO_MIN_CONF
            and not in_referee_zone(player_foot_point(box[:4]))
            and not any(iou(box[:4], tracked) > RAW_DETECTION_DEDUP_IOU for tracked in tracked_boxes_this_frame)
        ]
    ball_positions_px = {f: (cx, cy) for f, (cx, cy, _interp) in ball_by_frame.items()}
    touches = attribute_touches(in_play_contacts, ball_positions_px, in_play_player_boxes_px, raw_player_boxes_px)
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
             "distance_px": round(t.distance_px, 1) if t.distance_px is not None else None}
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
