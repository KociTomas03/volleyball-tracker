"""Frozen snapshot of the pre-BotSort ByteTrack player tracker (from commit 739ef5b,
before f51b3ec swapped player tracking to BoT-SORT+Re-ID).

Why this exists: `data/annotations/touch_review_SLAP_SVIT_1z_upr.csv`'s
`correct_track_id`/`pipeline_track_id` columns are track IDs assigned by THIS
algorithm, against `data/annotations/video_detections/SLAP_SVIT_1z_upr_validation5min_segv3.csv`.
Track IDs are run-specific bookkeeping, not a stable per-player identifier - comparing
them against any run using the current BoT-SORT tracker (or a different detections
CSV) is meaningless, discovered 2026-07-19 when a re-eval came back 0% attribution
accuracy even for a control run that changed nothing except the tracker. This module
lets scripts/eval_touch_attribution_by_position.py reconstruct the ORIGINAL run's
per-frame box positions for those track IDs, converting the ground truth into a
position-referenced (tracker-version-agnostic) form once, so future re-evaluations
don't need this snapshot again for anything but rebuilding that one reference.

Do not use this for new tracking work - scripts/track_video.py's current
`track_players` (BoT-SORT+Re-ID) is production. This is a historical-reproduction
tool only.

KNOWN LIMITATION (2026-07-19, unresolved): this snapshot does NOT actually reproduce
the original run's track IDs even when fed the true historical detections from the
correct frame-0 start - it creates far more distinct tracks (700+ by frame 12089) than
the original apparently had (~31). Most likely cause: the installed `supervision`
version's `ByteTrack` (flagged deprecated) behaves differently now than whatever
version was active when the ground truth was built, not something this snapshot alone
can fix by matching source code. See eval_touch_attribution_by_position.py's docstring
for the full investigation.
"""

import supervision as sv

from detect_frame import PLAYER_CONF
from track_video import to_sv_detections


def track_players_bytetrack(per_frame: dict[int, dict[str, list]], frame_indices: range,
                             fps: float) -> dict[int, sv.Detections]:
    """Verbatim from commit 739ef5b - see that commit's track_video.py for the original
    tuning rationale (minimum_matching_threshold=0.85, minimum_consecutive_frames=2)."""
    player_tracker = sv.ByteTrack(track_activation_threshold=PLAYER_CONF, lost_track_buffer=120,
                                   minimum_matching_threshold=0.85, minimum_consecutive_frames=2,
                                   frame_rate=round(fps))
    frame_players: dict[int, sv.Detections] = {}
    for frame_idx in frame_indices:
        entry = per_frame.get(frame_idx, {"player": [], "ball": []})
        frame_players[frame_idx] = player_tracker.update_with_detections(to_sv_detections(entry["player"]))
    return frame_players
