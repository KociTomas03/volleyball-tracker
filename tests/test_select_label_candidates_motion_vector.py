import random

from select_label_candidates_motion_vector import delta_v_scores, select_low_score_negatives


def test_delta_v_scores_high_at_direction_reversal():
    # Ball moves steadily right (vx=+50) up to frame 20, then steadily left
    # (vx=-50) after - a clean velocity reversal, the shape a real contact
    # produces. smooth_ball_trajectory's median filter spreads/damps sharp
    # corners (verified separately), so this checks relative distinction -
    # scores near the reversal are far above the flat baseline away from it -
    # rather than an exact frame/magnitude, which the smoothing makes brittle.
    ball_by_frame = {}
    for f in range(0, 21):
        ball_by_frame[f] = (50.0 * f, 0.0, False)
    for f in range(21, 41):
        ball_by_frame[f] = (1000.0 - 50.0 * (f - 20), 0.0, False)
    scores = delta_v_scores(ball_by_frame, half_window=2)
    near_reversal_peak = max(scores[f] for f in range(17, 25) if f in scores)
    baseline = max(scores[f] for f in range(7, 16) if f in scores)  # clean flat zone, clear of edge effects
    assert baseline == 0.0
    assert near_reversal_peak > 15.0


def test_delta_v_scores_near_zero_on_steady_motion():
    # Interior frames (away from smooth_ball_trajectory's partial-window
    # edge effects near the start/end of the range) should score ~0 on
    # perfectly linear motion - no real discontinuity anywhere.
    ball_by_frame = {f: (5.0 * f, 2.0 * f, False) for f in range(30)}
    scores = delta_v_scores(ball_by_frame, half_window=2)
    interior = {f: v for f, v in scores.items() if 8 <= f <= 21}
    assert interior and all(v < 1.0 for v in interior.values())


def test_delta_v_scores_skips_frames_without_full_window():
    ball_by_frame = {0: (0.0, 0.0, False), 1: (1.0, 0.0, False)}
    scores = delta_v_scores(ball_by_frame, half_window=2)
    assert scores == {}


def test_select_low_score_negatives_prefers_low_scores():
    scores = {f: float(f) for f in range(100)}  # monotonically increasing
    rng = random.Random(0)
    chosen = select_low_score_negatives(scores, quota=5, min_spacing=3, exclude=set(), rng=rng)
    assert len(chosen) == 5
    assert all(scores[f] < 60 for f in chosen)  # drawn from the low half


def test_select_low_score_negatives_respects_exclude():
    scores = {f: float(f) for f in range(20)}
    rng = random.Random(0)
    exclude = set(range(10))  # exclude the entire low half
    chosen = select_low_score_negatives(scores, quota=3, min_spacing=1, exclude=exclude, rng=rng)
    assert not (set(chosen) & exclude)


def test_select_low_score_negatives_respects_min_spacing():
    scores = {f: 0.0 for f in range(20)}  # all tied - spacing is the only constraint
    rng = random.Random(0)
    chosen = select_low_score_negatives(scores, quota=10, min_spacing=5, exclude=set(), rng=rng)
    chosen_sorted = sorted(chosen)
    assert all(b - a >= 5 for a, b in zip(chosen_sorted, chosen_sorted[1:]))
