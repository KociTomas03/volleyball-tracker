import pytest

from track_video import (
    BallEstimate,
    best_box,
    box_center,
    interpolate_gaps,
    interpolate_gaps_ballistic,
    to_sv_detections,
    track_ball,
    track_ball_heuristic,
    track_ball_states,
)

TRACKER_FNS = [track_ball, track_ball_heuristic]


def test_box_center_computes_midpoint():
    assert box_center((0.0, 0.0, 10.0, 20.0)) == (5.0, 10.0)


def test_box_center_handles_negative_and_fractional_coords():
    assert box_center((-4.0, 2.0, 6.0, 8.0)) == (1.0, 5.0)


def test_best_box_picks_highest_confidence():
    boxes = [(0, 0, 10, 10, 0.3), (5, 5, 15, 15, 0.9), (1, 1, 2, 2, 0.5)]
    assert best_box(boxes) == (5, 5, 15, 15, 0.9)


def test_best_box_empty_returns_none():
    assert best_box([]) is None


def test_interpolate_gaps_fills_short_gap_linearly():
    positions = {0: (0.0, 0.0), 4: (8.0, 4.0)}
    filled = interpolate_gaps(positions, max_gap=15)
    assert filled[0] == (0.0, 0.0, False)
    assert filled[4] == (8.0, 4.0, False)
    assert filled[1] == (2.0, 1.0, True)
    assert filled[2] == (4.0, 2.0, True)
    assert filled[3] == (6.0, 3.0, True)


def test_interpolate_gaps_does_not_bridge_beyond_max_gap():
    # gap of 20 frames with max_gap=15 should stay a hole - Phase 3's design intent
    # (per plan.md: long gaps stay untracked rather than guessing across them)
    positions = {0: (0.0, 0.0), 20: (20.0, 0.0)}
    filled = interpolate_gaps(positions, max_gap=15)
    assert set(filled) == {0, 20}


def test_interpolate_gaps_single_position_untouched():
    filled = interpolate_gaps({7: (1.0, 2.0)}, max_gap=15)
    assert filled == {7: (1.0, 2.0, False)}


def test_to_sv_detections_empty_list_returns_empty():
    dets = to_sv_detections([])
    assert len(dets) == 0


def test_to_sv_detections_preserves_bbox_and_confidence():
    dets = to_sv_detections([(1.0, 2.0, 3.0, 4.0, 0.75)])
    assert len(dets) == 1
    assert dets.xyxy[0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert dets.confidence[0] == pytest.approx(0.75)


@pytest.mark.parametrize("tracker_fn", TRACKER_FNS)
def test_track_ball_accepts_isolated_high_conf_detection(tracker_fn):
    # the whole point of both trackers over ByteTrack: a single isolated high-confidence
    # hit is trusted immediately, with no two-frame confirmation delay.
    per_frame = {5: {"player": [], "ball": [(10.0, 10.0, 20.0, 20.0, 0.5)]}}
    positions = tracker_fn(per_frame, range(0, 10), high_conf=0.25, low_conf=0.1)
    assert positions == {5: (15.0, 15.0)}


@pytest.mark.parametrize("tracker_fn", TRACKER_FNS)
def test_track_ball_ignores_low_conf_detection_with_no_recent_track(tracker_fn):
    per_frame = {5: {"player": [], "ball": [(10.0, 10.0, 20.0, 20.0, 0.15)]}}
    positions = tracker_fn(per_frame, range(0, 10), high_conf=0.25, low_conf=0.1)
    assert positions == {}


@pytest.mark.parametrize("tracker_fn", TRACKER_FNS)
def test_track_ball_extends_recent_track_with_low_conf_detection(tracker_fn):
    per_frame = {
        3: {"player": [], "ball": [(0.0, 0.0, 10.0, 10.0, 0.5)]},
        7: {"player": [], "ball": [(0.0, 0.0, 10.0, 10.0, 0.15)]},  # within max_extend_gap of frame 3
    }
    positions = tracker_fn(per_frame, range(0, 10), high_conf=0.25, low_conf=0.1, max_extend_gap=15)
    assert positions == {3: (5.0, 5.0), 7: (5.0, 5.0)}


@pytest.mark.parametrize("tracker_fn", TRACKER_FNS)
def test_track_ball_drops_low_conf_detection_beyond_extend_gap(tracker_fn):
    per_frame = {
        3: {"player": [], "ball": [(0.0, 0.0, 10.0, 10.0, 0.5)]},
        20: {"player": [], "ball": [(0.0, 0.0, 10.0, 10.0, 0.15)]},  # beyond max_extend_gap=15
    }
    positions = tracker_fn(per_frame, range(0, 25), high_conf=0.25, low_conf=0.1, max_extend_gap=15)
    assert positions == {3: (5.0, 5.0)}


@pytest.mark.parametrize("tracker_fn", TRACKER_FNS)
def test_track_ball_no_detections_returns_empty(tracker_fn):
    assert tracker_fn({}, range(0, 10), high_conf=0.25, low_conf=0.1) == {}


def test_track_ball_heuristic_rejects_spatially_implausible_anchored_jump():
    # anchor at (5,5), then a high-conf detection clear across the frame one frame later -
    # this is exactly the "false positive on a player's head" glitch: without a distance
    # gate it would be accepted just for having high confidence.
    per_frame = {
        3: {"player": [], "ball": [(0.0, 0.0, 10.0, 10.0, 0.9)]},
        4: {"player": [], "ball": [(990.0, 990.0, 1000.0, 1000.0, 0.9)]},
    }
    positions = track_ball_heuristic(per_frame, range(0, 10), high_conf=0.25, low_conf=0.1,
                                      max_extend_gap=15, hard_speed=50)
    assert positions == {3: (5.0, 5.0)}


def test_track_ball_heuristic_prefers_plausible_candidate_over_distant_higher_conf():
    per_frame = {
        3: {"player": [], "ball": [(0.0, 0.0, 10.0, 10.0, 0.9)]},
        4: {"player": [], "ball": [
            (995.0, 995.0, 1005.0, 1005.0, 0.95),  # far, higher conf - rejected by distance gate
            (2.0, 2.0, 12.0, 12.0, 0.3),  # near, lower conf but plausible
        ]},
    }
    positions = track_ball_heuristic(per_frame, range(0, 10), high_conf=0.25, low_conf=0.1,
                                      max_extend_gap=15, hard_speed=50)
    assert positions[4] == (7.0, 7.0)


def test_track_ball_heuristic_accepts_fast_jump_when_confirmed_next_frame():
    # a jump beyond soft_speed but within hard_speed (e.g. a real spike) is trusted once a
    # follow-up detection lands near the new position - a genuine fast ball keeps getting
    # detected there, unlike a one-frame false-positive flicker.
    per_frame = {
        3: {"player": [], "ball": [(0.0, 0.0, 10.0, 10.0, 0.9)]},
        4: {"player": [], "ball": [(150.0, 150.0, 160.0, 160.0, 0.9)]},
        5: {"player": [], "ball": [(152.0, 152.0, 162.0, 162.0, 0.5)]},
    }
    positions = track_ball_heuristic(per_frame, range(0, 10), high_conf=0.25, low_conf=0.1, max_extend_gap=15,
                                      soft_speed=100, hard_speed=220, confirm_window=2, confirm_radius=60)
    assert positions[4] == (155.0, 155.0)


def test_track_ball_heuristic_rejects_unconfirmed_fast_jump():
    per_frame = {
        3: {"player": [], "ball": [(0.0, 0.0, 10.0, 10.0, 0.9)]},
        4: {"player": [], "ball": [(150.0, 150.0, 160.0, 160.0, 0.9)]},
        # no corroborating detection near (155, 155) in the following frames
    }
    positions = track_ball_heuristic(per_frame, range(0, 10), high_conf=0.25, low_conf=0.1, max_extend_gap=15,
                                      soft_speed=100, hard_speed=220, confirm_window=2, confirm_radius=60)
    assert positions == {3: (5.0, 5.0)}


# --- track_ball_states() (Kalman-filter tracker): trajectory-gated behavior ---
#
# These exercise the KF gate directly through track_ball_states() rather than the
# thin track_ball() wrapper, since the richer BallEstimate (velocity, is_contact) is
# what actually proves the gating/contact logic is doing the right thing, not just
# that *some* position came out. Verified empirically before being written down here
# (see plan discussion) - the KF's post-seed uncertainty is deliberately generous
# (INIT_VEL_VAR in ball_kalman.py), so a jump shortly after a fresh seed can pass the
# ordinary gate without needing the contact path at all; the contact mechanism is
# what matters once a trajectory has actually been established over several frames.

def test_track_ball_states_rejects_jump_beyond_contact_bound():
    # a jump far beyond even the loose contact_bound backstop is rejected outright,
    # regardless of confidence - the KF equivalent of the heuristic's hard_speed gate.
    per_frame = {
        3: {"player": [], "ball": [(0.0, 0.0, 10.0, 10.0, 0.9)]},
        4: {"player": [], "ball": [(990.0, 990.0, 1000.0, 1000.0, 0.9)]},
    }
    states = track_ball_states(per_frame, range(0, 10), high_conf=0.25, low_conf=0.1,
                                max_extend_gap=15, contact_bound=50)
    assert set(states) == {3}
    assert states[3].pos == (5.0, 5.0)


def test_track_ball_states_prefers_trajectory_consistent_candidate():
    per_frame = {
        3: {"player": [], "ball": [(0.0, 0.0, 10.0, 10.0, 0.9)]},
        4: {"player": [], "ball": [
            (995.0, 995.0, 1005.0, 1005.0, 0.95),  # far, higher conf - gated out
            (2.0, 2.0, 12.0, 12.0, 0.3),  # near, lower conf, trajectory-consistent - wins
        ]},
    }
    states = track_ball_states(per_frame, range(0, 10), high_conf=0.25, low_conf=0.1,
                                max_extend_gap=15, contact_bound=50)
    assert states[4].pos == pytest.approx((7.0, 7.0), abs=1.0)


def _steady_trajectory(n=10, vx=15.0, vy=10.0, start_frame=3):
    """n frames of constant-velocity motion, for tests that need an established
    trajectory (tight gate) before probing contact/rejection behavior."""
    per_frame = {}
    for i in range(n):
        f = start_frame + i
        x, y = 5.0 + vx * i, 5.0 + vy * i
        per_frame[f] = {"player": [], "ball": [(x - 5, y - 5, x + 5, y + 5, 0.9)]}
    return per_frame, start_frame + n - 1


def test_track_ball_states_confirmed_direction_change_is_contact():
    per_frame, last = _steady_trajectory()
    per_frame[last + 1] = {"player": [], "ball": [(-55.0, 195.0, -45.0, 205.0, 0.9)]}  # sharp reversal
    per_frame[last + 2] = {"player": [], "ball": [(-57.0, 200.0, -47.0, 210.0, 0.5)]}  # confirms new direction
    states = track_ball_states(per_frame, range(0, 20), high_conf=0.25, low_conf=0.1,
                                max_extend_gap=15, contact_bound=220, confirm_window=2, confirm_radius=60)
    assert states[last + 1].is_contact is True
    # velocity re-seeded from the jump, not still pointing in the pre-reversal direction
    assert states[last + 1].vel[0] < 0
    # subsequent frame keeps tracking the new direction, not snapping back
    assert states[last + 2].is_contact is False
    assert states[last + 2].pos[0] < states[last + 1].pos[0]


def test_track_ball_states_unconfirmed_jump_after_established_trajectory_rejected():
    per_frame, last = _steady_trajectory()
    per_frame[last + 1] = {"player": [], "ball": [(-55.0, 195.0, -45.0, 205.0, 0.9)]}
    # no confirming detection follows
    states = track_ball_states(per_frame, range(0, 20), high_conf=0.25, low_conf=0.1,
                                max_extend_gap=15, contact_bound=220, confirm_window=2, confirm_radius=60)
    assert (last + 1) not in states


def test_track_ball_wrapper_matches_states_positions():
    per_frame = {5: {"player": [], "ball": [(10.0, 10.0, 20.0, 20.0, 0.5)]}}
    states = track_ball_states(per_frame, range(0, 10), high_conf=0.25, low_conf=0.1)
    positions = track_ball(per_frame, range(0, 10), high_conf=0.25, low_conf=0.1)
    assert positions == {f: est.pos for f, est in states.items()}


# --- interpolate_gaps_ballistic() ---

def test_interpolate_gaps_ballistic_curves_with_acceleration():
    # true parabolic motion: y(t) = 0.5*ay*t^2 with ay=4 -> y(2) should be exactly 8,
    # not the linear interpolation's 16.
    states = {
        0: BallEstimate(pos=(0.0, 0.0), vel=(2.0, 0.0), acc=(0.0, 4.0), is_contact=False),
        4: BallEstimate(pos=(8.0, 32.0), vel=(2.0, 16.0), acc=(0.0, 4.0), is_contact=False),
    }
    filled = interpolate_gaps_ballistic(states, max_gap=15)
    assert filled[2] == pytest.approx((4.0, 8.0, True))


def test_interpolate_gaps_ballistic_reduces_to_linear_when_acc_zero():
    states = {
        0: BallEstimate(pos=(0.0, 0.0), vel=(2.0, 1.0), acc=(0.0, 0.0), is_contact=False),
        4: BallEstimate(pos=(8.0, 4.0), vel=(2.0, 1.0), acc=(0.0, 0.0), is_contact=False),
    }
    filled = interpolate_gaps_ballistic(states, max_gap=15)
    linear = interpolate_gaps({0: (0.0, 0.0), 4: (8.0, 4.0)}, max_gap=15)
    assert filled == linear


def test_interpolate_gaps_ballistic_falls_back_to_linear_on_contact():
    states = {
        0: BallEstimate(pos=(0.0, 0.0), vel=(2.0, 0.0), acc=(0.0, 4.0), is_contact=False),
        4: BallEstimate(pos=(8.0, 32.0), vel=(2.0, 16.0), acc=(0.0, 4.0), is_contact=True),
    }
    filled = interpolate_gaps_ballistic(states, max_gap=15)
    linear = interpolate_gaps({0: (0.0, 0.0), 4: (8.0, 32.0)}, max_gap=15)
    assert filled == linear
