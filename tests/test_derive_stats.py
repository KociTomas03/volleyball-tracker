import pytest

from calibrate import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M
from derive_stats import (
    Rally,
    Touch,
    attribute_touches,
    compute_zone_occupancy,
    detect_net_crossings,
    find_ball_contacts,
    find_nearest_player,
    player_foot_point,
    segment_rallies,
    zone_for_position,
)


def test_player_foot_point_is_bottom_center():
    assert player_foot_point((0.0, 10.0, 20.0, 50.0)) == (10.0, 50.0)


# --- find_ball_contacts ---

def _parabola_y(center: int, width: int, peak: float) -> dict[int, float]:
    """Synthetic ball trajectory: pixel-y rises to `peak` at `center` (screen-low =
    physically low = a contact) and falls off on both sides, matching the shape a real
    dig/set/serve produces."""
    return {f: peak - (f - center) ** 2 for f in range(center - width, center + width + 1)}


def test_find_ball_contacts_detects_single_peak():
    y_by_frame = _parabola_y(center=20, width=10, peak=100.0)
    contacts = find_ball_contacts(y_by_frame, window=5, min_excursion_px=15.0)
    assert contacts == [20]


def test_find_ball_contacts_ignores_small_jitter():
    # Flat trajectory with a tiny 2px wobble - well under the excursion floor.
    y_by_frame = {f: 50.0 for f in range(40)}
    y_by_frame[20] = 52.0
    contacts = find_ball_contacts(y_by_frame, window=5, min_excursion_px=15.0)
    assert contacts == []


def test_find_ball_contacts_skips_candidate_without_full_window():
    # Peak sits right at the edge of the available data - no full +/-window neighborhood.
    y_by_frame = _parabola_y(center=3, width=3, peak=100.0)  # frames 0..6
    contacts = find_ball_contacts(y_by_frame, window=5, min_excursion_px=15.0)
    assert contacts == []


def test_find_ball_contacts_keeps_first_frame_of_plateau():
    y_by_frame = _parabola_y(center=20, width=10, peak=100.0)
    y_by_frame[21] = y_by_frame[20]  # flat-top plateau, both frames tied for the max
    contacts = find_ball_contacts(y_by_frame, window=5, min_excursion_px=15.0)
    assert contacts == [20]


def test_find_ball_contacts_separates_two_distant_peaks():
    y_by_frame = {}
    y_by_frame.update(_parabola_y(center=10, width=8, peak=100.0))
    y_by_frame.update(_parabola_y(center=40, width=8, peak=90.0))
    contacts = find_ball_contacts(y_by_frame, window=5, min_excursion_px=15.0)
    assert contacts == [10, 40]


# --- detect_net_crossings ---

def test_detect_net_crossings_finds_sign_change_in_bounds():
    positions = {0: (4.0, NET_Y_M - 2.0), 1: (4.0, NET_Y_M + 2.0)}
    assert detect_net_crossings(positions, NET_Y_M, x_bounds=(0.0, COURT_WIDTH_M)) == [1]


def test_detect_net_crossings_ignores_out_of_x_bounds():
    positions = {0: (-5.0, NET_Y_M - 2.0), 1: (-5.0, NET_Y_M + 2.0)}
    assert detect_net_crossings(positions, NET_Y_M, x_bounds=(0.0, COURT_WIDTH_M)) == []


def test_detect_net_crossings_ignores_non_adjacent_frames():
    # Same sign change, but frames aren't consecutive (a tracking gap) - not a real crossing.
    positions = {0: (4.0, NET_Y_M - 2.0), 10: (4.0, NET_Y_M + 2.0)}
    assert detect_net_crossings(positions, NET_Y_M, x_bounds=(0.0, COURT_WIDTH_M)) == []


def test_detect_net_crossings_ignores_no_crossing():
    positions = {0: (4.0, NET_Y_M - 2.0), 1: (4.0, NET_Y_M - 1.0)}
    assert detect_net_crossings(positions, NET_Y_M, x_bounds=(0.0, COURT_WIDTH_M)) == []


def test_detect_net_crossings_finds_multiple_back_and_forth():
    positions = {
        0: (4.0, NET_Y_M - 2.0), 1: (4.0, NET_Y_M + 2.0),  # serve crosses
        2: (4.0, NET_Y_M + 1.0), 3: (4.0, NET_Y_M - 1.0),  # return crosses back
    }
    assert detect_net_crossings(positions, NET_Y_M, x_bounds=(0.0, COURT_WIDTH_M)) == [1, 3]


# --- segment_rallies ---

def _dense_positions(frames: range, x: float = 4.0, y: float = NET_Y_M) -> dict[int, tuple[float, float]]:
    return {f: (x, y) for f in frames}


def test_segment_rallies_keeps_segment_with_enough_crossings():
    positions = _dense_positions(range(0, 20))
    rallies = segment_rallies(positions, net_crossings=[5, 10], max_dead_gap_frames=10, min_crossings=2)
    assert rallies == [Rally(start_frame=0, end_frame=19, net_crossings=2)]


def test_segment_rallies_drops_segment_with_too_few_crossings():
    positions = _dense_positions(range(0, 20))
    rallies = segment_rallies(positions, net_crossings=[5], max_dead_gap_frames=10, min_crossings=2)
    assert rallies == []


def test_segment_rallies_splits_on_long_gap():
    positions = {**_dense_positions(range(0, 10)), **_dense_positions(range(200, 210))}
    rallies = segment_rallies(positions, net_crossings=[2, 5, 202, 205], max_dead_gap_frames=10, min_crossings=2)
    assert rallies == [
        Rally(start_frame=0, end_frame=9, net_crossings=2),
        Rally(start_frame=200, end_frame=209, net_crossings=2),
    ]


def test_segment_rallies_treats_sustained_out_of_bounds_as_a_gap():
    in_bounds_a = _dense_positions(range(0, 10))
    out_of_bounds = _dense_positions(range(10, 30), x=-100.0, y=-100.0)  # far outside court+margin
    in_bounds_b = _dense_positions(range(30, 40))
    positions = {**in_bounds_a, **out_of_bounds, **in_bounds_b}
    rallies = segment_rallies(positions, net_crossings=[2, 5, 32, 35],
                               max_dead_gap_frames=10, min_crossings=2, out_of_bounds_margin_m=3.0)
    assert rallies == [
        Rally(start_frame=0, end_frame=9, net_crossings=2),
        Rally(start_frame=30, end_frame=39, net_crossings=2),
    ]


def test_segment_rallies_no_positions_returns_empty():
    assert segment_rallies({}, net_crossings=[], max_dead_gap_frames=10, min_crossings=2) == []


# --- touch attribution ---

def test_find_nearest_player_picks_closest():
    candidates = {1: (0.0, 0.0), 2: (0.8, 0.0), 3: (5.0, 5.0)}
    assert find_nearest_player((0.5, 0.0), candidates) == (2, pytest.approx(0.3))


def test_find_nearest_player_empty_candidates_returns_none():
    assert find_nearest_player((0.0, 0.0), {}) is None


def test_attribute_touches_matches_nearest_player_per_frame():
    ball_positions = {10: (1.0, 1.0), 20: (5.0, 5.0)}
    players_by_frame = {
        10: {1: (1.1, 1.0), 2: (9.0, 9.0)},
        20: {1: (1.1, 1.0), 2: (5.1, 5.0)},
    }
    touches = attribute_touches([10, 20], ball_positions, players_by_frame)
    assert touches[0].frame_idx == 10 and touches[0].track_id == 1
    assert touches[1].frame_idx == 20 and touches[1].track_id == 2


def test_attribute_touches_no_players_tracked_is_unattributed():
    ball_positions = {10: (1.0, 1.0)}
    touches = attribute_touches([10], ball_positions, {10: {}})
    assert touches == [Touch(frame_idx=10, track_id=None, distance_m=None)]


def test_attribute_touches_skips_contact_with_no_ball_position():
    touches = attribute_touches([10], {}, {10: {1: (0.0, 0.0)}})
    assert touches == []


# --- zone bucketing ---

@pytest.mark.parametrize("point, expected_zone", [
    ((0.5, 0.5), 5),                                     # near back-left
    ((4.5, 4.5), 6),                                     # near back-mid
    ((8.5, 0.5), 1),                                     # near back-right
    ((0.5, NET_Y_M - 0.5), 4),                            # near front-left
    ((4.5, NET_Y_M - 0.5), 3),                            # near front-mid
    ((8.5, NET_Y_M - 0.5), 2),                            # near front-right
    ((8.5, NET_Y_M + 0.5), 4),                            # far front-left (mirrored)
    ((4.5, NET_Y_M + 0.5), 3),                            # far front-mid
    ((0.5, NET_Y_M + 0.5), 2),                            # far front-right (mirrored)
    ((8.5, COURT_LENGTH_M - 0.5), 5),                     # far back-left (mirrored)
    ((4.5, COURT_LENGTH_M - 0.5), 6),                     # far back-mid
    ((0.5, COURT_LENGTH_M - 0.5), 1),                     # far back-right (mirrored)
])
def test_zone_for_position_matches_standard_layout(point, expected_zone):
    assert zone_for_position(point) == expected_zone


def test_zone_for_position_clamps_out_of_court_points():
    # Slightly outside the court (click/detection noise) shouldn't raise or wrap around.
    assert zone_for_position((-0.1, -0.1)) == 5
    # High x, high y: the far team's mirrored "left" back corner -> zone 5, not 1.
    assert zone_for_position((COURT_WIDTH_M + 0.1, COURT_LENGTH_M + 0.1)) == 5


def test_compute_zone_occupancy_counts_frames_per_track_per_zone():
    positions_by_frame = {
        0: {1: (0.5, 0.5), 2: (8.5, 0.5)},   # zone 5, zone 1
        1: {1: (0.5, 0.5), 2: (8.5, 0.5)},   # zone 5, zone 1
        2: {1: (4.5, 4.5)},                  # zone 6
    }
    occupancy = compute_zone_occupancy(positions_by_frame)
    assert occupancy == {1: {5: 2, 6: 1}, 2: {1: 2}}
