import pytest

from calibrate import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M
from derive_stats import (
    REFEREE_ZONE_PX,
    Rally,
    Touch,
    attribute_touches,
    compute_zone_occupancy,
    detect_net_crossings,
    distance_point_to_box,
    find_ball_contacts,
    find_nearest_player_by_box,
    frames_within_rallies,
    in_referee_zone,
    player_foot_point,
    segment_rallies,
    stationary_track_ids,
    zone_for_position,
)


def test_player_foot_point_is_bottom_center():
    assert player_foot_point((0.0, 10.0, 20.0, 50.0)) == (10.0, 50.0)


# --- stationary_track_ids ---

def test_stationary_track_ids_flags_long_still_track():
    # track 1 barely moves over 200 frames; track 2 moves freely.
    foot_px_by_frame = {}
    for f in range(200):
        foot_px_by_frame[f] = {1: (100.0 + (f % 3), 200.0), 2: (100.0 + f * 5.0, 200.0)}
    stationary = stationary_track_ids(foot_px_by_frame, min_frames=150, max_extent_px=150.0)
    assert stationary == {1}


def test_stationary_track_ids_ignores_short_lived_still_track():
    # track 1 is still, but only observed for 50 frames - not enough evidence.
    foot_px_by_frame = {f: {1: (100.0, 200.0)} for f in range(50)}
    stationary = stationary_track_ids(foot_px_by_frame, min_frames=150, max_extent_px=150.0)
    assert stationary == set()


def test_stationary_track_ids_no_tracks_returns_empty():
    assert stationary_track_ids({}, min_frames=150, max_extent_px=150.0) == set()


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


# --- frames_within_rallies ---

def test_frames_within_rallies_empty_list_returns_empty_set():
    assert frames_within_rallies([]) == set()


def test_frames_within_rallies_single_rally_covers_full_range():
    rallies = [Rally(start_frame=10, end_frame=13, net_crossings=2)]
    assert frames_within_rallies(rallies) == {10, 11, 12, 13}


def test_frames_within_rallies_multiple_non_overlapping_rallies():
    rallies = [
        Rally(start_frame=0, end_frame=2, net_crossings=2),
        Rally(start_frame=10, end_frame=11, net_crossings=3),
    ]
    assert frames_within_rallies(rallies) == {0, 1, 2, 10, 11}


# --- touch attribution ---

def test_distance_point_to_box_is_zero_inside_box():
    assert distance_point_to_box((5.0, 5.0), (0.0, 0.0, 10.0, 10.0)) == 0.0


def test_distance_point_to_box_measures_to_nearest_edge():
    # Point is directly above the box, 3 units clear of its top edge.
    assert distance_point_to_box((5.0, -3.0), (0.0, 0.0, 10.0, 10.0)) == pytest.approx(3.0)


def test_distance_point_to_box_measures_to_nearest_corner():
    assert distance_point_to_box((13.0, -4.0), (0.0, 0.0, 10.0, 10.0)) == pytest.approx(5.0)


def test_find_nearest_player_by_box_picks_closest():
    candidates = {1: (0.0, 0.0, 1.0, 1.0), 2: (0.8, 0.0, 1.8, 1.0), 3: (5.0, 5.0, 6.0, 6.0)}
    tid, dist = find_nearest_player_by_box((0.5, 0.5), candidates)
    assert tid == 1 and dist == pytest.approx(0.0)


def test_find_nearest_player_by_box_empty_candidates_returns_none():
    assert find_nearest_player_by_box((0.0, 0.0), {}) is None


def test_attribute_touches_matches_nearest_player_per_frame():
    ball_positions = {10: (1.0, 1.0), 20: (5.0, 5.0)}
    player_boxes_by_frame = {
        10: {1: (1.0, 1.0, 1.2, 1.2), 2: (9.0, 9.0, 9.2, 9.2)},
        20: {1: (1.0, 1.0, 1.2, 1.2), 2: (5.0, 5.0, 5.2, 5.2)},
    }
    touches = attribute_touches([10, 20], ball_positions, player_boxes_by_frame)
    assert touches[0].frame_idx == 10 and touches[0].track_id == 1
    assert touches[1].frame_idx == 20 and touches[1].track_id == 2


def test_attribute_touches_no_players_tracked_is_unattributed():
    ball_positions = {10: (1.0, 1.0)}
    touches = attribute_touches([10], ball_positions, {10: {}})
    assert touches == [Touch(frame_idx=10, track_id=None, distance_px=None)]


def test_attribute_touches_skips_contact_with_no_ball_position():
    touches = attribute_touches([10], {}, {10: {1: (0.0, 0.0, 0.0, 0.0)}})
    assert touches == []


def test_attribute_touches_rejects_implausibly_distant_nearest_player():
    # Nearest tracked player's box is 200px away - a spurious contact (e.g. dead-ball
    # ball roll) or a real touch whose actual toucher wasn't tracked that frame, not a
    # real touch by this distant player.
    ball_positions = {10: (0.0, 0.0)}
    players = {10: {1: (200.0, 0.0, 200.0, 0.0)}}
    touches = attribute_touches([10], ball_positions, players, max_distance_px=150.0)
    assert touches == [Touch(frame_idx=10, track_id=None, distance_px=200.0)]


def test_attribute_touches_accepts_nearest_player_within_max_distance():
    ball_positions = {10: (0.0, 0.0)}
    players = {10: {1: (50.0, 0.0, 50.0, 0.0)}}
    touches = attribute_touches([10], ball_positions, players, max_distance_px=150.0)
    assert touches == [Touch(frame_idx=10, track_id=1, distance_px=50.0)]


def test_attribute_touches_uses_full_box_not_just_a_single_point():
    # Ball sits inside the player's box (e.g. near their raised hands, far from their
    # foot point) - should attribute even though the box's bottom-center is far away.
    ball_positions = {10: (5.0, 2.0)}
    players = {10: {1: (0.0, 0.0, 10.0, 20.0)}}
    touches = attribute_touches([10], ball_positions, players, max_distance_px=150.0)
    assert touches == [Touch(frame_idx=10, track_id=1, distance_px=0.0)]


# --- referee zone exclusion ---

def test_in_referee_zone_true_inside_zone():
    x1, y1, x2, y2 = REFEREE_ZONE_PX
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    assert in_referee_zone(center) is True


def test_in_referee_zone_false_outside_zone():
    assert in_referee_zone((0.0, 0.0)) is False


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
