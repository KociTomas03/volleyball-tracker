from review_touches import (
    box_contains,
    build_row,
    find_clicked_track,
    rally_index_for_frame,
)


def test_rally_index_for_frame_finds_containing_rally():
    rallies = [
        {"start_frame": 0, "end_frame": 10},
        {"start_frame": 20, "end_frame": 30},
    ]
    assert rally_index_for_frame(25, rallies) == 1


def test_rally_index_for_frame_none_when_outside_all_rallies():
    rallies = [{"start_frame": 0, "end_frame": 10}]
    assert rally_index_for_frame(15, rallies) is None


def test_box_contains_inside():
    assert box_contains((5.0, 5.0), (0.0, 0.0, 10.0, 10.0)) is True


def test_box_contains_outside():
    assert box_contains((15.0, 5.0), (0.0, 0.0, 10.0, 10.0)) is False


def test_box_contains_on_edge_counts_as_inside():
    assert box_contains((10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) is True


def test_find_clicked_track_picks_smallest_overlapping_box():
    boxes = {
        1: (0.0, 0.0, 100.0, 100.0),  # large box, also contains the click
        2: (4.0, 4.0, 6.0, 6.0),  # small box, more specific
    }
    assert find_clicked_track((5.0, 5.0), boxes) == 2


def test_find_clicked_track_no_hit_returns_none():
    boxes = {1: (0.0, 0.0, 10.0, 10.0)}
    assert find_clicked_track((50.0, 50.0), boxes) is None


def test_build_row_correct_verdict_keeps_pipeline_track_id():
    touch = {"frame_idx": 100, "track_id": 7, "distance_px": 42.5}
    row = build_row(touch, rally_idx=1, verdict="correct", correct_track_id=7)
    assert row["is_real_touch"] == "true"
    assert row["attribution_verdict"] == "correct"
    assert row["correct_track_id"] == 7
    assert row["pipeline_track_id"] == 7
    assert row["pipeline_dist_px"] == 42.5


def test_build_row_false_positive_is_not_real():
    touch = {"frame_idx": 100, "track_id": None, "distance_px": 300.0}
    row = build_row(touch, rally_idx=0, verdict="false_positive", correct_track_id=None)
    assert row["is_real_touch"] == "false"
    assert row["attribution_verdict"] == "na"
    assert row["correct_track_id"] == ""


def test_build_row_wrong_verdict_records_clicked_player():
    touch = {"frame_idx": 100, "track_id": 3, "distance_px": 80.0}
    row = build_row(touch, rally_idx=2, verdict="wrong", correct_track_id=9)
    assert row["is_real_touch"] == "true"
    assert row["attribution_verdict"] == "wrong"
    assert row["correct_track_id"] == 9
    assert row["pipeline_track_id"] == 3


def test_build_row_unattributable_leaves_correct_track_id_blank():
    touch = {"frame_idx": 100, "track_id": None, "distance_px": None}
    row = build_row(touch, rally_idx=None, verdict="unattributable", correct_track_id=None)
    assert row["is_real_touch"] == "true"
    assert row["attribution_verdict"] == "missed"
    assert row["correct_track_id"] == ""
    assert row["pipeline_track_id"] == ""
    assert row["pipeline_dist_px"] == ""
    assert row["rally_index"] == ""
