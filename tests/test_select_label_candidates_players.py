import random

from select_label_candidates_players import clustering_scores, iou, select_low_score_negatives


def test_iou_identical_boxes_is_one():
    assert iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_clustering_scores_zero_for_well_separated_players():
    per_frame = {10: {"player": [(0.0, 0.0, 10.0, 10.0, 0.8), (500.0, 500.0, 510.0, 510.0, 0.8)]}}
    scores = clustering_scores(per_frame, range(10, 11))
    assert scores[10] == 0.0


def test_clustering_scores_high_for_overlapping_players():
    per_frame = {10: {"player": [(0.0, 0.0, 10.0, 10.0, 0.8), (2.0, 2.0, 12.0, 12.0, 0.8)]}}
    scores = clustering_scores(per_frame, range(10, 11))
    assert scores[10] > 0.0


def test_clustering_scores_ignores_low_confidence_boxes():
    per_frame = {10: {"player": [(0.0, 0.0, 10.0, 10.0, 0.1), (2.0, 2.0, 12.0, 12.0, 0.1)]}}
    scores = clustering_scores(per_frame, range(10, 11), min_conf=0.4)
    assert scores[10] == 0.0


def test_clustering_scores_single_box_is_zero():
    per_frame = {10: {"player": [(0.0, 0.0, 10.0, 10.0, 0.8)]}}
    scores = clustering_scores(per_frame, range(10, 11))
    assert scores[10] == 0.0


def test_clustering_scores_missing_frame_defaults_to_zero():
    scores = clustering_scores({}, range(10, 11))
    assert scores[10] == 0.0


def test_select_low_score_negatives_prefers_low_scores():
    scores = {f: float(f) for f in range(100)}
    rng = random.Random(0)
    chosen = select_low_score_negatives(scores, quota=5, min_spacing=3, exclude=set(), rng=rng)
    assert len(chosen) == 5
    assert all(scores[f] < 60 for f in chosen)


def test_select_low_score_negatives_respects_min_spacing():
    scores = {f: 0.0 for f in range(20)}
    rng = random.Random(0)
    chosen = select_low_score_negatives(scores, quota=10, min_spacing=5, exclude=set(), rng=rng)
    chosen_sorted = sorted(chosen)
    assert all(b - a >= 5 for a, b in zip(chosen_sorted, chosen_sorted[1:]))
