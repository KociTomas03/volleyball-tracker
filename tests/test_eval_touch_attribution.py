from eval_touch_attribution import GroundTruthRow, compare_touches


def gt_row(frame_idx: int, is_real_touch: bool, correct_track_id: int | None = None) -> GroundTruthRow:
    return GroundTruthRow(frame_idx=frame_idx, is_real_touch=is_real_touch, correct_track_id=correct_track_id)


def test_correct_attribution_counted():
    ground_truth = {10: gt_row(10, True, correct_track_id=5)}
    result = compare_touches([{"frame_idx": 10, "track_id": 5}], ground_truth)
    assert result.contacts_true_positive == 1
    assert result.attribution_correct == 1
    assert result.attribution_wrong == 0
    assert result.contact_precision == 1.0
    assert result.attribution_accuracy == 1.0


def test_wrong_attribution_counted():
    ground_truth = {10: gt_row(10, True, correct_track_id=5)}
    result = compare_touches([{"frame_idx": 10, "track_id": 7}], ground_truth)
    assert result.contacts_true_positive == 1
    assert result.attribution_wrong == 1
    assert result.attribution_correct == 0
    assert result.attribution_accuracy == 0.0


def test_missed_attribution_counted_separately_from_wrong():
    ground_truth = {10: gt_row(10, True, correct_track_id=5)}
    result = compare_touches([{"frame_idx": 10, "track_id": None}], ground_truth)
    assert result.attribution_none == 1
    assert result.attribution_wrong == 0
    assert result.attribution_correct == 0


def test_unknown_correct_id_not_scored_as_wrong():
    ground_truth = {10: gt_row(10, True, correct_track_id=None)}
    result = compare_touches([{"frame_idx": 10, "track_id": 7}], ground_truth)
    assert result.attribution_unknown == 1
    assert result.attribution_wrong == 0
    assert result.attribution_correct == 0
    assert result.attribution_accuracy is None


def test_false_positive_contact_counted():
    ground_truth = {10: gt_row(10, is_real_touch=False)}
    result = compare_touches([{"frame_idx": 10, "track_id": 5}], ground_truth)
    assert result.contacts_false_positive == 1
    assert result.contacts_true_positive == 0
    assert result.contact_precision == 0.0


def test_false_negative_when_pipeline_misses_real_touch_entirely():
    ground_truth = {10: gt_row(10, True, correct_track_id=5)}
    result = compare_touches([], ground_truth)
    assert result.contacts_false_negative == 1
    assert result.contact_recall == 0.0


def test_out_of_scope_touch_not_scored():
    ground_truth = {10: gt_row(10, True, correct_track_id=5)}
    result = compare_touches([{"frame_idx": 999, "track_id": 1}], ground_truth)
    assert result.contacts_out_of_scope == 1
    assert result.contacts_true_positive == 0
    assert result.contacts_false_positive == 0
    # the one ground-truth real touch was never even attempted -> false negative
    assert result.contacts_false_negative == 1


def test_contact_precision_and_recall_none_when_no_data():
    result = compare_touches([], {})
    assert result.contact_precision is None
    assert result.contact_recall is None
    assert result.attribution_accuracy is None


def test_mixed_batch_matches_rally5_style_scenario():
    ground_truth = {
        100: gt_row(100, True, correct_track_id=1),
        105: gt_row(105, False),
        110: gt_row(110, True, correct_track_id=2),
        115: gt_row(115, True, correct_track_id=3),
    }
    pipeline_touches = [
        {"frame_idx": 100, "track_id": 1},   # correct
        {"frame_idx": 105, "track_id": 9},   # false positive contact
        {"frame_idx": 110, "track_id": None},  # missed attribution (veto pattern)
        # 115 never flagged by the pipeline at all -> false negative
    ]
    result = compare_touches(pipeline_touches, ground_truth)
    assert result.contacts_true_positive == 2
    assert result.contacts_false_positive == 1
    assert result.contacts_false_negative == 1
    assert result.attribution_correct == 1
    assert result.attribution_none == 1
