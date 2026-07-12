import math

import pytest

from convert_boxes_to_seg_masks import box_to_ellipse_polygon, convert_label_file


def test_box_to_ellipse_polygon_point_count():
    polygon = box_to_ellipse_polygon(0.5, 0.5, 0.1, 0.2, n_points=16)
    assert len(polygon) == 16


def test_box_to_ellipse_polygon_points_within_box_bounds():
    xc, yc, w, h = 0.4, 0.6, 0.1, 0.3
    polygon = box_to_ellipse_polygon(xc, yc, w, h, n_points=32)
    for x, y in polygon:
        assert xc - w / 2 - 1e-9 <= x <= xc + w / 2 + 1e-9
        assert yc - h / 2 - 1e-9 <= y <= yc + h / 2 + 1e-9


def test_box_to_ellipse_polygon_square_box_is_circular():
    xc, yc, r = 0.5, 0.5, 0.1
    polygon = box_to_ellipse_polygon(xc, yc, 2 * r, 2 * r, n_points=16)
    for x, y in polygon:
        dist = math.dist((x, y), (xc, yc))
        assert dist == pytest.approx(r, abs=1e-9)


def test_convert_label_file_empty_stays_empty(tmp_path):
    label_path = tmp_path / "empty.txt"
    label_path.write_text("")
    assert convert_label_file(label_path) == ""


def test_convert_label_file_converts_box_to_polygon(tmp_path):
    label_path = tmp_path / "one_box.txt"
    label_path.write_text("0 0.5 0.5 0.1 0.2\n")
    result = convert_label_file(label_path)
    lines = result.strip().splitlines()
    assert len(lines) == 1
    tokens = lines[0].split()
    assert tokens[0] == "0"
    # 16 points * 2 coords = 32 floats
    assert len(tokens) == 1 + 32
    for t in tokens[1:]:
        float(t)  # doesn't raise


def test_convert_label_file_handles_multiple_boxes(tmp_path):
    label_path = tmp_path / "two_boxes.txt"
    label_path.write_text("0 0.2 0.2 0.05 0.05\n0 0.8 0.8 0.05 0.05\n")
    result = convert_label_file(label_path)
    assert len(result.strip().splitlines()) == 2
