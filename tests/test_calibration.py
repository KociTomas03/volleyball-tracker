import numpy as np
import pytest

from calibrate import (
    COURT_REFERENCE_POINTS,
    compute_homography,
    court_to_pixel,
    load_homography,
    pixel_to_court,
    save_homography,
)


def _synthetic_homography() -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    # Four real reference-point labels mapped to an arbitrary synthetic pixel
    # quadrilateral, so compute_homography's label-matching logic is exercised
    # alongside the projective math itself.
    points_px = {
        "baseline_near_left": (100.0, 500.0),
        "baseline_near_right": (900.0, 520.0),
        "baseline_far_left": (300.0, 100.0),
        "baseline_far_right": (700.0, 110.0),
    }
    homography = compute_homography(points_px)
    return homography, points_px


def test_compute_homography_requires_at_least_four_points():
    with pytest.raises(ValueError):
        compute_homography({"baseline_near_left": (0.0, 0.0), "baseline_near_right": (1.0, 0.0)})


def test_compute_homography_ignores_unknown_labels():
    points_px = {
        "baseline_near_left": (100.0, 500.0),
        "baseline_near_right": (900.0, 520.0),
        "baseline_far_left": (300.0, 100.0),
        "baseline_far_right": (700.0, 110.0),
        "not_a_real_point": (42.0, 42.0),
    }
    # Should compute successfully using only the 4 recognized labels, not raise/include the extra one.
    homography = compute_homography(points_px)
    assert homography.shape == (3, 3)


def test_pixel_to_court_recovers_known_reference_points():
    homography, points_px = _synthetic_homography()
    court_lookup = dict(COURT_REFERENCE_POINTS)
    for label, point_px in points_px.items():
        result = pixel_to_court(point_px, homography)
        expected = court_lookup[label]
        assert result[0] == pytest.approx(expected[0], abs=1e-6)
        assert result[1] == pytest.approx(expected[1], abs=1e-6)


def test_court_to_pixel_round_trip():
    homography, points_px = _synthetic_homography()
    for point_px in points_px.values():
        point_court = pixel_to_court(point_px, homography)
        recovered_px = court_to_pixel(point_court, homography)
        assert recovered_px[0] == pytest.approx(point_px[0], abs=1e-6)
        assert recovered_px[1] == pytest.approx(point_px[1], abs=1e-6)


def test_save_and_load_homography_round_trip(tmp_path):
    homography, _ = _synthetic_homography()
    path = tmp_path / "h.json"
    save_homography(homography, path)
    loaded = load_homography(path)
    assert np.allclose(loaded, homography)
