import numpy as np
import pytest

from ball_kalman import (
    BallKalmanFilter,
    build_F,
    build_Q,
    mahalanobis_sq,
    measurement_var,
)


def test_build_F_shape_and_identity_at_dt_zero():
    F = build_F(0.0)
    assert F.shape == (6, 6)
    assert np.allclose(F, np.eye(6))


def test_build_F_exact_values_at_dt_one():
    F = build_F(1.0)
    expected = np.array([
        [1, 0, 1, 0, 0.5, 0],
        [0, 1, 0, 1, 0, 0.5],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])
    assert np.allclose(F, expected)


def test_build_F_no_cross_axis_terms():
    F = build_F(3.0)
    x_idx, y_idx = [0, 2, 4], [1, 3, 5]
    assert np.allclose(F[np.ix_(x_idx, y_idx)], 0.0)
    assert np.allclose(F[np.ix_(y_idx, x_idx)], 0.0)


def test_build_Q_exact_values_at_dt_one_jerk_one():
    Q = build_Q(1.0, jerk_psd=1.0)
    assert Q[0, 0] == pytest.approx(1 / 20)
    assert Q[0, 2] == pytest.approx(1 / 8)
    assert Q[0, 4] == pytest.approx(1 / 6)
    assert Q[2, 2] == pytest.approx(1 / 3)
    assert Q[2, 4] == pytest.approx(1 / 2)
    assert Q[4, 4] == pytest.approx(1.0)


def test_build_Q_no_cross_axis_terms():
    Q = build_Q(2.0, jerk_psd=5.0)
    assert Q[0, 1] == 0.0
    assert Q[2, 3] == 0.0
    assert Q[4, 5] == 0.0


def test_build_Q_scales_linearly_with_jerk_psd():
    Q1 = build_Q(1.0, jerk_psd=1.0)
    Q10 = build_Q(1.0, jerk_psd=10.0)
    assert np.allclose(Q10, Q1 * 10)


def test_mahalanobis_sq_identity_covariance():
    y = np.array([3.0, 4.0])
    S = np.eye(2)
    assert mahalanobis_sq(y, S) == pytest.approx(25.0)


def test_mahalanobis_sq_zero_at_zero_innovation():
    assert mahalanobis_sq(np.array([0.0, 0.0]), np.eye(2) * 7) == pytest.approx(0.0)


def test_measurement_var_strictly_decreasing_in_confidence():
    confs = [0.0, 0.25, 0.5, 0.75, 1.0]
    variances = [measurement_var(c) for c in confs]
    assert all(a > b for a, b in zip(variances, variances[1:]))


def test_measurement_var_bounds():
    from ball_kalman import SIGMA_MAX_PX, SIGMA_MIN_PX
    assert measurement_var(0.0) == pytest.approx(SIGMA_MAX_PX**2)
    assert measurement_var(1.0) == pytest.approx(SIGMA_MIN_PX**2)


def test_predict_constant_velocity_zero_noise_exact():
    # jerk_psd=0 -> no process noise; a filter given the exact true velocity should
    # track a straight line exactly forever with no measurement updates at all.
    kf = BallKalmanFilter(jerk_psd=0.0)
    kf.init_state(pos=(0.0, 0.0), vel=(5.0, -2.0))
    x_pred, _ = kf.predict(dt=10.0)
    assert x_pred[0] == pytest.approx(50.0)
    assert x_pred[1] == pytest.approx(-20.0)
    assert x_pred[2] == pytest.approx(5.0)
    assert x_pred[3] == pytest.approx(-2.0)


def test_update_recovers_exact_position_despite_wrong_velocity_prior():
    # Zero measurement noise: even starting from a deliberately wrong velocity guess,
    # each update should snap the position estimate exactly onto the measurement -
    # a direct, noise-free observation of a state must drive its posterior variance
    # (and hence its estimate) to exactly match the measurement, regardless of any
    # correlated uncertainty in the unobserved velocity/acceleration states.
    kf = BallKalmanFilter(jerk_psd=0.0, sigma_min=0.0, sigma_max=0.0)
    kf.init_state(pos=(0.0, 0.0), vel=(0.0, 0.0))  # wrong on purpose - true velocity is (5, 2)
    for t in range(1, 6):
        true_pos = (5.0 * t, 2.0 * t)
        x_pred, P_pred = kf.predict(dt=1.0)
        kf.update(x_pred, P_pred, true_pos, conf=1.0)
        assert kf.position() == pytest.approx(true_pos, abs=1e-6)


def test_update_converges_to_true_constant_acceleration():
    # A ball under constant "gravity"-like acceleration in pixel space (ay=10), zero
    # noise. Even without giving the filter the true acceleration up front (init_state
    # always starts accel at 0), a handful of updates along the true parabola should
    # converge the acceleration estimate close to the true value.
    kf = BallKalmanFilter(jerk_psd=0.0, sigma_min=0.0, sigma_max=0.0)
    true_vx, true_vy, true_ay = 3.0, -8.0, 10.0

    def true_pos(t):
        return (true_vx * t, true_vy * t + 0.5 * true_ay * t * t)

    kf.init_state(pos=true_pos(0), vel=(true_vx, true_vy))
    for t in range(1, 6):
        x_pred, P_pred = kf.predict(dt=1.0)
        kf.update(x_pred, P_pred, true_pos(t), conf=1.0)

    ax, ay = kf.acceleration()
    assert ax == pytest.approx(0.0, abs=1e-3)
    assert ay == pytest.approx(true_ay, abs=1e-2)


def test_update_with_tiny_measurement_noise_snaps_to_measurement():
    kf = BallKalmanFilter(jerk_psd=1.0, sigma_min=1e-6, sigma_max=1e-6)
    kf.init_state(pos=(0.0, 0.0), vel=(1.0, 1.0))
    x_pred, P_pred = kf.predict(dt=1.0)
    far_z = (500.0, 500.0)
    kf.update(x_pred, P_pred, far_z, conf=1.0)
    assert kf.position() == pytest.approx(far_z, abs=1e-2)


def test_update_with_huge_measurement_noise_stays_near_prediction():
    kf = BallKalmanFilter(jerk_psd=1.0, sigma_min=1e6, sigma_max=1e6)
    kf.init_state(pos=(0.0, 0.0), vel=(1.0, 1.0))
    x_pred, P_pred = kf.predict(dt=1.0)
    far_z = (500.0, 500.0)
    kf.update(x_pred, P_pred, far_z, conf=1.0)
    assert kf.position() == pytest.approx((x_pred[0], x_pred[1]), abs=1e-2)


def test_reinit_after_contact_replaces_velocity_estimate():
    kf = BallKalmanFilter()
    kf.init_state(pos=(0.0, 0.0), vel=(10.0, 0.0))
    kf.reinit_after_contact(pos=(50.0, 50.0), vel=(-5.0, 20.0))
    assert kf.position() == pytest.approx((50.0, 50.0))
    assert kf.velocity() == pytest.approx((-5.0, 20.0))
    assert kf.acceleration() == pytest.approx((0.0, 0.0))
