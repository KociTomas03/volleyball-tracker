"""Pure Kalman-filter math for trajectory-aware ball tracking.

A 6-state constant-acceleration filter in raw pixel space (no court calibration exists
yet - see plan.md Phase 4 - so this is deliberately empirical: it estimates whatever
acceleration the ball's apparent pixel motion shows, rather than assuming a real-world
gravity constant). State order is [px, py, vx, vy, ax, ay]; x-axis lives at indices
(0, 2, 4), y-axis at (1, 3, 5), and the two axes never interact (no cross terms in F or Q)
since horizontal and vertical pixel motion are independent under this simple model.

No new dependency: hand-rolled with plain numpy rather than scipy/filterpy, matching the
project's minimal-dependency convention - the matrix math involved is small.

See scripts/track_video.py's track_ball_states() for how this is used per-frame
(predict -> gate candidates -> select -> update, with a contact/direction-change path via
reinit_after_contact()).
"""

import numpy as np

# px - localization noise for a high-confidence detection (the ball is ~15-20px across
# at 1280px width per detect_frame.py's BALL_IMGSZ comment, so a few px of centroid
# jitter is a reasonable floor).
SIGMA_MIN_PX = 3.0
# px - localization noise for a low-confidence/noisy detection.
SIGMA_MAX_PX = 30.0
# Process noise (jerk power spectral density) - how much acceleration may drift between
# frames. A reasonable default, not swept: the current footage is disposable placeholder
# data (different camera/ball/court than the eventual real footage), so this is scoped as
# "sound default", not a tuning target - see plan discussion.
JERK_PSD = 50.0
# Chi-square 99% threshold at 2 degrees of freedom - a candidate is trajectory-consistent
# if its Mahalanobis distance to the predicted position falls within this.
GATE_CHI2 = 9.21
# px^2 - initial position variance at track seed/reinit: the seed detection itself is
# trusted tightly (a few px std).
INIT_POS_VAR = 4.0
# (px/frame)^2 - bounds initial velocity uncertainty at seed time. Originally set to the
# old heuristic tracker's hard_speed cap (220), but verification against real detection
# data (online_match_01, first ~20 frames after a seed) showed that's too generous: right
# after a seed the filter doesn't yet know the true velocity, so a large INIT_VEL_VAR
# inflates predicted-position uncertainty enormously for the next frame or two, letting
# low-confidence noise (unrelated candidates elsewhere in frame) pass the Mahalanobis gate
# as "trajectory-consistent" before any real trajectory exists to be consistent with -
# produced several 500-900px single-frame jumps. Tightened to the old soft_speed cap (100,
# "normal in-play motion") instead - still generous enough to accept a genuinely fast
# ball at the moment of seeding, but no longer wide enough to wave through arbitrary noise.
INIT_VEL_VAR = 100.0**2
# (px/frame^2)^2 - generous initial acceleration uncertainty; converges quickly once a
# few real detections arrive.
INIT_ACC_VAR = 100.0**2

# Measurement matrix - observes position only (px, py), not velocity/acceleration.
H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

_X_AXIS_IDX = [0, 2, 4]  # px, vx, ax
_Y_AXIS_IDX = [1, 3, 5]  # py, vy, ay


def build_F(dt: float) -> np.ndarray:
    """Constant-acceleration state transition matrix for a step of dt frames."""
    block = np.array([[1.0, dt, 0.5 * dt**2],
                       [0.0, 1.0, dt],
                       [0.0, 0.0, 1.0]])
    F = np.zeros((6, 6))
    F[np.ix_(_X_AXIS_IDX, _X_AXIS_IDX)] = block
    F[np.ix_(_Y_AXIS_IDX, _Y_AXIS_IDX)] = block
    return F


def build_Q(dt: float, jerk_psd: float) -> np.ndarray:
    """Process noise for a step of dt frames, from the discretized white-noise-jerk
    model (each axis's acceleration is driven by white noise in its derivative, jerk)."""
    block = jerk_psd * np.array([
        [dt**5 / 20, dt**4 / 8, dt**3 / 6],
        [dt**4 / 8, dt**3 / 3, dt**2 / 2],
        [dt**3 / 6, dt**2 / 2, dt],
    ])
    Q = np.zeros((6, 6))
    Q[np.ix_(_X_AXIS_IDX, _X_AXIS_IDX)] = block
    Q[np.ix_(_Y_AXIS_IDX, _Y_AXIS_IDX)] = block
    return Q


def measurement_var(conf: float, sigma_min: float = SIGMA_MIN_PX, sigma_max: float = SIGMA_MAX_PX) -> float:
    """Measurement noise variance for a detection of the given confidence - higher
    confidence means lower assumed localization noise, so it pulls the filter harder."""
    conf = min(max(conf, 0.0), 1.0)
    sigma = sigma_min + (sigma_max - sigma_min) * (1 - conf)
    return sigma**2


def inv2x2(M: np.ndarray) -> np.ndarray:
    """Closed-form inverse of a 2x2 matrix (avoids a general solver for the one place
    we need it - the innovation covariance is always 2x2 since we only observe position)."""
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    det = a * d - b * c
    return (1.0 / det) * np.array([[d, -b], [-c, a]])


def innovation(x_pred: np.ndarray, P_pred: np.ndarray, z: tuple[float, float],
               R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Measurement residual y and its covariance S for candidate position z."""
    y = np.array(z) - H @ x_pred
    S = H @ P_pred @ H.T + R
    return y, S


def mahalanobis_sq(y: np.ndarray, S: np.ndarray) -> float:
    """Squared Mahalanobis distance of innovation y under covariance S."""
    return float(y @ inv2x2(S) @ y)


class BallKalmanFilter:
    """Single-object constant-acceleration Kalman filter over one ball trajectory
    segment. init_state()/reinit_after_contact() (re)anchor the filter; predict() is
    non-mutating (a rejected candidate leaves the committed state untouched, matching
    the pure per-frame loop style in track_ball_states()); update() commits."""

    def __init__(self, jerk_psd: float = JERK_PSD, sigma_min: float = SIGMA_MIN_PX,
                 sigma_max: float = SIGMA_MAX_PX, init_vel_var: float = INIT_VEL_VAR,
                 init_acc_var: float = INIT_ACC_VAR, init_pos_var: float = INIT_POS_VAR):
        self.jerk_psd = jerk_psd
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.init_vel_var = init_vel_var
        self.init_acc_var = init_acc_var
        self.init_pos_var = init_pos_var
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None

    def init_state(self, pos: tuple[float, float], vel: tuple[float, float] = (0.0, 0.0)) -> None:
        px, py = pos
        vx, vy = vel
        self.x = np.array([px, py, vx, vy, 0.0, 0.0])
        self.P = np.diag([self.init_pos_var, self.init_pos_var,
                           self.init_vel_var, self.init_vel_var,
                           self.init_acc_var, self.init_acc_var]).astype(float)

    def reinit_after_contact(self, pos: tuple[float, float], vel: tuple[float, float]) -> None:
        """A confirmed trajectory-inconsistent jump: the ball's velocity actually
        changed (serve/spike/set/dig), so the stale estimate is discarded and
        re-seeded from the jump, same as a fresh init_state() but keeping the filter
        in its tracking role rather than dropping to SEARCHING."""
        self.init_state(pos, vel)

    def predict(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = build_F(dt)
        Q = build_Q(dt, self.jerk_psd)
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q
        return x_pred, P_pred

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, z: tuple[float, float], conf: float) -> None:
        R = measurement_var(conf, self.sigma_min, self.sigma_max) * np.eye(2)
        y, S = innovation(x_pred, P_pred, z, R)
        K = P_pred @ H.T @ inv2x2(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(6) - K @ H) @ P_pred

    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    def acceleration(self) -> tuple[float, float]:
        return float(self.x[4]), float(self.x[5])
