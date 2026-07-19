"""Calibrate a pixel-to-court homography for a fixed-camera volleyball clip.

Manually click known court reference points on a representative frame, fit a
homography from clicked pixel coordinates to real-world court coordinates
(meters, standard 18m x 9m court), and save it for reuse by later phases.
Also renders a validation overlay (court lines drawn back onto the frame via
the fitted transform) so the fit can be visually sanity-checked.

The interactive click window lets you scrub to a different frame of the same
clip mid-session (e.g. if a player is standing on a point you need) - since
the camera is fixed, points already placed stay valid across frames.

Usage:
    python scripts/calibrate.py --video data/real_fotage/SLAP_SVIT_1z_upr.mp4 --frame-index 500

    # skip interactive clicking, load pixel points from a JSON file instead
    # (e.g. {"baseline_near_left": [120.0, 810.0], ...}) - useful for
    # scripted/automated sanity checks where no display/mouse is available:
    python scripts/calibrate.py --video <path> --frame-index 500 --points-json points.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

COURT_LENGTH_M = 18.0
COURT_WIDTH_M = 9.0
NET_Y_M = COURT_LENGTH_M / 2
ATTACK_LINE_M = 3.0

# Ordered list of standard court reference points a user clicks, in order:
# label -> known real-world court-space coordinate (meters). Origin (0, 0) is
# whichever baseline corner is clicked first - pick the corner nearer the
# camera for a natural click order, but the convention only has to be
# self-consistent per clip.
COURT_REFERENCE_POINTS: list[tuple[str, tuple[float, float]]] = [
    ("baseline_near_left", (0.0, 0.0)),
    ("baseline_near_right", (COURT_WIDTH_M, 0.0)),
    ("attackline_near_left", (0.0, ATTACK_LINE_M)),
    ("attackline_near_right", (COURT_WIDTH_M, ATTACK_LINE_M)),
    ("net_left", (0.0, NET_Y_M)),
    ("net_right", (COURT_WIDTH_M, NET_Y_M)),
    ("attackline_far_left", (0.0, COURT_LENGTH_M - ATTACK_LINE_M)),
    ("attackline_far_right", (COURT_WIDTH_M, COURT_LENGTH_M - ATTACK_LINE_M)),
    ("baseline_far_left", (0.0, COURT_LENGTH_M)),
    ("baseline_far_right", (COURT_WIDTH_M, COURT_LENGTH_M)),
]

# Court lines to draw for the validation overlay, as (start, end) in court-space meters.
COURT_LINES_M: list[tuple[tuple[float, float], tuple[float, float]]] = [
    ((0.0, 0.0), (0.0, COURT_LENGTH_M)),  # left sideline
    ((COURT_WIDTH_M, 0.0), (COURT_WIDTH_M, COURT_LENGTH_M)),  # right sideline
    ((0.0, 0.0), (COURT_WIDTH_M, 0.0)),  # near baseline
    ((0.0, COURT_LENGTH_M), (COURT_WIDTH_M, COURT_LENGTH_M)),  # far baseline
    ((0.0, NET_Y_M), (COURT_WIDTH_M, NET_Y_M)),  # net line
    ((0.0, ATTACK_LINE_M), (COURT_WIDTH_M, ATTACK_LINE_M)),  # near attack line
    ((0.0, COURT_LENGTH_M - ATTACK_LINE_M), (COURT_WIDTH_M, COURT_LENGTH_M - ATTACK_LINE_M)),  # far attack line
]


def extract_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_index} from {video_path}")
    return frame


FRAME_STEP_SMALL = 30  # ~1s at ~30fps
FRAME_STEP_LARGE = 300  # ~10s at ~30fps


def click_points_interactive(
    video_path: Path,
    start_frame_index: int,
    labels: list[str],
    initial_points: dict[str, tuple[float, float]] | None = None,
    display_scale: float = 1.0,
) -> dict[str, tuple[float, float]]:
    """Let the user click each labeled reference point, scrubbing between frames
    of `video_path` as needed (e.g. to dodge a player standing on a point).

    The camera is fixed, so points already placed stay valid when the displayed
    frame changes - only the background image updates when scrubbing.

    A cursor points at one label at a time (shown highlighted in yellow, vs.
    red for other placed points); clicking sets/overwrites that label's point
    and advances the cursor. This makes individual points correctable: navigate
    back to a wrong one and re-click it, no need to restart the whole session.
    `initial_points` (e.g. loaded from a prior run's *_points.json) seeds the
    session so only the wrong points need to be redone.

    `display_scale` shrinks (or grows) only the on-screen window - e.g. 0.6 on
    a 1920x1080 source shows a ~1152x648 window - while `clicked` still stores
    native-resolution pixel coordinates (divided back out of the click
    position), so the saved homography is identical regardless of what scale
    was used to place the points.

    Controls: click to place/overwrite the point at the cursor and advance;
    'b' moves the cursor back, 'f' moves it forward (also doubles as "skip"
    when the current point isn't visible) - neither changes anything, just
    moves the cursor; 'x' erases the point at the cursor; 'n'/'p' step ~1s
    forward/back through the video, 'N'/'P' step ~10s; 'q' finishes with
    whatever's placed (compute_homography just needs >=4 matched points).
    (Deliberately plain lowercase letters, not '[' ']', since those need
    AltGr on some non-US keyboard layouts and didn't register reliably.)
    """
    clicked: dict[str, tuple[float, float]] = dict(initial_points or {})
    cursor = 0
    window = "calibrate - click court points (b/f=move, x=erase, n/p=frame, q=finish)"
    state = {"index": start_frame_index, "frame": extract_frame(video_path, start_frame_index)}

    def redraw() -> np.ndarray:
        native = state["frame"]
        display = (cv2.resize(native, (int(native.shape[1] * display_scale), int(native.shape[0] * display_scale)))
                    if display_scale != 1.0 else native.copy())
        for label, point in clicked.items():
            x, y = int(point[0] * display_scale), int(point[1] * display_scale)
            color = (0, 255, 255) if label == labels[cursor] else (0, 0, 255)
            cv2.circle(display, (x, y), 5, color, -1)
            cv2.putText(display, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        current = labels[cursor]
        status = "set" if current in clicked else "not set"
        cv2.putText(display, f"point: {current} ({status})", (10, display.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, f"frame: {state['index']}", (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return display

    def on_click(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        nonlocal cursor
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked[labels[cursor]] = (x / display_scale, y / display_scale)
            cursor = min(cursor + 1, len(labels) - 1)
            cv2.imshow(window, redraw())

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    cv2.imshow(window, redraw())

    print("Click to set/overwrite the highlighted point ('b'/'f' move, 'x' erase, 'q' finish):")
    for i, label in enumerate(labels, 1):
        marker = " (already set)" if label in clicked else ""
        print(f"  {i}. {label}{marker}")

    step_keys = {
        ord("n"): FRAME_STEP_SMALL,
        ord("p"): -FRAME_STEP_SMALL,
        ord("N"): FRAME_STEP_LARGE,
        ord("P"): -FRAME_STEP_LARGE,
    }
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            cursor = max(cursor - 1, 0)
            cv2.imshow(window, redraw())
        elif key == ord("f"):
            cursor = min(cursor + 1, len(labels) - 1)
            cv2.imshow(window, redraw())
        elif key == ord("x"):
            clicked.pop(labels[cursor], None)
            cv2.imshow(window, redraw())
        elif key in step_keys:
            state["index"] = max(0, state["index"] + step_keys[key])
            state["frame"] = extract_frame(video_path, state["index"])
            cv2.imshow(window, redraw())

    cv2.destroyWindow(window)
    return clicked


def compute_homography(points_px: dict[str, tuple[float, float]]) -> np.ndarray:
    """Fit a pixel-to-court homography from clicked pixel points to their known court coords."""
    labels = [label for label, _ in COURT_REFERENCE_POINTS if label in points_px]
    if len(labels) < 4:
        raise ValueError(f"need at least 4 matched reference points, got {len(labels)}")
    court_lookup = dict(COURT_REFERENCE_POINTS)
    src = np.array([points_px[label] for label in labels], dtype=np.float64)
    dst = np.array([court_lookup[label] for label in labels], dtype=np.float64)
    # No robust method (RANSAC/LMEDS) here on purpose: every point is a manually
    # verified click, not a machine-matched candidate that might be a mismatch, so
    # there's nothing to reject as an "outlier". Verified empirically on real
    # calibration data (SLAP_SVIT_1z_upr) - the far court corners cluster tightly
    # in pixel space (severe foreshortening) while the near corner sits far away;
    # cv2.RANSAC's consensus check favored the tight cluster and silently dropped
    # the near point from the fit entirely, producing a badly wrong homography
    # despite every click being correct. Plain least-squares over all points avoids
    # that.
    homography, _ = cv2.findHomography(src, dst)
    if homography is None:
        raise RuntimeError("cv2.findHomography failed to compute a homography")
    return homography


def _apply_homography(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    vec = np.array([point[0], point[1], 1.0])
    out = homography @ vec
    out /= out[2]
    return float(out[0]), float(out[1])


def pixel_to_court(point_px: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    """Convert one (x, y) pixel-space point to (x, y) court-space meters via `homography`."""
    return _apply_homography(point_px, homography)


def court_to_pixel(point_court: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    """Convert one (x, y) court-space meters point to (x, y) pixel-space via `homography`."""
    return _apply_homography(point_court, np.linalg.inv(homography))


def save_homography(homography: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(homography.tolist(), indent=2))


def load_homography(path: Path) -> np.ndarray:
    return np.array(json.loads(path.read_text()))


def render_validation_overlay(frame: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Draw the standard court lines onto `frame` using `homography`, for visual QA."""
    overlay = frame.copy()
    for start_m, end_m in COURT_LINES_M:
        start_px = court_to_pixel(start_m, homography)
        end_px = court_to_pixel(end_m, homography)
        cv2.line(
            overlay,
            (int(round(start_px[0])), int(round(start_px[1]))),
            (int(round(end_px[0])), int(round(end_px[1]))),
            (0, 255, 0),
            2,
        )
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--points-json",
        type=Path,
        default=None,
        help="skip interactive clicking entirely; load {label: [x, y]} pixel points from this file instead",
    )
    parser.add_argument(
        "--resume-json",
        type=Path,
        default=None,
        help="pre-load {label: [x, y]} points (e.g. a prior run's *_points.json) into the "
        "interactive session so only the wrong ones need to be re-clicked, instead of starting empty",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path for the homography JSON (default: data/calibration/<video-stem>_homography.json)",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="shrink (e.g. 0.6) or grow the interactive click window if the native frame "
        "resolution doesn't fit your screen - clicked points are still stored at full "
        "native resolution regardless of this value",
    )
    args = parser.parse_args()

    if args.points_json is not None:
        raw_points = json.loads(args.points_json.read_text())
        points_px = {label: (float(xy[0]), float(xy[1])) for label, xy in raw_points.items()}
    else:
        labels = [label for label, _ in COURT_REFERENCE_POINTS]
        initial_points = None
        if args.resume_json is not None:
            raw_points = json.loads(args.resume_json.read_text())
            initial_points = {label: (float(xy[0]), float(xy[1])) for label, xy in raw_points.items()}
        points_px = click_points_interactive(args.video, args.frame_index, labels, initial_points,
                                              display_scale=args.display_scale)

    homography = compute_homography(points_px)

    out_path = args.out or Path("data/calibration") / f"{args.video.stem}_homography.json"
    save_homography(homography, out_path)
    points_path = out_path.parent / f"{out_path.stem}_points.json"
    points_path.write_text(json.dumps({label: list(xy) for label, xy in points_px.items()}, indent=2))
    print(f"Saved homography ({len(points_px)} points) to {out_path}")
    print(f"Saved raw clicked points to {points_path} (reusable via --points-json, useful for debugging a bad fit)")

    # Validation overlay always renders on the --frame-index frame (court lines are
    # frame-invariant for a fixed camera, so this doesn't need to match whatever
    # frame clicking scrubbed to).
    frame = extract_frame(args.video, args.frame_index)
    overlay = render_validation_overlay(frame, homography)
    overlay_path = out_path.parent / f"{out_path.stem}_validation.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"Saved validation overlay to {overlay_path}")


if __name__ == "__main__":
    main()
