"""Run player + ball detection across every frame of a video and cache results.

Unlike scan_ball_dense.py (ball only, sparse sampled frames), this runs both
detectors across every sequential frame of a clip and keeps class labels, so
track_video.py can build per-class detection streams for tracking.

Usage:
    python scripts/detect_video.py --video data/raw/online_match_01.mp4
    python scripts/detect_video.py --video data/raw/online_match_01.mp4 --max-frames 500
"""

import argparse
import csv
from pathlib import Path

import cv2

from detect_frame import detect_ball, detect_players


def detect_video(video_path: Path, out_csv: Path, max_frames: int | None = None, start_frame: int = 0,
                  min_conf: float = 0.1) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {video_path} ({total} frames, {fps:.1f} fps)")

    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    limit = min(max_frames, total - start_frame) if max_frames else total - start_frame

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame_idx = start_frame
    n_written = 0
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_idx", "class", "x1", "y1", "x2", "y2", "confidence"])
        while n_written < limit:
            ok, frame = cap.read()
            if not ok:
                break
            for d in detect_players(frame, min_conf=min_conf) + detect_ball(frame, min_conf=min_conf):
                x1, y1, x2, y2 = d["bbox"]
                writer.writerow([frame_idx, d["class"], x1, y1, x2, y2, d["confidence"]])
            frame_idx += 1
            n_written += 1
            if n_written % 200 == 0:
                print(f"  {n_written}/{limit} frames processed")

    cap.release()
    print(f"done. {n_written} frames -> {out_csv}")
    return n_written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Limit for fast iteration (default: whole clip)")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--min-conf", type=float, default=0.1,
                         help="Confidence floor for the cache - deliberately lower than the production "
                              "PLAYER_CONF/BALL_CONF thresholds so ByteTrack's own two-tier association "
                              "(low-confidence detections can maintain an existing track) has data to work with")
    args = parser.parse_args()

    out_csv = args.out_csv or Path("data/annotations/video_detections") / f"{args.video.stem}.csv"
    detect_video(args.video, out_csv, max_frames=args.max_frames, start_frame=args.start_frame, min_conf=args.min_conf)


if __name__ == "__main__":
    main()
