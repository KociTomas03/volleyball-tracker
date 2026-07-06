"""Extract evenly-spaced sample frames from a video clip.

Usage:
    python scripts/extract_frames.py --video data/raw/online_match_01.mp4 --count 10
"""

import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: Path, out_dir: Path, count: int, dense: bool = False) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps else 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # Evenly spaced across the whole clip, skipping the very first/last 2%
    # so we don't land on black frames/transitions.
    margin = max(1, int(total_frames * 0.02))
    indices = [
        margin + int(i * (total_frames - 2 * margin) / max(1, count - 1))
        for i in range(count)
    ]

    saved = []
    if dense:
        # Sequential decode + save-on-match, instead of repeated seeks -
        # much faster for large counts (seeking is expensive on some codecs).
        # Filename encodes the true source frame index (not a sequential
        # counter) so re-extracting the same video at a different --count
        # never collides with or silently reinterprets a previous run's files.
        wanted = sorted(set(indices))
        wanted_iter = iter(wanted)
        next_idx = next(wanted_iter, None)
        frame_idx = 0
        while next_idx is not None:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx == next_idx:
                out_path = out_dir / f"frame_{next_idx:05d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved.append(str(out_path))
                next_idx = next(wanted_iter, None)
            frame_idx += 1
    else:
        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            out_path = out_dir / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append(str(out_path))

    cap.release()

    return {
        "video": str(video_path),
        "resolution": f"{width}x{height}",
        "fps": round(fps, 2),
        "duration_sec": round(duration_sec, 1),
        "frames_saved": saved,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to data/frames/<video_stem>/",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--dense", action="store_true", help="Sequential decode, faster for large --count")
    args = parser.parse_args()

    out_dir = args.out_dir or Path("data/frames") / args.video.stem
    info = extract_frames(args.video, out_dir, args.count, dense=args.dense)

    print(f"video:      {info['video']}")
    print(f"resolution: {info['resolution']}")
    print(f"fps:        {info['fps']}")
    print(f"duration:   {info['duration_sec']}s")
    print(f"frames:     {len(info['frames_saved'])} saved to {out_dir}")


if __name__ == "__main__":
    main()
