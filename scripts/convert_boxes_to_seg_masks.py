"""Convert existing box-format ball labels into approximate segmentation masks.

Experiment: does mask supervision (a denser per-pixel training signal than a box)
teach the ball detector better shape-discriminative features, since a box only
classifies a region as a unit while a mask has to match a precise boundary? A
volleyball is round, so each existing trusted bounding box is converted into an
inscribed ellipse polygon - no new manual mask-drawing needed for this first pass.

Output lands in a fully separate data/self_labeled_seg/ tree (images copied,
labels converted) so this can't corrupt data/self_labeled/, which train_ball.py's
production (box) detector still trains on.

Usage:
    python scripts/convert_boxes_to_seg_masks.py
"""

import argparse
import math
import shutil
from pathlib import Path

SRC_ROOT = Path("data/self_labeled")
DST_ROOT = Path("data/self_labeled_seg")
ELLIPSE_POINTS = 16


def box_to_ellipse_polygon(xc: float, yc: float, w: float, h: float,
                            n_points: int = ELLIPSE_POINTS) -> list[tuple[float, float]]:
    """n_points around the ellipse inscribed in a box centered at (xc, yc) with
    width w, height h - same normalized 0-1 coordinate space as the source box."""
    rx, ry = w / 2, h / 2
    points = []
    for i in range(n_points):
        theta = 2 * math.pi * i / n_points
        points.append((xc + rx * math.cos(theta), yc + ry * math.sin(theta)))
    return points


def convert_label_file(src_path: Path) -> str:
    """Reads a YOLO-detect label file (`0 xc yc w h` per line, or empty for
    no-ball) and returns the equivalent YOLO-seg content (`0 x1 y1 x2 y2 ...`
    per line, or empty)."""
    content = src_path.read_text().strip()
    if not content:
        return ""
    lines = []
    for line in content.splitlines():
        cls, xc, yc, w, h = line.split()
        polygon = box_to_ellipse_polygon(float(xc), float(yc), float(w), float(h))
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
        lines.append(f"{cls} {coords}")
    return "\n".join(lines) + "\n"


def convert_dataset(src_root: Path = SRC_ROOT, dst_root: Path = DST_ROOT,
                     train_dirname: str = "train", val_dirname: str = "val") -> dict[str, int]:
    """Converts `src_root/{train_dirname,val_dirname}` into `dst_root/{train,val}`
    (destination split names always normalized to train/val regardless of the
    source's naming - e.g. the public Roboflow datasets use `valid`, not `val` -
    so every converted seg dataset has a consistent shape for train_ball_seg.py's
    combined dataset config to point at)."""
    counts = {}
    for src_split, dst_split in [(train_dirname, "train"), (val_dirname, "val")]:
        src_labels = src_root / src_split / "labels"
        src_images = src_root / src_split / "images"
        dst_labels = dst_root / dst_split / "labels"
        dst_images = dst_root / dst_split / "images"
        dst_labels.mkdir(parents=True, exist_ok=True)
        dst_images.mkdir(parents=True, exist_ok=True)

        n = 0
        for label_path in sorted(src_labels.glob("*.txt")):
            stem = label_path.stem
            image_path = src_images / f"{stem}.jpg"
            if not image_path.exists():
                continue
            (dst_labels / f"{stem}.txt").write_text(convert_label_file(label_path))
            shutil.copy(image_path, dst_images / f"{stem}.jpg")
            n += 1
        counts[dst_split] = n
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-root", type=Path, default=SRC_ROOT)
    parser.add_argument("--dst-root", type=Path, default=DST_ROOT)
    parser.add_argument("--train-dirname", default="train",
                         help="Source split directory name for train (default: %(default)s)")
    parser.add_argument("--val-dirname", default="val",
                         help="Source split directory name for val - the public Roboflow "
                              "datasets use 'valid' instead (default: %(default)s)")
    args = parser.parse_args()

    counts = convert_dataset(args.src_root, args.dst_root, args.train_dirname, args.val_dirname)
    for split, n in counts.items():
        print(f"{split}: {n} frames -> {args.dst_root / split}")


if __name__ == "__main__":
    main()
