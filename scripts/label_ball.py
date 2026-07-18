"""Interactive click-to-box tool for labeling the ball in candidate frames.

Usage:
    python scripts/label_ball.py --manifest data/annotations/label_candidates.csv

Controls:
    left-click-drag  draw a box (can draw more than one per frame)
    z                undo last box on this frame
    x                confirm as "no ball" (ignores any pending boxes)
    n / Space / Enter  confirm frame with whatever boxes are drawn (0 = no ball)
    s                skip (leave unlabeled, revisit later)
    b / Backspace    go back one frame
    q / Esc          quit
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2

DATASET_ROOT = Path("data/self_labeled")


def label_path_for(row: dict, dataset_root: Path = DATASET_ROOT) -> Path:
    stem = f"{row['clip']}_{Path(row['frame_id']).name.rsplit('.', 1)[0]}"
    return dataset_root / row["split"] / "labels" / f"{stem}.txt"


def image_path_for(row: dict, dataset_root: Path = DATASET_ROOT) -> Path:
    stem = f"{row['clip']}_{Path(row['frame_id']).name.rsplit('.', 1)[0]}"
    return dataset_root / row["split"] / "images" / f"{stem}.jpg"


def save_frame(row: dict, boxes: list[tuple[int, int, int, int]], img_w: int, img_h: int,
               dataset_root: Path = DATASET_ROOT):
    label_path = label_path_for(row, dataset_root)
    image_path = image_path_for(row, dataset_root)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(row["image_path"], image_path)

    lines = []
    for x1, y1, x2, y2 in boxes:
        xc = (x1 + x2) / 2 / img_w
        yc = (y1 + y2) / 2 / img_h
        w = abs(x2 - x1) / img_w
        h = abs(y2 - y1) / img_h
        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


class BoxDrawer:
    def __init__(self, scale: float):
        self.scale = scale
        self.boxes = []  # display-space boxes (x1,y1,x2,y2)
        self.dragging = False
        self.start = None
        self.cur = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.cur = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.cur = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x1, y1 = self.start
            x2, y2 = x, y
            if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:
                self.boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
            self.start = None
            self.cur = None

    def render(self, base_img):
        img = base_img.copy()
        for x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if self.dragging and self.start and self.cur:
            cv2.rectangle(img, self.start, self.cur, (0, 255, 255), 1)
        return img

    def native_boxes(self):
        return [(x1 / self.scale, y1 / self.scale, x2 / self.scale, y2 / self.scale) for x1, y1, x2, y2 in self.boxes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/annotations/label_candidates.csv"))
    parser.add_argument("--display-scale", type=float, default=1.5)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT,
                         help="output dataset root - override for a non-ball dataset "
                         "(e.g. data/self_labeled_players) so labels don't mix with the ball dataset")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))
    todo = [r for r in rows if not label_path_for(r, args.dataset_root).exists()]
    print(f"{len(rows)} total candidates, {len(rows) - len(todo)} already labeled, {len(todo)} remaining")

    if not todo:
        print("Nothing left to label.")
        return

    window = "label_ball  (drag=box, z=undo, x=no-ball, n/space=confirm, s=skip, b=back, q=quit)"
    cv2.namedWindow(window)

    i = 0
    while 0 <= i < len(todo):
        row = todo[i]
        img = cv2.imread(row["image_path"])
        h, w = img.shape[:2]
        disp = cv2.resize(img, (int(w * args.display_scale), int(h * args.display_scale)))

        drawer = BoxDrawer(args.display_scale)
        cv2.setMouseCallback(window, drawer.on_mouse)

        result = None
        while result is None:
            frame = drawer.render(disp)
            cv2.putText(frame, f"{i + 1}/{len(todo)}  {row['frame_id']}  ({row['selection_reason']})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(window, frame)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("z") and drawer.boxes:
                drawer.boxes.pop()
            elif key == ord("x"):
                save_frame(row, [], w, h, args.dataset_root)
                result = "next"
            elif key in (ord("n"), 13, 32):
                save_frame(row, drawer.native_boxes(), w, h, args.dataset_root)
                result = "next"
            elif key == ord("s"):
                result = "next"
            elif key in (ord("b"), 8):
                result = "back"
            elif key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return

        i += 1 if result == "next" else -1

    cv2.destroyAllWindows()
    print("Done with this batch.")


if __name__ == "__main__":
    main()
