"""Train a YOLOv8n-seg ball detector on the ellipse-mask-converted combined dataset.

Deliberately separate from train_ball.py, which trains the box-format detector on a
different set of public datasets - this trains a segmentation model instead (see
scripts/convert_boxes_to_seg_masks.py for how box labels were converted to mask
polygons). A first small-scale pass (self-labeled data only, 737 images) showed the
segmentation model tracks the ball substantially better than the box model during
active play; this combined run adds public-dataset diversity to see if it also fixes
the segmentation model's remaining weak spot (confusing shoes/hands for the ball
during dead-ball periods).

Public datasets here are NOT the same 3 train_ball.py uses - one of those
(volleyballtracking-gouow, ~24k images) turned out on inspection to be a single
low-diversity CCTV-style video of solo practice, not real match footage, and would
have dominated this combined set by sheer volume. Replaced with 3 smaller, vetted,
genuinely diverse real-match-footage datasets found on Roboflow Universe instead
(qc/volleyball-hwxp2, volleyball-training/volley-ball-unwlq, and
salo-levy-nlqrn/volley-ball-detection - the last being beach volleyball, a different
sub-domain but still real ball/real play, kept for the additional diversity).

Usage:
    python scripts/train_ball_seg.py
    python scripts/train_ball_seg.py --epochs 20 --name seg_v2_combined
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASETS = [
    REPO_ROOT / "data" / "self_labeled_seg",
    REPO_ROOT / "data" / "roboflow_ball_seg" / "aivolleyballref_volleyball_detection_2",
    REPO_ROOT / "data" / "roboflow_ball_seg" / "primaryws_volleyball_ball_object_detection_dataset_2",
    REPO_ROOT / "data" / "roboflow_ball_seg" / "qc_volleyball-hwxp2_3",
    REPO_ROOT / "data" / "roboflow_ball_seg" / "volleyball-training_volley-ball-unwlq_1",
    REPO_ROOT / "data" / "roboflow_ball_seg" / "salo-levy-nlqrn_volley-ball-detection_9",
]


def build_combined_data_yaml(out_path: Path) -> Path:
    """Same pattern as train_ball.py's build_combined_data_yaml - each converted
    seg dataset already has a normalized train/val (not valid) split, per
    convert_boxes_to_seg_masks.py, so no per-dataset fallback logic is needed here."""
    train_dirs, val_dirs = [], []
    for ds in DATASETS:
        train_dir, val_dir = ds / "train" / "images", ds / "val" / "images"
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(f"expected train/images and val/images under {ds}")
        train_dirs.append(str(train_dir))
        val_dirs.append(str(val_dir))

    data = {"train": train_dirs, "val": val_dirs, "nc": 1, "names": ["ball"]}
    out_path.write_text(yaml.dump(data, sort_keys=False))
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/ball_finetune_seg")
    parser.add_argument("--name", default="seg_v2_combined")
    args = parser.parse_args()

    data_yaml = build_combined_data_yaml(REPO_ROOT / "data" / "combined_ball_dataset_seg.yaml")
    print(f"combined seg dataset config -> {data_yaml}")

    model = YOLO(args.base_model)
    model.train(data=str(data_yaml), epochs=args.epochs, patience=args.patience,
                batch=args.batch, imgsz=args.imgsz, device=args.device,
                project=args.project, name=args.name)


if __name__ == "__main__":
    main()
