"""Fine-tune the ball detector on self-labeled + public Roboflow volleyball-ball data.

Combines data/self_labeled/ (our own footage, 383 frames) with the public datasets
already downloaded into data/roboflow_ball/ (see scripts/fetch_ball_dataset.py) into a
single training run. The point of mixing in the public data isn't extra volume for its
own sake - v2 (self_labeled alone, 383 frames from 4 clips of one tournament) overfit to
idiosyncratic cues in that narrow footage (e.g. a specific player's shoe color got
mistaken for the ball repeatedly). Diversity across independently-sourced datasets - a
different camera, ball, and court in each - is what teaches the model actual ball
shape/texture instead of a shortcut specific to one set of clips. This matters more than
usual here because the current 4 clips are borrowed placeholder footage, not the
project's own eventual camera setup - so overfitting to their specifics has little
long-term value anyway.

Usage:
    python scripts/train_ball.py
    python scripts/train_ball.py --epochs 20 --name v3_combined
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASETS = [
    REPO_ROOT / "data" / "self_labeled",
    REPO_ROOT / "data" / "roboflow_ball" / "aivolleyballref_volleyball_detection_2",
    REPO_ROOT / "data" / "roboflow_ball" / "primaryws_volleyball_ball_object_detection_dataset_2",
    REPO_ROOT / "data" / "roboflow_ball" / "volleyballtracking-gouow_volleyball-tracking-7ovg1_5",
]


def build_combined_data_yaml(out_path: Path) -> Path:
    """Point Ultralytics at each dataset's existing train/images + val-or-valid/images
    directories directly (label files are found automatically via the standard
    images/->labels/ convention each already follows) - no copying 31k+ images around."""
    train_dirs = []
    val_dirs = []
    for ds in DATASETS:
        train_dir = ds / "train" / "images"
        val_dir = ds / "val" / "images"
        if not val_dir.exists():
            val_dir = ds / "valid" / "images"
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(f"expected train/images and val(id)/images under {ds}")
        train_dirs.append(str(train_dir))
        val_dirs.append(str(val_dir))

    data = {"train": train_dirs, "val": val_dirs, "nc": 1, "names": ["ball"]}
    out_path.write_text(yaml.dump(data, sort_keys=False))
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="models/yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/ball_finetune")
    parser.add_argument("--name", default="v3_combined_external")
    args = parser.parse_args()

    data_yaml = build_combined_data_yaml(REPO_ROOT / "data" / "combined_ball_dataset.yaml")
    print(f"combined dataset config -> {data_yaml}")

    model = YOLO(args.base_model)
    model.train(data=str(data_yaml), epochs=args.epochs, patience=args.patience,
                batch=args.batch, imgsz=args.imgsz, device=args.device,
                project=args.project, name=args.name)


if __name__ == "__main__":
    main()
