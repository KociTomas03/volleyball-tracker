"""Fine-tune the player detector on self-labeled real-footage data plus a public
exhaustively-labeled player dataset.

Unlike the ball detector's original problem (teaching the model what a ball even
looks like), COCO-pretrained yolov8n.pt already knows "person" well - the gap this
closes is volleyball-specific: dense clustering/occlusion at net play and defensive
scrambles, which is where a diagnostic against real touch-attribution ground truth
found the true toucher sometimes isn't tracked at all (see
select_label_candidates_players.py for the targeting rationale).

Includes personal-tajuk/volleyball-detection-gs7kt (Roboflow, CC BY 4.0, 4131 images,
~11.6 boxes/frame - confirmed to box essentially every player per frame, not just one
featured action's actor, unlike several other public volleyball-action datasets that
were checked and rejected for that reason) via
scripts/remap_roboflow_player_classes.py, which collapses its 9 action-state classes
(standing, moving, blocking, ...) into a single player class. This substantially cuts
how much of our own footage needs manual clustering-candidate labeling, while the
self-labeled data still teaches this project's specific camera/occlusion patterns the
public dataset won't have. Pass --self-labeled-only to fall back to the original
self-labeled-only behavior.

Usage:
    python scripts/train_player.py
    python scripts/train_player.py --epochs 60 --name v1_round1
    python scripts/train_player.py --self-labeled-only
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
SELF_LABELED = REPO_ROOT / "data" / "self_labeled_players"
PUBLIC_DATASETS = [
    REPO_ROOT / "data" / "roboflow_players" / "personal-tajuk_volleyball-detection-gs7kt",
]


def build_combined_data_yaml(out_path: Path, extra_datasets: list[Path]) -> Path:
    """Point Ultralytics at each dataset's existing train/images + val-or-valid/images
    directories directly (same pattern as train_ball.py's build_combined_data_yaml) -
    no copying images, labels found via the standard images/->labels/ convention."""
    datasets = [SELF_LABELED, *extra_datasets]
    train_dirs, val_dirs = [], []
    for ds in datasets:
        train_dir = ds / "train" / "images"
        val_dir = ds / "val" / "images"
        if not val_dir.exists():
            val_dir = ds / "valid" / "images"
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(f"expected train/images and val(id)/images under {ds}")
        train_dirs.append(str(train_dir))
        val_dirs.append(str(val_dir))

    data = {"train": train_dirs, "val": val_dirs, "nc": 1, "names": ["player"]}
    out_path.write_text(yaml.dump(data, sort_keys=False))
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="models/yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/player_finetune")
    parser.add_argument("--name", default="v1_round1")
    parser.add_argument("--self-labeled-only", action="store_true",
                         help="skip the public dataset, train on data/self_labeled_players/ alone")
    args = parser.parse_args()

    extra = [] if args.self_labeled_only else PUBLIC_DATASETS
    data_yaml = build_combined_data_yaml(REPO_ROOT / "data" / "combined_player_dataset.yaml", extra)
    print(f"combined dataset config -> {data_yaml}")

    model = YOLO(args.base_model)
    model.train(data=str(data_yaml), epochs=args.epochs, patience=args.patience,
                batch=args.batch, imgsz=args.imgsz, device=args.device,
                project=args.project, name=args.name)


if __name__ == "__main__":
    main()
