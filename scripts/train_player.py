"""Fine-tune the player detector on self-labeled real-footage data.

Unlike the ball detector's problem (teaching the model what a ball even looks like),
COCO-pretrained yolov8n.pt already knows "person" well - the gap this closes is
volleyball-specific: dense clustering/occlusion at net play and defensive scrambles,
which is where a diagnostic against real touch-attribution ground truth found the true
toucher sometimes isn't tracked at all (see select_label_candidates_players.py for the
targeting rationale). That's a narrower, more targeted gap than the ball's "learn the
object from scratch" problem, so this starts self-labeled-only (no public dataset
mixed in) - revisit adding public data only if cluster/occlusion recall is still weak
after self-labeled rounds.

Usage:
    python scripts/train_player.py
    python scripts/train_player.py --epochs 60 --name v1_round1
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = REPO_ROOT / "data" / "self_labeled_players" / "data.yaml"


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
    args = parser.parse_args()

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"{DATASET_YAML} not found - label candidates first via "
            f"label_ball.py --dataset-root data/self_labeled_players "
            f"and/or bootstrap_player_labels.py"
        )

    model = YOLO(args.base_model)
    model.train(data=str(DATASET_YAML), epochs=args.epochs, patience=args.patience,
                batch=args.batch, imgsz=args.imgsz, device=args.device,
                project=args.project, name=args.name)


if __name__ == "__main__":
    main()
