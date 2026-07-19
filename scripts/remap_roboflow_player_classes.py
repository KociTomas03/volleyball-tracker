"""Collapse a multi-class player-action Roboflow dataset into single-class 'player'.

personal-tajuk/volleyball-detection-gs7kt (and similarly-shaped public datasets)
label every player's action state (standing, moving, blocking, digging, ...) as a
separate class rather than a single 'player' class - useful for action recognition,
but for our player *detector* (nc=1, names=[player], see data/self_labeled_players/
data.yaml) all 9 classes mean the same thing: "this box is a player." Rewrites every
label file's leading class id to 0 in place and rewrites data.yaml to nc=1,
names=[player] - the images/boxes themselves are untouched, only the class id.

Usage:
    python scripts/remap_roboflow_player_classes.py --dataset-root data/roboflow_players/personal-tajuk_volleyball-detection-gs7kt
"""

import argparse
from pathlib import Path

import yaml


def remap_labels(dataset_root: Path) -> int:
    n_files = 0
    for labels_dir in dataset_root.glob("*/labels"):
        for label_path in labels_dir.glob("*.txt"):
            text = label_path.read_text()
            if not text.strip():
                continue
            lines = []
            for line in text.splitlines():
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    lines.append(f"0 {parts[1]}")
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
            n_files += 1
    return n_files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()

    data_yaml_path = args.dataset_root / "data.yaml"
    data = yaml.safe_load(data_yaml_path.read_text())
    original_names = data.get("names")
    data["names"] = ["player"]
    data["nc"] = 1
    data_yaml_path.write_text(yaml.dump(data, sort_keys=False))
    print(f"data.yaml: {original_names} -> ['player'] (nc=1)")

    n = remap_labels(args.dataset_root)
    print(f"remapped {n} label files to class 0")


if __name__ == "__main__":
    main()
