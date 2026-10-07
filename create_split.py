import os
import json
import argparse
import random
from pathlib import Path


def create_identity_split(data_dir, train_ratio=0.7, seed=42, output_dir="data/pairlists"):
    data_dir = Path(data_dir)
    random.seed(seed)

    identity_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    identities = [d.name for d in identity_dirs]

    random.shuffle(identities)
    split_idx = int(len(identities) * train_ratio)
    train_ids = identities[:split_idx]
    open_ids = identities[split_idx:]

    split = {
        "train_identities": train_ids,
        "open_set_identities": open_ids,
        "num_train_identities": len(train_ids),
        "num_open_set_identities": len(open_ids),
        "total_identities": len(identities),
        "train_ratio": train_ratio,
        "seed": seed,
    }

    os.makedirs(output_dir, exist_ok=True)
    split_path = os.path.join(output_dir, "identity_split.json")
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)

    train_list_path = os.path.join(output_dir, "train_list.txt")
    open_list_path = os.path.join(output_dir, "open_set_list.txt")

    with open(train_list_path, "w") as f:
        for identity in train_ids:
            for img in sorted((data_dir / identity).iterdir()):
                if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    f.write(str(img) + "\n")

    with open(open_list_path, "w") as f:
        for identity in open_ids:
            for img in sorted((data_dir / identity).iterdir()):
                if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    f.write(str(img) + "\n")

    print(f"=== Identity Split Created ===")
    print(f"Train identities: {len(train_ids)} ({train_ids})")
    print(f"Open-set identities: {len(open_ids)} ({open_ids})")
    print(f"Split saved to: {split_path}")
    print(f"Train image list: {train_list_path}")
    print(f"Open-set image list: {open_list_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create identity-disjoint train/open-set split")
    parser.add_argument("data_dir", help="Path to dataset directory (identity subdirs)")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="data/pairlists", help="Output directory")
    args = parser.parse_args()
    create_identity_split(args.data_dir, args.train_ratio, args.seed, args.output_dir)
