import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


def load_image(path, target_size=(224, 224)):
    img = Image.open(path).convert("RGB").resize(target_size)
    return np.array(img, dtype=np.float32) / 255.0


def compute_pairwise_similarity(image_paths, batch_size=50):
    n = len(image_paths)
    sims = np.zeros((n, n))
    for i in range(0, n, batch_size):
        for j in range(i, n, batch_size):
            batch_i = np.stack([load_image(p) for p in image_paths[i : i + batch_size]])
            batch_j = np.stack([load_image(p) for p in image_paths[j : j + batch_size]])
            flat_i = batch_i.reshape(batch_i.shape[0], -1)
            flat_j = batch_j.reshape(batch_j.shape[0], -1)
            sims[i : i + batch_size, j : j + batch_size] = cosine_similarity(flat_i, flat_j)
            sims[j : j + batch_size, i : i + batch_size] = sims[i : i + batch_size, j : j + batch_size].T
    return sims


def audit_directory(data_dir, output_path="audit_report.json", similarity_threshold=0.95):
    data_dir = Path(data_dir)
    results = {"dataset_path": str(data_dir), "total_images": 0, "identities": [], "burst_pairs": []}

    identity_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not identity_dirs:
        image_files = sorted(data_dir.rglob("*"))
        print(f"No subdirectories found. Found {len(image_files)} images flat.")
        sys.exit(1)

    print(f"Found {len(identity_dirs)} identity directories")

    for identity_dir in identity_dirs:
        images = sorted([f for f in identity_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
        if not images:
            continue
        identity_name = identity_dir.name
        results["identities"].append({"name": identity_name, "num_images": len(images)})
        results["total_images"] += len(images)

        if len(images) < 2:
            continue

        print(f"  Auditing {identity_name}: {len(images)} images...")
        paths = [str(p) for p in images]
        sims = compute_pairwise_similarity(paths)

        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if sims[i, j] > similarity_threshold:
                    pair = {
                        "identity": identity_name,
                        "image_a": os.path.basename(paths[i]),
                        "image_b": os.path.basename(paths[j]),
                        "similarity": round(float(sims[i, j]), 4),
                    }
                    results["burst_pairs"].append(pair)

    burst_count = len(results["burst_pairs"])
    total_pairs = sum(
        d["num_images"] * (d["num_images"] - 1) / 2 for d in results["identities"]
    )
    burst_ratio = burst_count / total_pairs if total_pairs > 0 else 0

    results["burst_pairs_count"] = burst_count
    results["total_pairs"] = total_pairs
    results["burst_ratio"] = round(burst_ratio, 4)
    results["similarity_threshold"] = similarity_threshold

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Audit Complete ===")
    print(f"Total identities: {len(results['identities'])}")
    print(f"Total images: {results['total_images']}")
    print(f"Total pairs compared: {results['total_pairs']}")
    print(f"Burst pairs (sim > {similarity_threshold}): {burst_count}")
    print(f"Burst ratio: {burst_ratio:.2%}")
    print(f"Report saved to: {output_path}")

    if burst_ratio > 0.3:
        print("\n⚠️  HIGH BURST RATIO: Consider session-level splitting for train/test.")
    elif burst_ratio > 0.1:
        print("\n⚠️  MODERATE BURST RATIO: Check individual identities before splitting.")
    else:
        print("\n✅ LOW BURST RATIO: Image-level splitting appears safe.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit dataset for near-duplicate burst frames")
    parser.add_argument("data_dir", help="Path to dataset directory (identity subdirs)")
    parser.add_argument("--threshold", type=float, default=0.95, help="SSIM/cosine threshold for burst detection")
    parser.add_argument("--output", default="audit_report.json", help="Output JSON report path")
    args = parser.parse_args()
    audit_directory(args.data_dir, args.output, args.threshold)
