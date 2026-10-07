#!/usr/bin/env python3
"""Local muzzle-quality pipeline for small datasets.

This script is designed for a laptop-first workflow:
1. detect or load muzzle crops from identity folders,
2. compute blur/quality values,
3. accept or reject images before sending them to Colab.

It supports two modes:
- default: detect a muzzle crop from each image and then quality-filter it
- --skip-detection: assume images are already crops and only apply quality filtering

The accepted set is intended to be used later for training/SSL in Colab.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_MODEL = "yolov8n-pose.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_image_paths(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    images = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
    return images


def compute_blur_score(image_bgr: np.ndarray) -> float:
    """Return a blur measure where higher = sharper. Uses Laplacian variance."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def diagnose_crop(image_bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # A complementary roughness measure from the frequency domain.
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = max(5, min(h, w) // 10)
    mask = np.ones((h, w), dtype=np.float32)
    mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2] = 0.0
    high_freq = float(np.sum(magnitude * mask))
    low_freq = float(np.sum(magnitude * (1.0 - mask)))
    freq_score = high_freq / (low_freq + 1e-8)

    return {"laplacian_var": lap_var, "freq_score": freq_score}


def detect_muzzle_crop(image_path: Path, model: YOLO) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    results = model(str(image_path), verbose=False)
    if not results:
        return image

    # Prefer bounding boxes if available; fall back to pose keypoints if not.
    if results[0].boxes is not None and results[0].boxes.xyxy is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if boxes.size > 0:
            x1, y1, x2, y2 = [int(v) for v in boxes[0]]
            h, w = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 > x1 and y2 > y1:
                return image[y1:y2, x1:x2]

    keypoints = getattr(results[0], 'keypoints', None)
    if keypoints is not None and getattr(keypoints, 'xy', None) is not None:
        pts_tensor = keypoints.xy
        if len(pts_tensor) > 0:
            first_pts = pts_tensor[0].cpu().numpy()
            if first_pts.size > 0:
                xs = first_pts[:, 0]
                ys = first_pts[:, 1]
                h, w = image.shape[:2]
                x1 = max(0, int(xs.min()) - 10)
                y1 = max(0, int(ys.min()) - 10)
                x2 = min(w, int(xs.max()) + 10)
                y2 = min(h, int(ys.max()) + 10)
                if x2 > x1 and y2 > y1:
                    return image[y1:y2, x1:x2]

    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to write crop to {path}")


def quality_decision(score: float, threshold: float) -> bool:
    return score >= threshold


def build_processing_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "accepted": root / "accepted",
        "rejected": root / "rejected",
        "cropped": root / "cropped",
        "metadata": root / "metadata",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def process_dataset(
    input_dir: Path,
    output_root: Path,
    blur_threshold: float,
    model_name: str,
    skip_detection: bool,
) -> Dict[str, object]:
    output_root = Path(output_root)
    dirs = build_processing_dirs(output_root)

    model = None if skip_detection else YOLO(model_name)

    metadata: List[Dict[str, object]] = []
    accepted_count = 0
    rejected_count = 0

    for identity_dir in sorted(input_dir.iterdir()):
        if not identity_dir.is_dir():
            continue

        for img_path in sorted(identity_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            crop = image
            if not skip_detection:
                crop = detect_muzzle_crop(img_path, model)

            crop_path = dirs["cropped"] / identity_dir.name / img_path.name
            save_image(crop_path, crop)

            metrics = diagnose_crop(crop)
            score = metrics["laplacian_var"]
            accepted = quality_decision(score, blur_threshold)

            target_dir = dirs["accepted"] if accepted else dirs["rejected"]
            target_path = target_dir / identity_dir.name / img_path.name
            save_image(target_path, crop)

            metadata.append(
                {
                    "identity": identity_dir.name,
                    "original": str(img_path),
                    "crop": str(crop_path),
                    "selected": accepted,
                    "blur_score": round(score, 4),
                    "laplacian_var": round(metrics["laplacian_var"], 4),
                    "freq_score": round(metrics["freq_score"], 4),
                    "threshold": blur_threshold,
                }
            )

            if accepted:
                accepted_count += 1
            else:
                rejected_count += 1

    report = {
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "model": model_name,
        "skip_detection": skip_detection,
        "blur_threshold": blur_threshold,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "total_processed": accepted_count + rejected_count,
        "accepted_fraction": (accepted_count / (accepted_count + rejected_count)) if (accepted_count + rejected_count) else 0.0,
        "metadata": metadata,
    }

    report_path = output_root / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Processed {report['total_processed']} images")
    print(f"Accepted: {accepted_count}")
    print(f"Rejected: {rejected_count}")
    print(f"Report saved to: {report_path}")
    return report


def build_demo_data(dataset_dir: Path) -> None:
    """Create a tiny synthetic dataset to smoke-test the filter without a downloaded dataset."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    identities = ["cow_001", "cow_002", "cow_003"]
    for identity in identities:
        identity_dir = dataset_dir / identity
        identity_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(2):
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.circle(img, (100, 100), 40, (255, 255, 255), -1)
            if idx == 0:
                # Sharp image, likely accepted
                cv2.circle(img, (100, 100), 30, (0, 0, 0), -1)
            else:
                # Blur-ish image, likely rejected
                kernel = np.ones((7, 7), np.float32) / 49
                img = cv2.filter2D(img, -1, kernel)
            cv2.imwrite(str(identity_dir / f"img_{idx}.png"), img)

    print(f"Synthetic dataset created at: {dataset_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local muzzle quality-filtering pipeline for small cattle datasets.")
    parser.add_argument("--input-dir", default="data/raw/mcnd", help="Directory containing identity folders with raw muzzle images.")
    parser.add_argument("--output-root", default="data/processed", help="Root directory for accepted/rejected/cropped outputs.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model to use for detection. Defaults to yolov8n-pose.pt.")
    parser.add_argument("--blur-threshold", type=float, default=120.0, help="Minimal Laplacian variance needed to accept an image as sharp enough.")
    parser.add_argument("--skip-detection", action="store_true", help="Skip YOLO detection and treat each input image as already-cropped. Useful for smoke tests or pre-cropped data.")
    parser.add_argument("--demo", action="store_true", help="Create a tiny synthetic dataset for validation and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.demo:
        demo_dir = Path("data/demo_local_dataset")
        build_demo_data(demo_dir)
        return

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input dataset does not exist: {input_dir}. "
            "Create it by downloading a dataset or call --demo for a smoke-test dataset."
        )

    process_dataset(
        input_dir=input_dir,
        output_root=output_root,
        blur_threshold=args.blur_threshold,
        model_name=args.model,
        skip_detection=args.skip_detection,
    )


if __name__ == "__main__":
    main()
