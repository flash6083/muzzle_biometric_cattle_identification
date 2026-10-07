import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def train_detector(data_yaml, epochs=100, imgsz=640, batch=16, model="yolov8n-pose.pt", output="detect/weights/best.pt"):
    model = YOLO(model)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="detect",
        name="muzzle_pose",
        exist_ok=True,
    )
    print(f"Detection model trained: {output}")


def detect_and_crop(image_paths, model_path, output_dir, keypoint_indices=[0, 1, 2], target_resolutions=None):
    if target_resolutions is None:
        target_resolutions = [(256, 192), (384, 384)]

    model = YOLO(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned_dir = output_dir / "aligned"
    roi_dir = output_dir / "roi"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model(img, verbose=False)
        if not results or not results[0].keypoints:
            continue

        kpts = results[0].keypoints.xy.cpu().numpy()
        bbox = results[0].boxes.xyxy.cpu().numpy()[0]

        if kpts.shape[0] < 3:
            continue

        points = kpts[:3]

        for res_w, res_h in target_resolutions:
            aligned_crop = similarity_transform_crop(img, points, res_w, res_h)
            roi_crop = letterbox_crop(img, bbox, res_w, res_h)

            stem = Path(img_path).stem
            aligned_path = aligned_dir / f"{stem}_{res_w}x{res_h}.jpg"
            roi_path = roi_dir / f"{stem}_{res_w}x{res_h}.jpg"

            cv2.imwrite(str(aligned_path), aligned_crop)
            cv2.imwrite(str(roi_path), roi_crop)

            metadata.append({
                "original": str(img_path),
                "aligned": str(aligned_path),
                "roi": str(roi_path),
                "resolution": f"{res_w}x{res_h}",
                "keypoints": points.tolist(),
            })

    meta_path = output_dir / "crop_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {len(metadata)} crops in {output_dir}")
    return metadata


def similarity_transform_crop(img, points, target_w, target_h):
    pts_src = points.astype(np.float32)
    margin = 20
    x_min = max(0, int(pts_src[:, 0].min() - margin))
    x_max = min(img.shape[1], int(pts_src[:, 0].max() + margin))
    y_min = max(0, int(pts_src[:, 1].min() - margin))
    y_max = min(img.shape[0], int(pts_src[:, 1].max() + margin))
    roi = img[y_min:y_max, x_min:x_max]
    return cv2.resize(roi, (target_w, target_h))


def letterbox_crop(img, bbox, target_w, target_h):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    margin = 10
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img.shape[1], x2 + margin)
    y2 = min(img.shape[0], y2 + margin)
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (target_w, target_h))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cattle muzzle detection and cropping")
    parser.add_argument("--mode", choices=["train", "detect"], required=True)
    parser.add_argument("--data-yaml", default="configs/detect_data.yaml")
    parser.add_argument("--images", nargs="*", help="Paths to images for detection")
    parser.add_argument("--model-path", default="detect/weights/best.pt")
    parser.add_argument("--output-dir", default="data/crops")
    args = parser.parse_args()

    if args.mode == "train":
        train_detector(args.data_yaml)
    elif args.mode == "detect":
        if not args.images:
            print("Provide image paths with --images")
            exit(1)
        detect_and_crop(args.images, args.model_path, args.output_dir)
