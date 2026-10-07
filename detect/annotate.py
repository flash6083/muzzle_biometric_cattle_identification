import os
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2


def _mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        params['points'].append((x, y))
        cv2.circle(params['image'], (x, y), 3, (0, 255, 0), -1)
        cv2.imshow('annotate', params['image'])


def annotate_yolo(data_dir, output_dir, num_keypoints=3):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Annotation tool: manually mark {num_keypoints} keypoints per image")
    print(f"Source: {data_dir}")
    print(f"Output: {output_dir}")

    for identity_dir in sorted(data_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        anno_dir = output_dir / identity_dir.name / 'annotations'
        anno_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(identity_dir.iterdir()):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            print(f"Annotate: {img_path} (click points, press 's' to save, 'n' to skip)")
            display = img.copy()
            callback_state = {'image': display, 'points': []}
            cv2.namedWindow('annotate', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('annotate', _mouse_callback, callback_state)
            cv2.imshow('annotate', display)

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    break
                if key == ord('n'):
                    callback_state['points'] = []
                    cv2.destroyAllWindows()
                    break
                if len(callback_state['points']) >= num_keypoints:
                    # allow a save after enough clicks
                    continue

            if key == ord('n'):
                continue

            pts = callback_state['points'][:num_keypoints]
            if len(pts) != num_keypoints:
                print(f"Skipped {img_path}: only {len(pts)} points selected.")
                cv2.destroyAllWindows()
                continue

            out_path = anno_dir / (img_path.stem + '.json')
            with open(out_path, 'w') as f:
                json.dump({"image": str(img_path), "keypoints": [[int(x), int(y)] for x, y in pts]}, f, indent=2)
            print(f"Saved annotation to {out_path}")
            cv2.destroyAllWindows()


def create_yolo_yaml(data_dir, output_path='configs/detect_data.yaml'):
    yaml_content = f"""train: {data_dir}/train/images
val: {data_dir}/val/images

nc: 1
names: ['muzzle']

kpt_shape: [3, 3]

model: yolov8n-pose.pt

epochs: 100
batch: 16
imgsz: 640
"""
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    print(f"YOLO config saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='data/raw/annotations')
    args = parser.parse_args()
    annotate_yolo(args.data_dir, args.output_dir)
