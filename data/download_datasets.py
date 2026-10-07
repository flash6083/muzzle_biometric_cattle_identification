import os
import json
import numpy as np
from pathlib import Path
from PIL import Image


def download_dataset(dataset_name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = {
        'cmid': 'https://www.kaggle.com/datasets/kollabathulakaushik/cattle-images-db-for-muzzle-based-identification',
        'muzzle_268': 'https://doi.org/10.5281/zenodo.6324361',
        'cffd': 'https://doi.org/10.1016/j.dib.2024.110125',
        'mcnd': 'https://doi.org/10.5281/zenodo.7988559',
        'muzzle_211385': 'https://www.kaggle.com/datasets/norashimah/muzzle-211385',
        'roboflow': 'https://universe.roboflow.com/muzzle-identification9161/cattle-identification',
    }

    print(f"Downloading {dataset_name}")
    print(f"URL: {urls.get(dataset_name, 'NOT FOUND')}")
    print("Manual download required - see Pipeline.md §3 for links")
    print(f"Place extracted data in: {output_dir}")


def organize_dataset(source_dir, output_dir, structure='identity'):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if structure == 'identity':
        for img in source_dir.rglob('*'):
            if img.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                parts = img.parts[-3:-1] if len(img.parts) > 3 else [img.parent.name]
                identity_dir = output_dir / parts[0] if parts else output_dir / 'unknown'
                identity_dir.mkdir(parents=True, exist_ok=True)
                dest = identity_dir / img.name
                if not dest.exists():
                    os.symlink(img, dest)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', default='data/raw')
    args = parser.parse_args()
    download_dataset(args.dataset, args.output_dir)
