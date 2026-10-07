# Cattle Muzzle Biometrics — Pipeline

Open-set cattle identification from muzzle images.

## Quick Start

### 1. Setup environment
```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

### 2. Local dataset for pipeline testing
Use **MCND** (Muzzle Cow New Dataset): 210 images, 30 identities — smallest complete dataset with enough identities for verification testing. The current local run contains 150 images from 30 identities, so verify whether the remaining images are available before reporting dataset totals.

Download from: https://doi.org/10.5281/zenodo.7988559

Place in: `data/raw/mcnd/` with structure:
```
data/raw/mcnd/
  cow_001/
    img1.jpg
    img2.jpg
    ...
  cow_002/
    ...
```

### 3. Audit dataset (Phase 0)
```bash
python audit_dataset.py data/raw/mcnd --threshold 0.95
```

### 4. Create identity split
```bash
python create_split.py data/raw/mcnd --train-ratio 0.7 --seed 42
```

### 5. Crop and quality-filter images
```bash
python local_muzzle_pipeline.py --input-dir data/raw/mcnd --output-root data/crops --blur-threshold 50
```

### 6. Train an identity embedding baseline
This creates an identity-disjoint split, trains on accepted crops, and saves a model checkpoint. Pretrained MobileNetV3 weights may require an internet connection on the first run.
```bash
python -m models.baseline_train --data-dir data/crops/accepted --checkpoint models/checkpoints/muzzle_embedding.pt --device mps
```

### 7. Evaluate unseen-cattle verification
```bash
python -m eval.evaluate_embeddings --checkpoint models/checkpoints/muzzle_embedding.pt --data-dir data/crops/cropped --quality-report data/crops/quality_report.json --thresholds 0,50,100 --device mps
```

The evaluation exports per-image embeddings, EER, ROC-AUC, TAR at selected FARs, and image coverage for each blur threshold. Treat results on this small subset as a pilot, not a final benchmark.

## Phases
- **Phase 0**: Dataset audit + identity-disjoint split
- **Phase 1**: YOLOv8n-pose detection + alignment
- **Phase 2**: Baseline reproduction (ArcFace/CosFace)
- **Phase 3**: SSL pretraining (Colab GPU recommended)
- **Phase 4**: Attention-based multi-shot fusion
- **Phase 5**: Quality gate
- **Phase 6**: Open-set benchmark & evaluation
- **Phase 7**: Edge deployment (optional)
- **Phase 8**: Writing

## Key Directories
- `data/raw/` — raw images
- `data/crops/` — generated crops (aligned + ROI)
- `data/pairlists/` — splits, pair lists, split configs
- `detect/` — YOLO detection code
- `align/` — cropping + alignment
- `models/` — backbones, heads, losses
- `ssl/` — SSL pretraining
- `fusion/` — attention pooling
- `eval/` — verification metrics
- `configs/` — experiment configs

## Tech Stack
- Detection: Ultralytics YOLOv8n-pose
- Alignment: Kornia, OpenCV
- Backbones: timm
- SSL: lightly / solo-learn
- Metric losses: custom ArcFace/CosFace/AdaFace
- Tracking: Weights & Biases
- Eval: scikit-learn, scipy
