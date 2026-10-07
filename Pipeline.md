Cattle Muzzle Biometrics — Project Plan
Goal: Build a publishable pipeline for open-set cattle identification from muzzle images, using a 100-identity / 20–40-image-per-cow dataset, with a contribution that is clearly disjoint from the two reference papers below.

## Midterm Pilot: Quality-Filtered Embeddings
The current local MCND run has 150 images from 30 identities. The pilot in this repository trains an identity classifier on accepted crops from training identities, then evaluates cosine-similarity verification on held-out identities. It also exports per-image embeddings and compares verification metrics and image coverage across blur thresholds. This is a preliminary experiment, not evidence that the quality gate or embedding method improves performance until the evaluation has been run and analyzed.

```bash
python -m models.baseline_train --data-dir data/crops/accepted --checkpoint models/checkpoints/muzzle_embedding.pt --device mps
python -m eval.evaluate_embeddings --checkpoint models/checkpoints/muzzle_embedding.pt --data-dir data/crops/cropped --quality-report data/crops/quality_report.json --thresholds 0,50,100 --device mps
```

The trainer excludes identities with fewer than three accepted images by default and saves the identity split next to the checkpoint. The evaluator uses only held-out identities, creates genuine pairs from different images of the same cow, and creates impostor pairs across cows. Report image coverage alongside EER, ROC-AUC, and TAR; the small number of held-out identities makes these pilot estimates uncertain.

1. Context from the reference papers
Paper 1 — Awad, A.I. (2016). From classical methods to animal biometrics: A review on cattle identification and tracking. Computers and Electronics in Agriculture, 123, 423–435.
Survey paper, no ML contribution.
Establishes muzzle prints, iris patterns, and retinal vascular patterns as candidate biometric identifiers for cattle, and reviews classical methods (ear notching, tattooing, branding, RFID) and their failure modes (loss, fraud, welfare concerns).
Useful for the Introduction / Related Work section — good source for classical-method comparison tables and terminology (FMR, FNMR, EER, ABIS architecture).
Paper 2 — Goldmann, T., Šimún, S., Orság, F. (2026). The muzzle as a biometric feature: Cattle identification using unique muzzle patterns. Pattern Recognition, 180, 114417.
Current SOTA. Establishes open-set 1:1 verification as the standard protocol (vs. closed-set classification used by prior work).
Curated 5 public datasets into 448 train / 34 verification identities, disjoint splits, within-dataset impostor pairing to suppress domain shortcuts.
Swept 20 backbones (lightweight CNNs, CNNs, ViTs, hybrid CNN-Transformer) × 3 angular-margin losses (ArcFace, CosFace, AdaFace) × 2 resolutions (256×192, 384×384) × 2 preprocessing modes (keypoint-aligned vs. ROI crop).
Best results: VGG11-BN + ArcFace, aligned, 256×192 → 95.45% acc / 4.93% EER. Swin Transformer + ArcFace, aligned, 384×384 → 96.54% acc / 3.58% EER (best overall). MobileNetV3-L + CosFace, ROI (unaligned), 384×384 → 96.14% acc.
Key finding: alignment sensitivity is architecture-dependent — Swin needs alignment, CNNs and DeiT are robust without it.
Explicitly stated gaps / future work (our opening):
No test-time image-quality gating — flagged as a way to reduce false rejections.
All backbones ImageNet-pretrained only — flagged as limiting statistical power given the small muzzle-domain dataset.
Single-image-vs-single-image verification only — no use of multiple images per identity as a set/template, despite most real acquisition producing several shots per animal.
Code/pair-lists released at: https://github.com/tgoldmann/muzzle_recognition
Our differentiation
Both papers work in single-shot, ImageNet-initialized, no-quality-gating regime. Our contribution stack:

(A) Domain-specific self-supervised pretraining on unlabeled muzzle crops (public datasets, no labels needed).
(B) Attention-based multi-shot template fusion — k images per identity → one embedding via learned attention pooling.
(C) Learned quality gate for false-rejection control at enrollment/query time.
(D, optional) Edge deployment / quantization benchmark of the winning lightweight backbone.
2. To-do list (structured, semester timeline)
Phase 0 — Setup (Week 1)
 Audit the 100-cow dataset: confirm whether the 20–40 images/cow are independent captures or video-frame bursts (changes near-duplicate handling downstream)
 Set up repo structure (data/, detect/, align/, models/, ssl/, fusion/, eval/, configs/)
 Set up experiment tracking (Weights & Biases project)
 Identity-disjoint split: ~70 cows train/closed-set, ~30 cows held out for open-set verification
 Download the 5 public muzzle datasets (links in §3) for unlabeled SSL pretraining pool
Phase 1 — Detection & alignment pipeline (Weeks 1–2)
 Fine-tune YOLOv8n-Pose (or v11-pose) for muzzle bbox + 3 keypoints (2 nostril-lowest points + plate center, per Goldmann et al.'s scheme)
 Implement ROI (letterbox) cropping branch
 Implement keypoint-based similarity-transform alignment branch (Kornia or custom)
 Generate aligned + unaligned crops at 256×192 and 384×384 for all identities
 Sanity-check detection completeness (target: 100% keypoint recovery, as in the reference paper)
Phase 2 — Baseline reproduction (Week 3)
 Reproduce Goldmann's best single-shot configs on your own data (MobileNetV3-L / VGG11-BN + ArcFace/CosFace)
 Implement ArcFace, CosFace, AdaFace loss functions
 Establish this as your control condition / sanity check before building anything novel
Phase 3 — Contribution A: domain-specific SSL pretraining (Weeks 4–6)
 Pool unlabeled muzzle crops (public datasets + own data, no labels used)
 Run SimCLR or DINO pretraining (via lightly or solo-learn) on a ResNet18/MobileNetV3 backbone
 Fine-tune SSL-pretrained backbone on labeled data with metric-learning loss
 Ablate: SSL-init vs. ImageNet-init, same architecture, same loss, same split
Phase 4 — Contribution B: attention-based multi-shot fusion (Weeks 6–9)
 Design attention-pooling head over per-image embeddings → single template vector per identity
 Train end-to-end: image encoder → attention fusion → ArcFace/AdaFace loss on fused template
 Build k-shot evaluation curve: verification accuracy as a function of k = 1, 3, 5, 10 images/template
 Compare template-vs-template and template-vs-single-probe protocols against single-shot baseline (Phase 2)
Phase 5 — Contribution C: quality gate (Weeks 9–10)
 Train a lightweight quality-regression head (weak labels from DFT-based blur score, reused from reference paper's Eq. 1)
 Integrate as (i) hard rejection filter and (ii) soft weight inside the attention fusion from Phase 4
 Report FRR before/after gating at fixed FAR
Phase 6 — Open-set benchmark & evaluation (Weeks 10–12)
 Reproduce Goldmann's protocol structure: within-dataset impostor pairs, EER, AUC, TAR@FAR (10⁻³, 10⁻², 10⁻¹)
 Extend protocol to k-shot templates
 10-fold cross-validation × 5 seeds, report mean ± std with 95% CI
 Statistical testing: paired t-tests with Holm correction across ablations
 Failure analysis: qualitative false-accept/false-reject examples
Phase 7 — Optional: edge deployment (Weeks 12–13)
 Quantize winning backbone (INT8, ONNX or TFLite)
 Benchmark latency/memory on Jetson Nano or Raspberry Pi 4
 Report accuracy-vs-latency tradeoff table
Phase 8 — Writing (Weeks 13–15)
 Structure: Intro → Related Work → Method (A+B+C) → Protocol → Results/Ablations → Failure Analysis → Limitations → Conclusion
 Target venue selection (see §4)
 Prepare code + pair-list release (mirror Goldmann's reproducibility practice)
3. Dataset references and links
Dataset	Description	Link
CMID (Cattle Muzzle Images Dataset)	743 images, 24 identities	https://www.kaggle.com/datasets/kollabathulakaushik/cattle-images-db-for-muzzle-based-identification
Muzzle-268	4,923 images, 268 identities	https://doi.org/10.5281/zenodo.6324361
CFFD (Cows Frontal Face Dataset)	2,893 images, 459 identities (raw; contains duplicates)	https://doi.org/10.1016/j.dib.2024.110125 / https://www.sciencedirect.com/science/article/pii/S2352340924000970
MCND (Muzzle Cow New Dataset)	210 images, 30 identities	https://doi.org/10.5281/zenodo.7988559
Muzzle-211385	863 images, 40 identities	https://www.kaggle.com/datasets/norashimah/muzzle-211385
Roboflow — Cattle Identification	59 images	https://universe.roboflow.com/muzzle-identification9161/cattle-identification
Use these as the unlabeled pretraining pool for Phase 3 (SSL). Your own 100-cow set stays the clean, held-out evaluation benchmark — this is what makes the SSL ablation credible (no label leakage between pretraining and evaluation).

4. Paper references
Awad, A.I. (2016). From classical methods to animal biometrics: A review on cattle identification and tracking. Computers and Electronics in Agriculture, 123, 423–435. https://doi.org/10.1016/j.compag.2016.03.014

Goldmann, T., Šimún, S., Orság, F. (2026). The muzzle as a biometric feature: Cattle identification using unique muzzle patterns. Pattern Recognition, 180, 114417. https://doi.org/10.1016/j.patcog.2026.114417 — code/data: https://github.com/tgoldmann/muzzle_recognition

Deng, J., Guo, J., Yang, J., Xue, N., Kotsia, I., Zafeiriou, S. (2022). ArcFace: Additive angular margin loss for deep face recognition. IEEE TPAMI, 44(10), 5962–5979. https://doi.org/10.1109/TPAMI.2021.3087709

Wang, H., et al. (2018). CosFace: Large margin cosine loss for deep face recognition. CVPR. https://doi.org/10.1109/CVPR.2018.00552

Kim, M., Jain, A.K., Liu, X. (2022). AdaFace: Quality adaptive margin for face recognition. CVPR. https://doi.org/10.1109/CVPR52688.2022.01819

Li, G., Erickson, G.E., Xiong, Y. (2022). Individual beef cattle identification using muzzle images and deep learning techniques. Animals, 12(11), 1453. https://doi.org/10.3390/ani12111453

Lee, T., Na, Y., Kim, B.G., Lee, S., Choi, Y. (2023). Identification of individual Hanwoo cattle by muzzle pattern images through deep learning. Animals, 13(18), 2856. https://doi.org/10.3390/ani13182856

Dulal, R., Zheng, L., Kabir, M.A. (2025). MHAFF: Multihead attention feature fusion of CNN and transformer for cattle identification. IEEE Trans. AgriFood Electronics, 3(2), 506–517. https://doi.org/10.1109/TAFE.2025.3574708

Čermák, V., Picek, L., Adam, L., Papafitsoros, K. (2024). WildlifeDatasets: An open-source toolkit for animal re-identification. WACV.

5. Tech stack summary
Purpose	Tool
Detection / keypoints	Ultralytics YOLOv8/v11-pose
Alignment	Kornia, OpenCV
Backbones	timm
SSL pretraining	lightly or solo-learn (SimCLR / DINO)
Metric learning losses	Custom ArcFace/CosFace/AdaFace, or pytorch-metric-learning
Experiment tracking	Weights & Biases
Evaluation	scikit-learn (EER/ROC/AUC), scipy (paired t-test, Holm correction)
Edge deployment (optional)	ONNX Runtime / TFLite, TensorRT (Jetson)
6. Target venues
Computers and Electronics in Agriculture (Elsevier) — strong precision-livestock fit, same venue as Paper 1
Smart Agricultural Technology (Elsevier) — newer, faster turnaround
CVPR/WACV Agriculture-Vision workshop — faster review cycle, good for a workshop-first version of contributions A+B before the full journal submission