from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval.metrics import evaluate_verification
from eval.pairlist import generate_verification_pairs
from models.baseline_train import (
    BaselineModel,
    IMAGE_EXTENSIONS,
    IdentityImageDataset,
    get_device,
    image_paths_by_identity,
)


def load_quality_scores(report_path: Path) -> dict[tuple[str, str], float]:
    report = json.loads(report_path.read_text())
    scores = {}
    for item in report.get("metadata", []):
        identity = item.get("identity")
        crop = item.get("crop") or item.get("original")
        score = item.get("laplacian_var", item.get("blur_score"))
        if identity and crop and score is not None:
            scores[(identity, Path(crop).name)] = float(score)
    return scores


def extract_embeddings(
    model: BaselineModel,
    image_paths: list[Path],
    image_size: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, np.ndarray]:
    dataset = IdentityImageDataset([(path, 0) for path in image_paths], image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    embeddings = {}
    model.eval()
    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(loader):
            vectors = model.encode(images.to(device)).cpu().numpy()
            start = batch_index * batch_size
            for path, vector in zip(image_paths[start : start + len(vectors)], vectors):
                embeddings[str(path)] = vector.astype(np.float32)
    return embeddings


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_dir: Path,
    output_path: Path,
    quality_report: Path | None = None,
    thresholds: list[float] | None = None,
    embeddings_path: Path | None = None,
    device_name: str = "auto",
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    split_path = checkpoint_path.with_suffix(".split.json")
    if not split_path.exists():
        raise FileNotFoundError(f"Identity split file not found: {split_path}")
    split = json.loads(split_path.read_text())
    eval_identities = set(split["eval_identities"])

    all_identities = image_paths_by_identity(data_dir)
    image_paths = [
        path
        for identity in sorted(eval_identities)
        for path in all_identities.get(identity, [])
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_paths:
        raise ValueError(f"No images found for held-out identities under {data_dir}")

    device = get_device(device_name)
    model = BaselineModel(
        backbone_name=checkpoint["backbone_name"],
        num_identities=checkpoint["num_identities"],
        embedding_dim=checkpoint["embedding_dim"],
        pretrained=False,
        freeze_backbone=checkpoint["freeze_backbone"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    embeddings = extract_embeddings(
        model,
        image_paths,
        checkpoint["image_size"],
        device,
        batch_size,
        num_workers,
    )

    if embeddings_path is None:
        embeddings_path = checkpoint_path.with_suffix(".eval_embeddings.npz")
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        embeddings_path,
        image_paths=np.asarray(list(embeddings)),
        identities=np.asarray([Path(path).parent.name for path in embeddings]),
        embeddings=np.stack(list(embeddings.values())),
    )

    quality_scores = load_quality_scores(quality_report) if quality_report else {}
    if quality_report:
        missing = [
            path
            for path in image_paths
            if (Path(path).parent.name, Path(path).name) not in quality_scores
        ]
        if missing:
            raise ValueError(
                f"Quality report has no score for {len(missing)} evaluation images; "
                f"first missing image: {missing[0]}"
            )
        if thresholds is None:
            thresholds = [0.0, 50.0, 100.0]
    else:
        thresholds = [None]

    results = []
    for threshold in thresholds:
        selected_paths = [
            path
            for path in image_paths
            if threshold is None
            or quality_scores[(Path(path).parent.name, Path(path).name)] >= threshold
        ]
        pairs = generate_verification_pairs(selected_paths)
        result = evaluate_verification(pairs, embeddings)
        result.update(
            {
                "quality_threshold": threshold,
                "images_used": len(selected_paths),
                "image_coverage": len(selected_paths) / len(image_paths),
                "identities_with_images": len({Path(path).parent.name for path in selected_paths}),
            }
        )
        results.append(result)

    report = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "split_file": str(split_path),
        "eval_identities": sorted(eval_identities),
        "device": str(device),
        "embeddings_file": str(embeddings_path),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Evaluation saved to {output_path}")
    print(f"Held-out image embeddings saved to {embeddings_path}")
    for result in results:
        print(
            f"threshold={result['quality_threshold']} "
            f"coverage={result['image_coverage']:.1%} "
            f"EER={result['eer']} AUC={result['roc_auc']} "
            f"pairs={result['num_pairs']}"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate unseen-cattle verification and quality-filter coverage."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/crops/cropped"))
    parser.add_argument("--quality-report", type=Path, default=Path("data/crops/quality_report.json"))
    parser.add_argument("--thresholds", help="Comma-separated Laplacian-variance thresholds")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--embeddings-output", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality_report = args.quality_report if args.quality_report.exists() else None
    thresholds = (
        [float(value) for value in args.thresholds.split(",")]
        if args.thresholds
        else None
    )
    output_path = args.output or args.checkpoint.with_suffix(".eval.json")
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_path=output_path,
        quality_report=quality_report,
        thresholds=thresholds,
        embeddings_path=args.embeddings_output,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()