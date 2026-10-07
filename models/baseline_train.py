from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_BACKBONE = "mobilenetv3_small_100"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


class BaselineModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        num_identities: int = 2,
        embedding_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.freeze_backbone = freeze_backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        with torch.no_grad():
            feature_dim = self.backbone(torch.zeros(1, 3, 224, 224)).shape[1]
        self.projection = nn.Linear(feature_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_identities)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> BaselineModel:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return F.normalize(self.projection(features), dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(images))


class IdentityImageDataset(Dataset):
    def __init__(self, samples: Sequence[tuple[Path, int]], image_size: int) -> None:
        self.samples = list(samples)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return image_tensor, label


def image_paths_by_identity(data_dir: Path) -> dict[str, list[Path]]:
    identities: dict[str, list[Path]] = {}
    for identity_dir in sorted(data_dir.iterdir()):
        if not identity_dir.is_dir():
            continue
        paths = sorted(
            path
            for path in identity_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if paths:
            identities[identity_dir.name] = paths
    return identities


def create_identity_split(
    identities: dict[str, list[Path]],
    train_ratio: float = 0.7,
    minimum_images: int = 3,
    seed: int = 42,
) -> dict[str, object]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    eligible = sorted(
        identity for identity, paths in identities.items() if len(paths) >= minimum_images
    )
    if len(eligible) < 4:
        raise ValueError(
            "At least four identities need the minimum number of accepted images; "
            f"found {len(eligible)} with minimum_images={minimum_images}."
        )

    random.Random(seed).shuffle(eligible)
    train_count = round(len(eligible) * train_ratio)
    train_count = min(max(train_count, 2), len(eligible) - 2)
    train_identities = sorted(eligible[:train_count])
    eval_identities = sorted(eligible[train_count:])
    return {
        "train_identities": train_identities,
        "eval_identities": eval_identities,
        "excluded_identities": sorted(set(identities) - set(eligible)),
        "minimum_images": minimum_images,
        "train_ratio": train_ratio,
        "seed": seed,
    }


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(
    data_dir: Path,
    checkpoint_path: Path,
    backbone_name: str = DEFAULT_BACKBONE,
    embedding_dim: int = 128,
    epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    train_ratio: float = 0.7,
    minimum_images: int = 3,
    image_size: int = 224,
    seed: int = 42,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    device_name: str = "auto",
    num_workers: int = 0,
) -> tuple[BaselineModel, dict[str, object]]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    identities = image_paths_by_identity(data_dir)
    split = create_identity_split(identities, train_ratio, minimum_images, seed)
    train_identities = split["train_identities"]
    class_to_index = {name: index for index, name in enumerate(train_identities)}
    samples = [
        (path, class_to_index[identity])
        for identity in train_identities
        for path in identities[identity]
    ]
    loader = DataLoader(
        IdentityImageDataset(samples, image_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    device = get_device(device_name)
    model = BaselineModel(
        backbone_name=backbone_name,
        num_identities=len(train_identities),
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        sample_count = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += len(labels)
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"loss={total_loss / sample_count:.4f} "
            f"train_acc={correct / sample_count:.3f}"
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "backbone_name": backbone_name,
        "embedding_dim": embedding_dim,
        "num_identities": len(train_identities),
        "train_identities": train_identities,
        "eval_identities": split["eval_identities"],
        "class_to_index": class_to_index,
        "image_size": image_size,
        "image_mean": IMAGE_MEAN,
        "image_std": IMAGE_STD,
        "pretrained": pretrained,
        "freeze_backbone": freeze_backbone,
        "seed": seed,
    }
    torch.save(checkpoint, checkpoint_path)
    split_path = checkpoint_path.with_suffix(".split.json")
    split_path.write_text(json.dumps(split, indent=2))
    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Identity split saved to {split_path}")
    return model, split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a cattle identity embedding model on accepted muzzle crops."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/crops/accepted"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/checkpoints/muzzle_embedding.pt")
    )
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--minimum-images", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        backbone_name=args.backbone,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_ratio=args.train_ratio,
        minimum_images=args.minimum_images,
        image_size=args.image_size,
        seed=args.seed,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        device_name=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()