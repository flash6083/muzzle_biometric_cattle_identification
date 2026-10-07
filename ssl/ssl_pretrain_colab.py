# Phase 3: SSL Pretraining on Cattle Muzzle Crops
# Run in Google Colab (free GPU tier)
# ============================================================

# --- Cell 1: Setup ---
# !pip install -r /content/requirements.txt
# !pip install lightly solo-learn

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')

# --- Cell 2: Import ---
import os
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm

# --- Cell 3: Dataset ---
class MuzzleCropDataset(Dataset):
    def __init__(self, crop_dir, transform=None):
        self.samples = []
        crop_dir = Path(crop_dir)
        for img_path in crop_dir.rglob('*'):
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                self.samples.append(str(img_path))
        print(f'Found {len(self.samples)} images in {crop_dir}')
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


# --- Cell 4: SimCLR Training ---
def train_simclr(
    crop_dir='data/crops/aligned',
    embedding_dim=512,
    batch_size=256,
    epochs=100,
    lr=3e-4,
    temperature=0.5,
    output_dir='ssl/checkpoints/simclr',
):
    import lightly
    from lightly.models.modules import SimCLRProjectionHead
    from lightly.loss import SimCLRLoss
    from lightly.data.collate import SimCLRCollateFunction

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = MuzzleCropDataset(crop_dir, transform=transform)
    collate_fn = SimCLRCollateFunction(transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    backbone = timm.create_model('resnet18', pretrained=True, features_only=True)
    projection_head = SimCLRProjectionHead(512, 512, embedding_dim)
    model = lightly.models.SimCLR(backbone, projection_head)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = SimCLRLoss(temperature=temperature)

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            (view1, view2), _, _ = batch
            view1, view2 = view1.to(device), view2.to(device)
            z1, z2 = model(view1, view2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: loss={avg_loss:.4f}')

    torch.save(model.backbone.state_dict(), os.path.join(output_dir, 'simclr_backbone.pth'))
    print(f'SSL checkpoint saved to {output_dir}')
    return model.backbone


# --- Cell 5: DINO Alternative (custom implementation) ---
class DINOTeacher(nn.Module):
    def __init__(self, backbone, projection_head):
        super().__init__()
        self.backbone = backbone
        self.projection = projection_head

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.projection(features)


def train_dino(
    crop_dir='data/crops/aligned',
    embedding_dim=512,
    batch_size=64,
    epochs=100,
    lr=5e-4,
    teacher_update_every=20,
    output_dir='ssl/checkpoints/dino',
):
    import lightly
    from lightly.models.modules import DINOProjectionHead
    from lightly.loss import DINOLoss

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = MuzzleCropDataset(crop_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    student_backbone = timm.create_model('resnet18', pretrained=True, features_only=True)
    teacher_backbone = timm.create_model('resnet18', pretrained=True, features_only=True)
    for param in teacher_backbone.parameters():
        param.requires_grad = False

    projection_head = DINOProjectionHead(512, 512, embedding_dim)
    teacher_projection = DINOProjectionHead(512, 512, embedding_dim)
    teacher_projection.load_state_dict(projection_head.state_dict())

    student = nn.Sequential(student_backbone, projection_head).to(device)
    teacher = DINOTeacher(teacher_backbone, teacher_projection).to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion = DINOLoss()

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            images = batch[0] if isinstance(batch, tuple) else batch
            images = images.to(device)
            student_out = student(images)
            with torch.no_grad():
                teacher_out = teacher(images)
            loss = criterion(student_out, teacher_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f'DINO Epoch {epoch}: loss={total_loss/len(dataloader):.4f}')
        if epoch % teacher_update_every == 0:
            teacher.backbone.load_state_dict(student.backbone.state_dict())
            teacher.projection.load_state_dict(student.projection.state_dict())

    torch.save(student.backbone.state_dict(), os.path.join(output_dir, 'dino_backbone.pth'))
    print(f'DINO checkpoint saved to {output_dir}')
    return student.backbone


# --- Cell 6: Run ---
# train_simclr(crop_dir='data/crops/aligned', epochs=100)
# train_dino(crop_dir='data/crops/aligned', epochs=100)
