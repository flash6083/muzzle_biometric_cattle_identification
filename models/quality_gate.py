import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityRegressionHead(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, embedding):
        return self.head(embedding).squeeze(-1)


def compute_dft_blur_score(image):
    import numpy as np
    import cv2
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.ndim == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)
    center_y, center_x = gray.shape[0] // 2, gray.shape[1] // 2
    mask = np.ones_like(magnitude)
    radius = 10
    mask[center_y - radius:center_y + radius, center_x - radius:center_x + radius] = 0
    high_freq = magnitude * mask
    low_freq = magnitude * (1 - mask)
    return float(np.sum(high_freq) / (np.sum(low_freq) + 1e-8))
