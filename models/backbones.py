import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def build_backbone(name, pretrained=True, embedding_dim=512):
    model = timm.create_model(name, pretrained=pretrained, features_only=True)
    return BackboneWrapper(model, embedding_dim)


class BackboneWrapper(nn.Module):
    def __init__(self, backbone, embedding_dim=512):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(self._feature_dim(), embedding_dim)
        self._init_fc()

    def _feature_dim(self):
        try:
            return self.backbone.feature_dim
        except AttributeError:
            return self.backbone.num_features

    def _init_fc(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.fc(features)
