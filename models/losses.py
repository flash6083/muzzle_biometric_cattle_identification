import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, num_identities, embedding_dim=512, scale=64.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_identities = num_identities
        self.weight = nn.Parameter(torch.randn(num_identities, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cos_theta = F.linear(embeddings, weight)
        theta = cos_theta.acos()
        target_logits = theta + self.margin
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        logits = torch.where(one_hot.bool(), target_logits, cos_theta)
        logits *= self.scale
        loss = self.ce(logits, labels)
        return loss


class CosFaceLoss(nn.Module):
    def __init__(self, num_identities, embedding_dim=512, scale=64.0, margin=0.35):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_identities = num_identities
        self.weight = nn.Parameter(torch.randn(num_identities, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cos_theta = F.linear(embeddings, weight)
        cos_theta_margin = cos_theta - self.margin
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        logits = torch.where(one_hot.bool(), cos_theta_margin, cos_theta)
        logits *= self.scale
        loss = self.ce(logits, labels)
        return loss


class AdaFaceLoss(nn.Module):
    def __init__(self, num_identities, embedding_dim=512, scale=64.0, margin=0.4):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_identities = num_identities
        self.weight = nn.Parameter(torch.randn(num_identities, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()
        self.adaptive_scale = nn.Parameter(torch.ones(num_identities))

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cos_theta = F.linear(embeddings, weight)
        theta = cos_theta.acos()
        target_logits = theta + self.margin
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        logits = torch.where(one_hot.bool(), target_logits, cos_theta)
        logits *= self.scale * self.adaptive_scale[labels]
        loss = self.ce(logits, labels)
        return loss
