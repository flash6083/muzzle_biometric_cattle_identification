import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionTemplateFusion(nn.Module):
    def __init__(self, embedding_dim=512, num_heads=4, hidden_dim=256):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.attention_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, embeddings, mask=None):
        batch_size, num_images, dim = embeddings.shape

        attn_weights = self.attention_mlp(embeddings)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1)

        weighted = (embeddings * attn_weights).sum(dim=1)
        template = self.layer_norm(weighted)

        return template, attn_weights


class QualityGate(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.quality_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, embedding):
        score = self.quality_head(embedding)
        return score


class QualityAwareFusion(nn.Module):
    def __init__(self, embedding_dim=512, num_heads=4, hidden_dim=256):
        super().__init__()
        self.fusion = AttentionTemplateFusion(embedding_dim, num_heads, hidden_dim)
        self.quality_gate = QualityGate(embedding_dim)

    def forward(self, embeddings, mask=None):
        template, attn_weights = self.fusion(embeddings, mask)
        quality_score = self.quality_gate(template)
        return template, attn_weights, quality_score
