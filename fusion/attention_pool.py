import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPoolingHead(nn.Module):
    def __init__(self, embedding_dim=512, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, embeddings, mask=None):
        attn_out, attn_weights = self.attention(embeddings, embeddings, embeddings, key_padding_mask=mask)
        output = self.norm(embeddings.mean(dim=1) + attn_out.mean(dim=1))
        return output, attn_weights


class TemplateBuilder:
    def __init__(self, pooling_head=None, embedding_dim=512):
        self.pooling_head = pooling_head or AttentionPoolingHead(embedding_dim)

    @torch.no_grad()
    def build_template(self, image_embeddings):
        if image_embeddings.dim() == 2:
            image_embeddings = image_embeddings.unsqueeze(0)
        template, weights = self.pooling_head(image_embeddings)
        return template.squeeze(0), weights

    @torch.no_grad()
    def build_templates_batch(self, batch_embeddings):
        templates = []
        for emb in batch_embeddings:
            template, _ = self.build_template(emb)
            templates.append(template)
        return torch.stack(templates)
