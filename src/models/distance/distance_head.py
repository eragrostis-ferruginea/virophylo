import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim=256):
        super().__init__()
        input_dim = embed_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, e_i, e_j):
        diff = torch.abs(e_i - e_j)
        prod = e_i * e_j
        combined = torch.cat([e_i, e_j, prod, diff], dim=-1)
        d = F.softplus(self.mlp(combined)).squeeze(-1)
        return d

    def pairwise_distances(self, embeddings):
        n = embeddings.shape[0]
        e_i = embeddings.unsqueeze(1).expand(n, n, -1)
        e_j = embeddings.unsqueeze(0).expand(n, n, -1)
        diff = torch.abs(e_i - e_j)
        prod = e_i * e_j
        combined = torch.cat([e_i, e_j, prod, diff], dim=-1)
        d = F.softplus(self.mlp(combined)).squeeze(-1)
        return d


class EuclideanDistanceHead(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, e_i, e_j):
        return torch.dist(e_i, e_j, p=self.p)

    def pairwise_distances(self, embeddings):
        return torch.cdist(embeddings, embeddings, p=self.p)


class CosineDistanceHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, e_i, e_j):
        sim = F.cosine_similarity(e_i, e_j, dim=-1)
        return 1.0 - sim

    def pairwise_distances(self, embeddings):
        normed = F.normalize(embeddings, p=2, dim=-1)
        sim = normed @ normed.T
        return 1.0 - sim
