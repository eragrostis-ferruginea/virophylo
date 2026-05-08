import torch
import torch.nn as nn


class SiteWeighting(nn.Module):
    def __init__(self, embed_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, embeddings):
        weights = self.net(embeddings)
        return embeddings * weights, weights.squeeze(-1)
