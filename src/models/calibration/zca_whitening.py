import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
import numpy as np


class ZCAWhitening(nn.Module):
    def __init__(self, embed_dim, eps=1e-6, momentum=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.momentum = momentum

        self.register_buffer("running_mean", torch.zeros(embed_dim))
        self.register_buffer("running_cov", torch.eye(embed_dim))
        self.register_buffer("whitening_matrix", torch.eye(embed_dim))
        self.register_buffer("bias", torch.zeros(embed_dim))

        self.learnable_scale = nn.Parameter(torch.ones(embed_dim))
        self.learnable_shift = nn.Parameter(torch.zeros(embed_dim))
        self._initialized = False

    def _compute_zca_matrix(self, cov):
        cov_np = cov.cpu().numpy().astype(np.float64)
        S, U = np.linalg.eigh(cov_np)
        S = np.maximum(S, self.eps)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(S))
        W = U @ D_inv_sqrt @ U.T
        return torch.from_numpy(W.astype(np.float32)).to(cov.device)

    def update_statistics(self, embeddings):
        if self._initialized and not self.training:
            return
        with torch.no_grad():
            mean = embeddings.mean(dim=0)
            centered = embeddings - mean
            cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
            if self._initialized:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_cov = (1 - self.momentum) * self.running_cov + self.momentum * cov
            else:
                self.running_mean.copy_(mean)
                self.running_cov.copy_(cov)
                self._initialized = True
            W = self._compute_zca_matrix(self.running_cov)
            self.whitening_matrix.copy_(W)
            self.bias.copy_(-W @ self.running_mean)

    def forward(self, embeddings):
        if self.training and embeddings.shape[0] > 1:
            self.update_statistics(embeddings)

        e_whitened = F.linear(embeddings, self.whitening_matrix, self.bias)
        e_calibrated = e_whitened * self.learnable_scale + self.learnable_shift
        return e_calibrated


class CompositionDebias(nn.Module):
    def __init__(self, embed_dim, n_composition_features=20):
        super().__init__()
        self.regressor = nn.Linear(n_composition_features, embed_dim, bias=True)
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, embeddings, composition_features):
        composition_signal = self.regressor(composition_features)
        gate = torch.sigmoid(self.gate)
        e_debiased = embeddings - gate * composition_signal
        return e_debiased


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


class EmbeddingCalibration(nn.Module):
    def __init__(self, embed_dim, n_composition_features=20, use_zca=True, use_debias=True, use_site_weight=True):
        super().__init__()
        self.use_zca = use_zca
        self.use_debias = use_debias
        self.use_site_weight = use_site_weight

        if use_zca:
            self.zca = ZCAWhitening(embed_dim)
        if use_debias:
            self.debias = CompositionDebias(embed_dim, n_composition_features)
        if use_site_weight:
            self.site_weight = SiteWeighting(embed_dim)

    def forward(self, embeddings, composition_features=None):
        e = embeddings
        if self.use_zca:
            e = self.zca(e)
        if self.use_debias and composition_features is not None:
            e = self.debias(e, composition_features)
        site_weights = None
        if self.use_site_weight:
            e, site_weights = self.site_weight(e)
        return e, site_weights
