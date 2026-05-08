import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GTRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_bases = 4

    def normalize_rates(self, raw_rates):
        rates = F.softplus(raw_rates)
        total = rates.sum()
        return rates * 6.0 / total.clamp(min=1e-10)

    def normalize_frequencies(self, raw_freq):
        return F.softmax(raw_freq, dim=-1)

    def compute_Q_matrix(self, rates, frequencies):
        Q = torch.zeros(self.n_bases, self.n_bases, device=rates.device)
        rate_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for idx, (i, j) in enumerate(rate_indices):
            Q[i, j] = rates[idx] * frequencies[j]
            Q[j, i] = rates[idx] * frequencies[i]

        diag = Q.sum(dim=-1)
        Q = Q - torch.diag(diag)

        mu = (frequencies * diag).sum()
        Q = Q / mu.clamp(min=1e-10)

        return Q

    def compute_P_matrix(self, Q, t):
        Qt = Q * t
        try:
            P = torch.linalg.matrix_exp(Qt)
        except Exception:
            eigenvalues, eigenvectors = torch.linalg.eig(Q)
            eigenvalues_real = eigenvalues.real
            eigenvectors_real = eigenvectors.real
            exp_diag = torch.diag(torch.exp(eigenvalues_real * t))
            P = eigenvectors_real @ exp_diag @ torch.linalg.inv(eigenvectors_real).real

        P = P.clamp(min=0)
        row_sums = P.sum(dim=-1, keepdim=True)
        P = P / row_sums.clamp(min=1e-10)

        return P

    def forward(self, raw_rates, raw_freq, raw_alpha, t):
        rates = self.normalize_rates(raw_rates)
        frequencies = self.normalize_frequencies(raw_freq)
        alpha = F.softplus(raw_alpha).clamp(min=0.1, max=10.0)

        Q = self.compute_Q_matrix(rates, frequencies)
        P = self.compute_P_matrix(Q, t)

        return P, rates, frequencies, alpha, Q


class GTRParameterHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.rate_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 6),
        )
        self.freq_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4),
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, embeddings):
        raw_rates = self.rate_head(embeddings)
        raw_freq = self.freq_head(embeddings)
        raw_alpha = self.alpha_head(embeddings).squeeze(-1)
        return raw_rates, raw_freq, raw_alpha

    def normalize_rates_fn(self, raw_rates):
        gtr = GTRModel()
        return gtr.normalize_rates(raw_rates)

    def normalize_frequencies_fn(self, raw_freq):
        gtr = GTRModel()
        return gtr.normalize_frequencies(raw_freq)
