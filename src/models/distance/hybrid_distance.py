import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridDistance(nn.Module):
    def __init__(self, learnable_alpha=True, init_alpha=0.8, use_cross_branch_correction=True):
        super().__init__()
        self.use_cross_branch_correction = use_cross_branch_correction
        if learnable_alpha:
            self.log_alpha = nn.Parameter(torch.tensor(float(init_alpha)).log())
        else:
            self.register_buffer("log_alpha", torch.tensor(float(init_alpha)).log())

        if use_cross_branch_correction:
            self.log_cross_weight = nn.Parameter(torch.tensor(0.7).log())

    @property
    def alpha(self):
        return torch.sigmoid(self.log_alpha)

    @property
    def cross_branch_weight(self):
        if self.use_cross_branch_correction:
            return torch.sigmoid(self.log_cross_weight)
        return 0.0

    def forward(self, d_llm, d_k2p, is_cross_branch_mask=None):
        alpha = self.alpha
        d_hybrid = alpha * d_llm + (1 - alpha) * d_k2p

        if self.use_cross_branch_correction and is_cross_branch_mask is not None:
            correction = torch.relu(d_k2p - d_llm) * self.cross_branch_weight
            d_hybrid = d_hybrid + correction * is_cross_branch_mask

        return d_hybrid


class AdaptiveHybridDistance(nn.Module):
    def __init__(self, embed_dim, hidden_dim=64):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, d_llm, d_k2p, e_i=None, e_j=None):
        if e_i is not None and e_j is not None:
            gate_input = torch.cat([e_i, e_j], dim=-1)
            alpha = self.gate_net(gate_input).squeeze(-1)
        else:
            alpha = 0.5
        return alpha * d_llm + (1 - alpha) * d_k2p
