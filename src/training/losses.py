import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations


class QuartetLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, dist_matrix, ref_tree_newick=None, quartet_indices=None, quartet_labels=None):
        if quartet_indices is not None:
            return self._compute_from_indices(dist_matrix, quartet_indices)
        raise ValueError("quartet_indices must be provided")

    def _compute_from_indices(self, dist_matrix, quartet_indices):
        batch_loss = torch.tensor(0.0, device=dist_matrix.device)
        count = 0
        for q in quartet_indices:
            a, b, c, d = q
            d_ab = dist_matrix[a, b] + dist_matrix[c, d]
            d_ac = dist_matrix[a, c] + dist_matrix[b, d]
            d_ad = dist_matrix[a, d] + dist_matrix[b, c]

            scores = torch.stack([d_ab, d_ac, d_ad]) / self.temperature
            log_probs = F.log_softmax(-scores, dim=0)
            batch_loss = batch_loss - log_probs[0]
            count += 1

        return batch_loss / max(count, 1)

    def sample_quartets(self, n_seqs, n_quartets=100, ref_tree=None, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        indices = []
        for _ in range(n_quartets):
            q = tuple(sorted(rng.choice(n_seqs, 4, replace=False).tolist()))
            indices.append(q)
        return indices


class DistanceRegressionLoss(nn.Module):
    def __init__(self, loss_type="huber", delta=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.delta = delta

    def forward(self, pred_dist, target_dist, mask=None):
        if self.loss_type == "huber":
            loss = F.huber_loss(pred_dist, target_dist, delta=self.delta, reduction='none')
        elif self.loss_type == "mse":
            loss = F.mse_loss(pred_dist, target_dist, reduction='none')
        elif self.loss_type == "mae":
            loss = F.l1_loss(pred_dist, target_dist, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1)
        return loss.mean()


class PhyloLikelihoodLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_likelihood):
        return -log_likelihood.mean()


class TripleLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.0, gamma=0.5, distance_loss_type="huber"):
        super().__init__()
        self.quartet_loss = QuartetLoss()
        self.distance_loss = DistanceRegressionLoss(loss_type=distance_loss_type)
        self.likelihood_loss = PhyloLikelihoodLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, dist_matrix, target_dist=None, log_likelihood=None,
                quartet_indices=None, dist_mask=None):
        loss = torch.tensor(0.0, device=dist_matrix.device)

        if self.alpha > 0 and quartet_indices is not None:
            l_q = self.quartet_loss(dist_matrix, quartet_indices=quartet_indices)
            loss = loss + self.alpha * l_q

        if self.gamma > 0 and target_dist is not None:
            l_d = self.distance_loss(dist_matrix, target_dist, mask=dist_mask)
            loss = loss + self.gamma * l_d

        if self.beta > 0 and log_likelihood is not None:
            l_l = self.likelihood_loss(log_likelihood)
            loss = loss + self.beta * l_l

        return loss

    def set_weights(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


class LossWeightScheduler:
    def __init__(self, total_epochs, schedule_type="phased"):
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type

    def get_weights(self, epoch):
        if self.schedule_type == "phased":
            return self._phased_schedule(epoch)
        elif self.schedule_type == "linear":
            return self._linear_schedule(epoch)
        elif self.schedule_type == "cosine":
            return self._cosine_schedule(epoch)
        return 1.0, 0.0, 0.5

    def _phased_schedule(self, epoch):
        phase = epoch / self.total_epochs
        if phase < 0.3:
            return 1.0, 0.0, 0.5
        elif phase < 0.7:
            return 0.5, 0.5, 0.5
        else:
            return 0.3, 0.7, 0.3

    def _linear_schedule(self, epoch):
        t = epoch / self.total_epochs
        alpha = 1.0 - 0.7 * t
        beta = 0.7 * t
        gamma = 0.5
        return alpha, beta, gamma

    def _cosine_schedule(self, epoch):
        t = epoch / self.total_epochs
        alpha = 0.3 + 0.7 * 0.5 * (1 + np.cos(np.pi * t))
        beta = 0.7 * 0.5 * (1 - np.cos(np.pi * t))
        gamma = 0.5
        return alpha, beta, gamma
