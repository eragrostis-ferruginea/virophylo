import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations


def get_quartet_topology_from_tree(ref_tree_newick, quartet_labels):
    try:
        import dendropy
        tns = dendropy.TaxonNamespace()
        tree = dendropy.Tree.get(data=ref_tree_newick, schema="newick", taxon_namespace=tns)

        taxa = [tns.get_taxon(l) for l in quartet_labels]
        if any(t is None for t in taxa):
            return 0

        mrca = tree.mrca(taxa=taxa)
        if mrca is None:
            return 0

        subtrees = {}
        for label in quartet_labels:
            node = tree.find_node_with_taxon_label(label)
            if node is None:
                return 0
            current = node
            while current.parent_node != mrca and current.parent_node is not None:
                current = current.parent_node
            partner = None
            for other_label in quartet_labels:
                if other_label == label:
                    continue
                other_node = tree.find_node_with_taxon_label(other_label)
                other_current = other_node
                while other_current.parent_node != mrca and other_current.parent_node is not None:
                    other_current = other_current.parent_node
                if other_current == current:
                    partner = other_label
                    break
            if partner:
                subtrees[label] = partner

        pairs = set()
        used = set()
        for k, v in subtrees.items():
            if k not in used and v not in used:
                pair = tuple(sorted([k, v]))
                pairs.add(pair)
                used.add(k)
                used.add(v)

        remaining = [l for l in quartet_labels if l not in used]
        if len(remaining) == 2:
            pair = tuple(sorted(remaining))
            pairs.add(pair)

        if len(pairs) != 2:
            return 0

        pair_list = sorted(pairs)
        a, b = quartet_labels[0], quartet_labels[1]
        c, d = quartet_labels[2], quartet_labels[3]

        pair_ab = tuple(sorted([a, b]))
        pair_cd = tuple(sorted([c, d]))
        pair_ac = tuple(sorted([a, c]))
        pair_bd = tuple(sorted([b, d]))
        pair_ad = tuple(sorted([a, d]))
        pair_bc = tuple(sorted([b, c]))

        if (pair_ab in pairs and pair_cd in pairs):
            return 0
        elif (pair_ac in pairs and pair_bd in pairs):
            return 1
        elif (pair_ad in pairs and pair_bc in pairs):
            return 2
        return 0
    except Exception:
        return 0


class QuartetLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, dist_matrix, ref_tree_newick=None, quartet_indices=None,
                quartet_topologies=None):
        if quartet_indices is not None:
            return self._compute_from_indices(dist_matrix, quartet_indices, quartet_topologies)
        raise ValueError("quartet_indices must be provided")

    def _compute_from_indices(self, dist_matrix, quartet_indices, quartet_topologies=None):
        batch_loss = torch.tensor(0.0, device=dist_matrix.device)
        count = 0
        for i, q in enumerate(quartet_indices):
            a, b, c, d = q
            d_ab = dist_matrix[a, b] + dist_matrix[c, d]
            d_ac = dist_matrix[a, c] + dist_matrix[b, d]
            d_ad = dist_matrix[a, d] + dist_matrix[b, c]

            scores = torch.stack([d_ab, d_ac, d_ad]) / self.temperature
            log_probs = F.log_softmax(-scores, dim=0)

            if quartet_topologies is not None and i < len(quartet_topologies):
                correct_topo = quartet_topologies[i]
            else:
                correct_topo = 0

            batch_loss = batch_loss - log_probs[correct_topo]
            count += 1

        return batch_loss / max(count, 1)

    def sample_quartets_with_topologies(self, n_seqs, ref_tree_newick, n_quartets=100, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        indices = []
        topologies = []
        for _ in range(n_quartets):
            q = tuple(sorted(rng.choice(n_seqs, 4, replace=False).tolist()))
            indices.append(q)

            if ref_tree_newick is not None:
                topo = get_quartet_topology_from_tree(ref_tree_newick, list(q))
                topologies.append(topo)
            else:
                topologies.append(0)

        return indices, topologies


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
                quartet_indices=None, quartet_topologies=None, dist_mask=None):
        loss = torch.tensor(0.0, device=dist_matrix.device)

        if self.alpha > 0 and quartet_indices is not None:
            l_q = self.quartet_loss(dist_matrix, quartet_indices=quartet_indices,
                                     quartet_topologies=quartet_topologies)
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
