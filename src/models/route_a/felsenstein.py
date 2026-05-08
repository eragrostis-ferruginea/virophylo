import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FelsensteinPruning(nn.Module):
    def __init__(self, n_bases=4, n_gamma_categories=4):
        super().__init__()
        self.n_bases = n_bases
        self.n_gamma_categories = n_gamma_categories

    def compute_gamma_rates(self, alpha, n_categories):
        if n_categories == 1:
            return torch.ones(1, device=alpha.device)

        try:
            import scipy.stats as stats
            quantiles = [(2 * i - 1) / (2 * n_categories) for i in range(1, n_categories + 1)]
            rates = torch.tensor(
                [stats.gamma.ppf(q, alpha.item(), scale=1.0 / alpha.item()) for q in quantiles],
                device=alpha.device,
                dtype=alpha.dtype,
            )
            mean_rate = rates.mean()
            rates = rates / mean_rate.clamp(min=1e-10)
            return rates
        except Exception:
            return torch.ones(n_categories, device=alpha.device)

    def compute_transition_prob(self, Q, t, rate_category=1.0):
        Qt = Q * t * rate_category
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(Qt)
            exp_diag = torch.diag(torch.exp(eigenvalues))
            P = eigenvectors @ exp_diag @ torch.linalg.inv(eigenvectors)
            P = P.clamp(min=1e-10)
            P = P / P.sum(dim=-1, keepdim=True)
            return P
        except Exception:
            return torch.eye(self.n_bases, device=Q.device) * (1.0 / self.n_bases)

    def pruning_algorithm(self, alignment_column, tree_structure, Q, frequencies, gamma_rates):
        n_seqs = alignment_column.shape[0]
        n_gamma = gamma_rates.shape[0]

        total_log_likelihood = torch.tensor(0.0, device=alignment_column.device)

        for g in range(n_gamma):
            rate = gamma_rates[g]
            partial_likelihoods = {}

            for node_id, node_info in tree_structure.items():
                if 'children' not in node_info or len(node_info['children']) == 0:
                    obs = alignment_column[node_info['seq_idx']]
                    if obs >= 0 and obs < self.n_bases:
                        L = torch.zeros(self.n_bases, device=alignment_column.device)
                        L[obs] = 1.0
                    else:
                        L = torch.ones(self.n_bases, device=alignment_column.device)
                    partial_likelihoods[node_id] = L
                else:
                    L_node = torch.ones(self.n_bases, device=alignment_column.device)
                    for child_id, branch_length in node_info['children']:
                        P = self.compute_transition_prob(Q, branch_length, rate)
                        L_child = partial_likelihoods[child_id]
                        L_from_child = P @ L_child
                        L_node = L_node * L_from_child
                    partial_likelihoods[node_id] = L_node

            root_id = tree_structure.get('root', 0)
            root_L = partial_likelihoods.get(root_id, torch.ones(self.n_bases, device=alignment_column.device))
            site_likelihood = (root_L * frequencies).sum()
            total_log_likelihood = total_log_likelihood + torch.log(site_likelihood.clamp(min=1e-30))

        return total_log_likelihood / n_gamma

    def forward(self, alignment, tree_structures, Q, frequencies, alpha):
        n_sites = alignment.shape[1] if alignment.dim() > 1 else 1
        gamma_rates = self.compute_gamma_rates(alpha, self.n_gamma_categories)

        total_ll = torch.tensor(0.0, device=alignment.device)
        for site_idx in range(n_sites):
            if alignment.dim() > 1:
                column = alignment[:, site_idx]
            else:
                column = alignment

            tree_struct = tree_structures if isinstance(tree_structures, dict) else tree_structures[0]
            ll = self.pruning_algorithm(column, tree_struct, Q, frequencies, gamma_rates)
            total_ll = total_ll + ll

        return total_ll / max(n_sites, 1)


class VectorizedFelsenstein(nn.Module):
    def __init__(self, n_bases=4, n_gamma_categories=4):
        super().__init__()
        self.n_bases = n_bases
        self.n_gamma_categories = n_gamma_categories

    def forward(self, alignment_onehot, branch_lengths, parent_indices, child_indices,
                Q, frequencies, alpha):
        n_seqs, n_sites, _ = alignment_onehot.shape
        n_nodes = len(parent_indices)

        gamma_rates = self._compute_gamma_rates(alpha)

        total_ll = torch.tensor(0.0, device=alignment_onehot.device)

        for g in range(self.n_gamma_categories):
            rate = gamma_rates[g]
            partial = torch.zeros(n_nodes, n_sites, self.n_bases, device=alignment_onehot.device)

            for i in range(n_seqs):
                partial[i] = alignment_onehot[i]

            for node_idx in range(n_seqs, n_nodes):
                for child_local_idx in range(2):
                    child_idx = child_indices[node_idx - n_seqs][child_local_idx]
                    bl = branch_lengths[node_idx - n_seqs][child_local_idx]
                    P = self._compute_P(Q, bl * rate)
                    child_L = partial[child_idx]
                    partial[node_idx] = partial[node_idx] + torch.log(P @ child_L.T.clamp(min=1e-30).T + 1e-30)

                partial[node_idx] = torch.exp(partial[node_idx])

            root_L = partial[n_nodes - 1]
            site_ll = (root_L * frequencies.unsqueeze(0)).sum(dim=-1)
            total_ll = total_ll + torch.log(site_ll.clamp(min=1e-30)).sum()

        return total_ll / self.n_gamma_categories

    def _compute_P(self, Q, t):
        Qt = Q * t
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(Qt)
            exp_diag = torch.diag(torch.exp(eigenvalues))
            P = eigenvectors @ exp_diag @ torch.linalg.inv(eigenvectors)
            P = P.clamp(min=1e-10)
            P = P / P.sum(dim=-1, keepdim=True)
            return P
        except Exception:
            return torch.eye(self.n_bases, device=Q.device) * 0.25

    def _compute_gamma_rates(self, alpha):
        try:
            import scipy.stats as stats
            n = self.n_gamma_categories
            quantiles = [(2 * i - 1) / (2 * n) for i in range(1, n + 1)]
            rates = torch.tensor(
                [stats.gamma.ppf(q, alpha.item(), scale=1.0 / alpha.item()) for q in quantiles],
                device=alpha.device,
                dtype=alpha.dtype,
            )
            return rates / rates.mean().clamp(min=1e-10)
        except Exception:
            return torch.ones(self.n_gamma_categories, device=alpha.device)
