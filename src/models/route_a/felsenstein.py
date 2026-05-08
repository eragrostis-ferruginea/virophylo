import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def newick_to_tree_structure(newick_str, leaf_names=None):
    tree_structure = {}
    node_counter = [0]

    def parse_newick(s):
        s = s.strip().rstrip(';').strip()
        node_id = node_counter[0]
        node_counter[0] += 1

        if not s.startswith('('):
            parts = s.split(':')
            name = parts[0].strip()
            bl = float(parts[1]) if len(parts) > 1 else 0.01
            tree_structure[node_id] = {
                'children': [],
                'seq_idx': -1,
                'branch_length': bl,
            }
            return node_id, name, bl

        depth = 0
        split_positions = []
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == ',' and depth == 1:
                split_positions.append(i)

        if not split_positions:
            inner = s[1:-1] if s.startswith('(') else s
            child_id, child_name, child_bl = parse_newick(inner)
            tree_structure[node_id] = {
                'children': [(child_id, child_bl)],
                'seq_idx': -1,
            }
            return node_id, None, 0.01

        parts = []
        prev = 1
        for pos in split_positions:
            parts.append(s[prev:pos])
            prev = pos + 1
        close_paren = s.rfind(')')
        parts.append(s[prev:close_paren])

        after_close = s[close_paren + 1:]
        parent_bl = 0.01
        parent_name = None
        if after_close:
            if ':' in after_close:
                ns, bs = after_close.rsplit(':', 1)
                parent_name = ns.strip() if ns.strip() else None
                try:
                    parent_bl = float(bs.strip())
                except ValueError:
                    parent_bl = 0.01
            else:
                parent_name = after_close.strip()

        children = []
        for part in parts:
            child_id, child_name, child_bl = parse_newick(part.strip())
            children.append((child_id, child_bl))

        tree_structure[node_id] = {
            'children': children,
            'seq_idx': -1,
        }
        return node_id, parent_name, parent_bl

    root_id, _, _ = parse_newick(newick_str)

    leaf_idx = 0
    name_to_idx = {}
    if leaf_names is not None:
        for i, name in enumerate(leaf_names):
            name_to_idx[name] = i

    for nid in list(tree_structure.keys()):
        info = tree_structure[nid]
        if len(info['children']) == 0:
            if leaf_names is not None:
                info['seq_idx'] = leaf_idx
                leaf_idx += 1
            else:
                info['seq_idx'] = leaf_idx
                leaf_idx += 1

    tree_structure['root'] = root_id
    return tree_structure


def topological_order(tree_structure):
    root_id = tree_structure.get('root', 0)
    order = []
    visited = set()

    def dfs(node_id):
        if node_id in visited:
            return
        visited.add(node_id)
        info = tree_structure.get(node_id)
        if info is None or not isinstance(info, dict):
            return
        if 'children' in info:
            for child_id, _ in info['children']:
                dfs(child_id)
        order.append(node_id)

    dfs(root_id)
    return order


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
            P = torch.linalg.matrix_exp(Qt)
            P = P.clamp(min=1e-10)
            P = P / P.sum(dim=-1, keepdim=True)
            return P
        except Exception:
            return torch.eye(self.n_bases, device=Q.device) * (1.0 / self.n_bases)

    def pruning_algorithm(self, alignment_column, tree_structure, Q, frequencies, gamma_rates):
        n_seqs = alignment_column.shape[0]
        n_gamma = gamma_rates.shape[0]

        total_log_likelihood = torch.tensor(0.0, device=alignment_column.device)

        traversal_order = topological_order(tree_structure)

        for g in range(n_gamma):
            rate = gamma_rates[g]
            partial_likelihoods = {}

            for node_id in traversal_order:
                info = tree_structure.get(node_id)
                if info is None or not isinstance(info, dict):
                    continue

                if 'children' not in info or len(info['children']) == 0:
                    obs = alignment_column[info['seq_idx']]
                    if obs >= 0 and obs < self.n_bases:
                        L = torch.zeros(self.n_bases, device=alignment_column.device)
                        L[obs] = 1.0
                    else:
                        L = torch.ones(self.n_bases, device=alignment_column.device)
                    partial_likelihoods[node_id] = L
                else:
                    L_node = torch.ones(self.n_bases, device=alignment_column.device)
                    for child_id, branch_length in info['children']:
                        P = self.compute_transition_prob(Q, branch_length, rate)
                        L_child = partial_likelihoods.get(child_id,
                            torch.ones(self.n_bases, device=alignment_column.device))
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
                    L_from_child = (child_L @ P.T).clamp(min=1e-30)
                    partial[node_idx] = partial[node_idx] + torch.log(L_from_child + 1e-30)

                partial[node_idx] = torch.exp(partial[node_idx])

            root_L = partial[n_nodes - 1]
            site_ll = (root_L * frequencies.unsqueeze(0)).sum(dim=-1)
            total_ll = total_ll + torch.log(site_ll.clamp(min=1e-30)).sum()

        return total_ll / self.n_gamma_categories

    def _compute_P(self, Q, t):
        Qt = Q * t
        try:
            P = torch.linalg.matrix_exp(Qt)
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
