import numpy as np
from collections import defaultdict
import copy


class NJTreeBuilder:
    def __init__(self):
        self.tree = None

    def build(self, dist_matrix, names=None):
        n = dist_matrix.shape[0]
        if names is None:
            names = [f"seq_{i}" for i in range(n)]

        if n <= 2:
            return self._build_trivial(dist_matrix, names)

        dist = dist_matrix.copy().astype(float)
        active = list(range(n))
        node_names = list(names)
        children = defaultdict(list)
        branch_lengths = {}
        next_node = n

        while len(active) > 2:
            r = len(active)
            totals = np.sum(dist[np.ix_(active, active)], axis=1)

            Q = np.full((r, r), np.inf)
            for i_idx in range(r):
                for j_idx in range(i_idx + 1, r):
                    i, j = active[i_idx], active[j_idx]
                    Q[i_idx, j_idx] = (r - 2) * dist[i, j] - totals[i_idx] - totals[j_idx]
                    Q[j_idx, i_idx] = Q[i_idx, j_idx]

            min_idx = np.unravel_index(np.argmin(Q), Q.shape)
            i_idx, j_idx = min_idx[0], min_idx[1]
            i, j = active[i_idx], active[j_idx]

            d_ij = dist[i, j]
            d_iu = 0.5 * d_ij + (totals[i_idx] - totals[j_idx]) / (2 * (r - 2))
            d_ju = d_ij - d_iu

            d_iu = max(d_iu, 0.0)
            d_ju = max(d_ju, 0.0)

            u = next_node
            next_node += 1
            node_names.append(f"internal_{u}")
            children[u] = [i, j]
            branch_lengths[(u, i)] = d_iu
            branch_lengths[(u, j)] = d_ju

            new_dist = np.zeros((next_node, next_node))
            new_dist[:dist.shape[0], :dist.shape[1]] = dist
            for k in active:
                if k != i and k != j:
                    d_uk = 0.5 * (dist[i, k] + dist[j, k] - d_ij)
                    new_dist[u, k] = d_uk
                    new_dist[k, u] = d_uk
            dist = new_dist

            active.remove(i)
            active.remove(j)
            active.append(u)

        if len(active) == 2:
            i, j = active[0], active[1]
            d_ij = dist[i, j] / 2
            root = next_node
            children[root] = [i, j]
            branch_lengths[(root, i)] = d_ij
            branch_lengths[(root, j)] = d_ij
        else:
            root = active[0]

        return self._to_newick(root, children, branch_lengths, node_names)

    def _build_trivial(self, dist_matrix, names):
        if len(names) == 1:
            return f"({names[0]}:0.0);"
        d = dist_matrix[0, 1] / 2
        return f"({names[0]}:{d:.6f},{names[1]}:{d:.6f});"

    def _to_newick(self, root, children, branch_lengths, node_names):
        def _recurse(node):
            if node not in children:
                return node_names[node]
            parts = []
            for child in children[node]:
                subtree = _recurse(child)
                bl = branch_lengths.get((node, child), 0.0)
                parts.append(f"{subtree}:{bl:.6f}")
            return f"({','.join(parts)})"

        return _recurse(root) + ";"


def nj_from_distance_matrix(dist_matrix, names=None):
    builder = NJTreeBuilder()
    return builder.build(dist_matrix, names)
