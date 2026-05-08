import numpy as np
from itertools import combinations
import dendropy
from dendropy.calculate.treecompare import symmetric_difference, unweighted_robinson_foulds_distance


def compute_rf_distance(tree1_newick, tree2_newick, taxa=None):
    try:
        if taxa is None:
            tns = dendropy.TaxonNamespace()
        else:
            tns = dendropy.TaxonNamespace(taxa)
        t1 = dendropy.Tree.get(data=tree1_newick, schema="newick", taxon_namespace=tns)
        t2 = dendropy.Tree.get(data=tree2_newick, schema="newick", taxon_namespace=tns)
        rf = symmetric_difference(t1, t2)
        max_rf = 2 * (len(t1.leaf_nodes()) - 3)
        nrf = rf / max_rf if max_rf > 0 else 0.0
        return rf, nrf
    except Exception as e:
        print(f"RF computation error: {e}")
        return None, None


def compute_quartet_accuracy(pred_tree_newick, ref_tree_newick, n_quartets=1000, seed=42):
    try:
        tns = dendropy.TaxonNamespace()
        t1 = dendropy.Tree.get(data=pred_tree_newick, schema="newick", taxon_namespace=tns)
        t2 = dendropy.Tree.get(data=ref_tree_newick, schema="newick", taxon_namespace=tns)

        leaves = [str(l.taxon.label) for l in t1.leaf_nodes()]
        if len(leaves) < 4:
            return 0.0

        rng = np.random.RandomState(seed)
        correct = 0
        total = 0

        for _ in range(n_quartets):
            quartet = rng.choice(leaves, 4, replace=False).tolist()
            q1 = _get_quartet_topology(t1, quartet)
            q2 = _get_quartet_topology(t2, quartet)
            if q1 is not None and q2 is not None:
                if q1 == q2:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0
    except Exception as e:
        print(f"Quartet accuracy error: {e}")
        return 0.0


def _get_quartet_topology(tree, quartet_labels):
    try:
        tns = tree.taxon_namespace
        quartet_taxa = [tns.get_taxon(l) for l in quartet_labels]
        if any(t is None for t in quartet_taxa):
            return None

        mrca = tree.mrca(taxa=quartet_taxa)
        if mrca is None:
            return None

        subtrees = {}
        for label in quartet_labels:
            node = tree.find_node_with_taxon_label(label)
            if node is None:
                return None
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

        if len(pairs) == 2:
            return frozenset(pairs)
        return None
    except Exception:
        return None


def compute_branch_length_correlation(tree1_newick, tree2_newick):
    try:
        tns = dendropy.TaxonNamespace()
        t1 = dendropy.Tree.get(data=tree1_newick, schema="newick", taxon_namespace=tns)
        t2 = dendropy.Tree.get(data=tree2_newick, schema="newick", taxon_namespace=tns)

        bl1 = _get_patristic_distances(t1)
        bl2 = _get_patristic_distances(t2)

        common_keys = set(bl1.keys()) & set(bl2.keys())
        if len(common_keys) < 2:
            return 0.0

        v1 = np.array([bl1[k] for k in common_keys])
        v2 = np.array([bl2[k] for k in common_keys])

        corr = np.corrcoef(v1, v2)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0


def _get_patristic_distances(tree):
    leaves = tree.leaf_nodes()
    distances = {}
    for i, l1 in enumerate(leaves):
        for l2 in leaves[i + 1:]:
            d = tree.distance_between_nodes(l1, l2)
            key = frozenset([str(l1.taxon.label), str(l2.taxon.label)])
            distances[key] = d
    return distances


def compute_kf_distance(tree1_newick, tree2_newick):
    try:
        tns = dendropy.TaxonNamespace()
        t1 = dendropy.Tree.get(data=tree1_newick, schema="newick", taxon_namespace=tns)
        t2 = dendropy.Tree.get(data=tree2_newick, schema="newick", taxon_namespace=tns)
        rf, _ = compute_rf_distance(tree1_newick, tree2_newick)
        bl_corr = compute_branch_length_correlation(tree1_newick, tree2_newick)
        if rf is None:
            return None
        return rf, bl_corr
    except Exception:
        return None


class TreeMetrics:
    def __init__(self):
        self.results = {}

    def evaluate(self, pred_newick, ref_newick, dataset_name="default"):
        rf, nrf = compute_rf_distance(pred_newick, ref_newick)
        qa = compute_quartet_accuracy(pred_newick, ref_newick)
        bl_corr = compute_branch_length_correlation(pred_newick, ref_newick)

        metrics = {
            "rf": rf,
            "nrf": nrf,
            "qa": qa,
            "branch_length_pearson_r": bl_corr,
        }
        self.results[dataset_name] = metrics
        return metrics

    def summary(self):
        lines = []
        for name, m in self.results.items():
            lines.append(f"{name}: nRF={m['nrf']:.4f}, QA={m['qa']:.4f}, BL_r={m['branch_length_pearson_r']:.4f}")
        return "\n".join(lines)
