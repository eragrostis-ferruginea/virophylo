import torch
import numpy as np
import os
from Bio import SeqIO
from src.models.tree.tree_metrics import compute_rf_distance, compute_quartet_accuracy
from src.models.tree.nj_builder import nj_from_distance_matrix


class LBAEvaluation:
    def __init__(self, output_dir="outputs/lba_eval"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_lba(self, model, aln_path, ref_tree_path, monophyletic_groups,
                     names=None, device="cuda", prior_groups=None):
        sequences = []
        seq_names = []
        for record in SeqIO.parse(aln_path, "fasta"):
            sequences.append(str(record.seq).upper())
            seq_names.append(record.id)

        with open(ref_tree_path) as f:
            ref_tree = f.read().strip()

        model.eval()
        with torch.no_grad():
            pred_tree, dist_matrix = model.predict_tree(sequences, seq_names)

        results = {
            "total_groups": len(monophyletic_groups),
            "recovered_no_prior": 0,
            "recovered_with_prior": 0,
            "group_details": [],
        }

        for group_name, member_indices in monophyletic_groups.items():
            no_prior = self._check_monophyly(pred_tree, [seq_names[i] for i in member_indices])
            with_prior = False
            if prior_groups and group_name in prior_groups:
                with_prior = self._check_monophyly_constrained(
                    dist_matrix, [seq_names[i] for i in member_indices],
                    [seq_names[i] for i in prior_groups[group_name]]
                )

            results["group_details"].append({
                "group": group_name,
                "members": [seq_names[i] for i in member_indices],
                "monophyletic_no_prior": no_prior,
                "monophyletic_with_prior": with_prior,
            })

            if no_prior:
                results["recovered_no_prior"] += 1
            if with_prior:
                results["recovered_with_prior"] += 1

        results["score_no_prior"] = f"{results['recovered_no_prior']}/{results['total_groups']}"
        results["score_with_prior"] = f"{results['recovered_with_prior']}/{results['total_groups']}"

        rf, nrf = compute_rf_distance(pred_tree, ref_tree)
        results["nRF"] = nrf
        results["QA"] = compute_quartet_accuracy(pred_tree, ref_tree)

        return results

    def _check_monophyly(self, newick_tree, member_names):
        try:
            import dendropy
            tns = dendropy.TaxonNamespace()
            tree = dendropy.Tree.get(data=newick_tree, schema="newick", taxon_namespace=tns)

            member_taxa = set(member_names)
            all_leaves = set(str(l.taxon.label) for l in tree.leaf_nodes())

            if not member_taxa.issubset(all_leaves):
                return False

            mrca = tree.mrca(taxon_labels=list(member_taxa & all_leaves))
            if mrca is None:
                return False

            descendant_labels = set(str(l.taxon.label) for l in mrca.leaf_nodes())
            return descendant_labels == (member_taxa & all_leaves)
        except Exception:
            return False

    def _check_monophyly_constrained(self, dist_matrix, member_names, outgroup_names):
        try:
            member_dists = []
            for i, n1 in enumerate(member_names):
                for j, n2 in enumerate(member_names):
                    if i < j:
                        member_dists.append(dist_matrix[i, j])

            outgroup_dists = []
            for n1 in member_names:
                for n2 in outgroup_names:
                    i = member_names.index(n1) if n1 in member_names else None
                    j = len(member_names) + outgroup_names.index(n2) if n2 in outgroup_names else None
                    if i is not None and j is not None and j < dist_matrix.shape[0]:
                        outgroup_dists.append(dist_matrix[i, j])

            if not member_dists or not outgroup_dists:
                return False

            avg_intra = np.mean(member_dists)
            avg_inter = np.mean(outgroup_dists)
            return avg_intra < avg_inter
        except Exception:
            return False

    def microsporidia_18s_test(self, model, aln_path, ref_tree_path, device="cuda"):
        micro_groups = {
            "Microsporidia": list(range(8)),
        }
        prior_outgroups = {
            "Microsporidia": list(range(8, 28)),
        }
        return self.evaluate_lba(
            model, aln_path, ref_tree_path,
            monophyletic_groups=micro_groups,
            prior_groups=prior_outgroups,
            device=device,
        )
