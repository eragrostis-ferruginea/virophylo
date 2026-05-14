#!/usr/bin/env python3
"""
Ground-truth evaluation: species-level clustering quality of phylogenetic trees.

For each VOGDB family, extracts virus species labels from FAA headers,
then evaluates whether the predicted NJ tree correctly clusters
sequences from the same species together.

Metric: Adjusted Rand Index (ARI) between tree-based clusters and species labels.
Comparison: PHYLA vs Hamming vs SeqIdentity vs Random.

This is a true ground-truth evaluation because species identity is an objective
biological fact, not an algorithmic artifact.

Usage:
  python evaluate_groundtruth.py \
    --ref-pickle virus_data/vogdb_treefam_v2.pickle \
    --pred-pickle virus_data/phyla_predictions.pickle \
    --faa-dir virus_data/faa \
    --msa-dir virus_data/msa \
    --output-dir eval_preds
"""
import sys
import os
import pickle
import csv
import argparse
import re
import random
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "phyla"))

from ete3 import Tree
from Bio import Phylo
from io import StringIO
from collections import Counter


def remove_branch_distances(tree_str):
    phylo_tree = Phylo.read(StringIO(tree_str), "newick")
    for i in phylo_tree.get_nonterminals():
        i.branch_length = None
    for i in phylo_tree.get_terminals():
        i.branch_length = None
    new_str_obj = StringIO()
    Phylo.write(phylo_tree, new_str_obj, "newick")
    new_str = new_str_obj.getvalue()
    new_str = re.sub(r':[^,();\n]*', '', new_str)
    new_str = new_str.replace("'", "")
    return new_str


def get_leaf_names(tree_str):
    t = Phylo.read(StringIO(tree_str), "newick")
    return sorted(str(x.name) for x in t.get_terminals())


def prune_tree_to_leaves(tree_str, leaves_to_keep):
    clean = remove_branch_distances(tree_str)
    t = Tree(clean)
    tree_leaves = set(t.get_leaf_names())
    keep = sorted([l for l in leaves_to_keep if l in tree_leaves])
    if len(keep) < 4:
        return None
    t.prune(keep)
    return t.write(format=5).replace(" ", "").replace("'", "")


def parse_species_labels(faa_path):
    """Extract species labels from FAA headers.
    Returns dict: {sequence_name: species_label}"""
    labels = {}
    with open(faa_path) as f:
        for line in f:
            if line.startswith(">"):
                # Sequence name is first token after '>' (up to first space)
                name = line[1:].split()[0]
                # Species is text in the LAST set of square brackets
                matches = re.findall(r'\[([^\]]+)\]', line)
                if matches:
                    labels[name] = matches[-1]
                else:
                    labels[name] = "unknown"
    return labels


def compute_clusters_from_tree(tree_str, seq_names):
    """Cut the NJ tree to get clusters: each internal node with >=2 leaves
    that are all within a certain subtree forms a cluster.
    
    Uses ete3 to traverse the tree and extract monophyletic groups.
    Returns list of clusters, each cluster is a list of leaf names.
    """
    clean = remove_branch_distances(tree_str)
    t = Tree(clean)
    
    # Get all internal nodes, extract their leaf sets as clusters
    clusters = []
    for node in t.traverse("postorder"):
        if not node.is_leaf():
            leaves = node.get_leaf_names()
            if len(leaves) >= 2:
                clusters.append(sorted(leaves))
    
    return clusters


def adjusted_rand_index(labels_true, labels_pred):
    """Compute Adjusted Rand Index between two label assignments.
    
    labels_true: list of true labels (e.g., species names)
    labels_pred: list of predicted cluster IDs
    """
    n = len(labels_true)
    if n < 2:
        return 1.0
    
    # Build contingency table
    true_set = sorted(set(labels_true))
    pred_set = sorted(set(labels_pred))
    true_idx = {t: i for i, t in enumerate(true_set)}
    pred_idx = {p: i for i, p in enumerate(pred_set)}
    
    contingency = [[0] * len(pred_set) for _ in range(len(true_set))]
    for i in range(n):
        contingency[true_idx[labels_true[i]]][pred_idx[labels_pred[i]]] += 1
    
    # Sum over rows and columns
    a = [sum(row) for row in contingency]
    b = [sum(contingency[i][j] for i in range(len(true_set))) for j in range(len(pred_set))]
    
    # Compute ARI
    sum_comb = sum(n_ij * (n_ij - 1) / 2 for row in contingency for n_ij in row)
    sum_a = sum(ai * (ai - 1) / 2 for ai in a)
    sum_b = sum(bj * (bj - 1) / 2 for bj in b)
    expected = sum_a * sum_b / (n * (n - 1) / 2) if n > 1 else 0
    max_val = (sum_a + sum_b) / 2
    
    if max_val == expected:
        return 1.0
    
    return (sum_comb - expected) / (max_val - expected)


def evaluate_clustering(pred_tree_str, seq_names, species_labels):
    """Evaluate how well a tree clusters sequences by species.
    Returns ARI score (0=random, 1=perfect clustering).
    """
    # Filter to sequences present in both tree and labels
    tree_leaves = set(get_leaf_names(pred_tree_str))
    valid_names = sorted([n for n in seq_names if n in tree_leaves and n in species_labels])
    if len(valid_names) < 4:
        return None
    
    # Prune tree to valid names
    pruned_tree = prune_tree_to_leaves(pred_tree_str, valid_names)
    if pruned_tree is None:
        return None
    
    # Get clusters from tree
    clusters = compute_clusters_from_tree(pruned_tree, valid_names)
    
    # Assign each sequence to its best cluster (the smallest cluster containing it)
    # Use hierarchical clustering: each sequence belongs to all its ancestor clusters
    # For ARI, we need a single cluster label per sequence.
    # Assign each leaf to the SMALLEST cluster that contains it.
    seq_to_cluster = {}
    clusters_sorted = sorted(clusters, key=len)
    for cluster in clusters_sorted:
        for seq in cluster:
            if seq not in seq_to_cluster:
                seq_to_cluster[seq] = tuple(cluster)  # use tuple as cluster ID
    
    # Build label arrays
    true_labels = [species_labels.get(n, "unknown") for n in valid_names]
    pred_labels = [str(seq_to_cluster.get(n, "singleton")) for n in valid_names]
    
    # Remove singletons for a cleaner signal
    non_singleton = [(tl, pl) for tl, pl in zip(true_labels, pred_labels) if pl != "singleton"]
    if len(non_singleton) < 2:
        return None
    
    true_labels_f = [x[0] for x in non_singleton]
    pred_labels_f = [x[1] for x in non_singleton]
    
    if len(set(true_labels_f)) < 2 or len(set(pred_labels_f)) < 2:
        return None
    
    return adjusted_rand_index(true_labels_f, pred_labels_f)


def hamming_distance(seq1, seq2):
    if len(seq1) != len(seq2):
        return 1.0
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-' and a != '.')
    valid = sum(1 for a, b in zip(seq1, seq2) if a != '-' and a != '.' and b != '-' and b != '.')
    if valid == 0:
        return 0.5
    return 1.0 - matches / valid


def seq_identity_distance(seq1, seq2):
    if len(seq1) != len(seq2):
        return 1.0
    identities = sum(1 for a, b in zip(seq1, seq2) if a == b and a not in ('-', '.', 'X'))
    aligned = sum(1 for a, b in zip(seq1, seq2)
                  if a not in ('-', '.', 'X') and b not in ('-', '.', 'X'))
    if aligned == 0:
        return 1.0
    return 1.0 - identities / aligned


def build_nj_tree_from_msa(sequences, seq_names, distance_func):
    from skbio import DistanceMatrix
    from skbio.tree import nj
    n = len(seq_names)
    dm_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_func(sequences[i], sequences[j])
            dm_matrix[i][j] = d
            dm_matrix[j][i] = d
    dm = DistanceMatrix(dm_matrix, seq_names)
    tree = nj(dm)
    return tree.__str__().replace(" ", "")


def load_msa_sequences(msa_path):
    seqs = {}
    name = None
    with open(msa_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = ""
            elif name:
                seqs[name] += line
    return seqs


def build_random_tree(seq_names):
    shuffled = seq_names[:]
    random.shuffle(shuffled)
    return "(" + ",".join(shuffled) + ");"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-pickle", default="virus_data/vogdb_treefam_v2.pickle")
    parser.add_argument("--pred-pickle", default="virus_data/phyla_predictions.pickle")
    parser.add_argument("--faa-dir", default="virus_data/faa")
    parser.add_argument("--msa-dir", default="virus_data/msa")
    parser.add_argument("--output-dir", default="eval_preds")
    parser.add_argument("--max-families", type=int, default=0, help="0=all")
    args = parser.parse_args()

    ref_path = os.path.join(SCRIPT_DIR, args.ref_pickle)
    pred_path = os.path.join(SCRIPT_DIR, args.pred_pickle)
    faa_dir = os.path.join(SCRIPT_DIR, args.faa_dir)
    msa_dir = os.path.join(SCRIPT_DIR, args.msa_dir)
    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    ref_data = pickle.load(open(ref_path, "rb"))
    pred_data = pickle.load(open(pred_path, "rb"))
    common = sorted(set(ref_data.keys()) & set(pred_data.keys()))
    print(f"  Common families: {len(common)}")

    if args.max_families > 0:
        common = common[:args.max_families]
        print(f"  Limited to {len(common)} families")

    # Evaluate each method
    methods = ["phyla", "hamming", "seqidentity", "random"]
    all_scores = {m: [] for m in methods}
    all_csv = {m: [] for m in methods}
    families_evaluated = 0
    families_skipped_no_labels = 0
    families_skipped_small = 0

    print(f"\nEvaluating ground-truth species clustering on {len(common)} families...")
    for i, vid in enumerate(common):
        # Get species labels from FAA
        faa_path = os.path.join(faa_dir, f"{vid}.faa")
        if not os.path.exists(faa_path):
            continue
        
        species_labels = parse_species_labels(faa_path)
        unique_species = set(species_labels.values())
        if len(unique_species) < 2:
            families_skipped_no_labels += 1
            continue

        # Get reference tree and find intersection
        ref_tree_str = ref_data[vid]["tree_newick"]
        ref_leaves = get_leaf_names(ref_tree_str)
        
        # --- PHYLA ---
        pred_tree_str = pred_data[vid]["pred_tree_newick"]
        pred_leaves = sorted(pred_data[vid].get("seq_names", []))
        
        common_leaves = sorted(set(ref_leaves) & set(pred_leaves))
        if len(common_leaves) < 4:
            families_skipped_small += 1
            continue
        
        # Prune both trees
        ref_pruned = prune_tree_to_leaves(ref_tree_str, common_leaves) if set(ref_leaves) != set(common_leaves) else ref_tree_str
        pred_pruned = prune_tree_to_leaves(pred_tree_str, common_leaves) if set(pred_leaves) != set(common_leaves) else pred_tree_str
        if ref_pruned is None or pred_pruned is None:
            families_skipped_small += 1
            continue

        # Evaluate PHYLA clustering
        ari_phyla = evaluate_clustering(pred_pruned, common_leaves, species_labels)
        
        # --- Hamming ---
        msa_path = os.path.join(msa_dir, f"{vid}.msa")
        ari_hamming = None
        ari_seqid = None
        if os.path.exists(msa_path):
            seqs = load_msa_sequences(msa_path)
            # Build NJ tree from MSA
            msa_names = sorted(set(seqs.keys()) & set(common_leaves))
            if len(msa_names) >= 4:
                msa_seqs = [seqs[n] for n in msa_names]
                try:
                    ham_tree = build_nj_tree_from_msa(msa_seqs, msa_names, hamming_distance)
                    ari_hamming = evaluate_clustering(ham_tree, msa_names, species_labels)
                    
                    seqid_tree = build_nj_tree_from_msa(msa_seqs, msa_names, seq_identity_distance)
                    ari_seqid = evaluate_clustering(seqid_tree, msa_names, species_labels)
                except:
                    pass

        # --- Random ---
        ari_random = None
        if len(common_leaves) >= 4:
            rand_tree = build_random_tree(common_leaves)
            ari_random = evaluate_clustering(rand_tree, common_leaves, species_labels)

        # Collect results
        if ari_phyla is not None:
            all_scores["phyla"].append(ari_phyla)
            all_csv["phyla"].append([vid, "phyla", f"{ari_phyla:.6f}", len(common_leaves), len(unique_species)])
        if ari_hamming is not None:
            all_scores["hamming"].append(ari_hamming)
            all_csv["hamming"].append([vid, "hamming", f"{ari_hamming:.6f}", len(common_leaves), len(unique_species)])
        if ari_seqid is not None:
            all_scores["seqidentity"].append(ari_seqid)
            all_csv["seqidentity"].append([vid, "seqidentity", f"{ari_seqid:.6f}", len(common_leaves), len(unique_species)])
        if ari_random is not None:
            all_scores["random"].append(ari_random)
            all_csv["random"].append([vid, "random", f"{ari_random:.6f}", len(common_leaves), len(unique_species)])

        families_evaluated += 1
        if (i + 1) % 2000 == 0:
            print(f"  [{i+1}/{len(common)}] evaluated={families_evaluated}, "
                  f"phyla_ARI={sum(all_scores['phyla'])/len(all_scores['phyla']):.4f} "
                  f"(n={len(all_scores['phyla'])})")

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"  GROUND-TRUTH SPECIES CLUSTERING EVALUATION")
    print(f"  Metric: Adjusted Rand Index (ARI)")
    print(f"  ARI=0: random clustering, ARI=1: perfect species grouping")
    print(f"{'='*60}")
    print(f"  Families evaluated:    {families_evaluated}")
    print(f"  Skipped (no labels):   {families_skipped_no_labels}")
    print(f"  Skipped (<4 seqs):     {families_skipped_small}")
    print()

    for m in methods:
        scores = all_scores[m]
        n = len(scores)
        if n == 0:
            print(f"  {m:<15}: 0 families")
            continue
        mean_ari = sum(scores) / n
        sorted_s = sorted(scores)
        median_ari = sorted_s[n // 2]
        std_ari = (sum((s - mean_ari) ** 2 for s in scores) / n) ** 0.5
        print(f"  {m:<15}: n={n:>6}  mean ARI={mean_ari:.4f}  median={median_ari:.4f}  "
              f"std={std_ari:.4f}")

    # Save CSVs
    for m in methods:
        if all_csv[m]:
            out_csv = os.path.join(output_dir, f"groundtruth_{m}_ari.csv")
            with open(out_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows([["vfam", "method", "ari", "n_seqs", "n_species"]])
                writer.writerows(all_csv[m])
            print(f"  Saved: {out_csv}")

    print(f"\n{'='*60}")
    print(f"  INTERPRETATION:")
    print(f"  ARI measures how well each method's tree clusters")
    print(f"  sequences by their true viral species identity.")
    print(f"  Higher ARI = better agreement with biological reality.")
    print(f"  This is a TRUE ground-truth metric (not algorithmic).")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
