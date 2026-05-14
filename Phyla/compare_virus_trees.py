"""
Compare PHYLA-predicted trees vs FastTree reference trees with proper baselines.
Uses EXACT algorithm from the paper's evo_reasoning_eval.py rf_distance().

Improvements over v1:
- Prunes trees to intersecting leaf sets instead of discarding mismatched families
  (recovers ~78% previously excluded data)
- Bootstrap confidence intervals for mean normRF
- Paired Wilcoxon signed-rank test between baselines
- Cohen's d effect size
- Stratified analysis by family size
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
    """Extract leaf names from a raw Newick string using Bio.Phylo."""
    t = Phylo.read(StringIO(tree_str), "newick")
    return sorted(str(x.name) for x in t.get_terminals())


def prune_tree_to_leaves(tree_str, leaves_to_keep):
    """Prune a tree (Newick string) to keep only specified leaves.
    Returns the pruned Newick string, or None if < 4 leaves remain."""
    clean = remove_branch_distances(tree_str)
    t = Tree(clean)
    tree_leaves = set(t.get_leaf_names())
    keep = sorted([l for l in leaves_to_keep if l in tree_leaves])
    if len(keep) < 4:
        return None
    t.prune(keep)
    return t.write(format=5).replace(" ", "").replace("'", "")


def compute_normrf(pred_tree_str, ref_tree_str):
    try:
        pred_clean = remove_branch_distances(pred_tree_str)
        ref_clean = remove_branch_distances(ref_tree_str)
        t1 = Tree(pred_clean)
        t2 = Tree(ref_clean)
        result = t1.compare(t2, unrooted=True)
        if isinstance(result["norm_rf"], str):
            return None
        return {"rf": int(result["rf"]), "max_rf": int(result["max_rf"]), "norm_rf": result["norm_rf"]}
    except:
        return None


def bootstrap_ci(values, n_resamples=10000, ci=0.95):
    """Compute bootstrap confidence interval for the mean."""
    n = len(values)
    if n == 0:
        return None, None, None
    means = []
    for _ in range(n_resamples):
        sample = [random.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lower_tail = (1.0 - ci) / 2.0
    lower_idx = int(lower_tail * n_resamples)
    upper_idx = int((1.0 - lower_tail) * n_resamples) - 1
    return sum(values) / n, means[lower_idx], means[upper_idx]


def paired_wilcoxon(x, y):
    """Paired Wilcoxon signed-rank test. Returns (statistic, p_value).
    Approximates p-value using normal approximation for n > 20."""
    n = len(x)
    if n != len(y) or n == 0:
        return None, None
    diffs = [x[i] - y[i] for i in range(n)]
    # Remove zero differences
    nonzero = [abs(d) for d in diffs if d != 0]
    signs = [1 if d > 0 else -1 for d in diffs if d != 0]
    n_nonzero = len(nonzero)
    if n_nonzero == 0:
        return 0.0, 1.0
    # Rank absolute differences
    sorted_pairs = sorted(zip(nonzero, signs))
    ranks = []
    i = 0
    while i < n_nonzero:
        j = i
        while j < n_nonzero and sorted_pairs[j][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            ranks.append(avg_rank)
        i = j
    W = sum(r * s for r, (_, s) in zip(ranks, sorted_pairs))
    # Normal approximation
    mu = n_nonzero * (n_nonzero + 1) / 4.0
    sigma = math.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24.0)
    z = (W - mu) / sigma if sigma > 0 else 0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return W, p


def cohens_d(x, y):
    """Cohen's d for paired samples (effect size)."""
    n = len(x)
    if n != len(y) or n < 2:
        return None
    diffs = [x[i] - y[i] for i in range(n)]
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    if var_diff == 0:
        return 0.0
    return mean_diff / math.sqrt(var_diff)


def hamming_distance(seq1, seq2):
    if len(seq1) != len(seq2):
        return 1.0
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-' and a != '.')
    valid = sum(1 for a, b in zip(seq1, seq2) if a != '-' and a != '.' and b != '-' and b != '.')
    if valid == 0:
        return 0.5
    return 1.0 - matches / valid


def build_nj_tree_from_hamming(sequences, seq_names):
    from skbio import DistanceMatrix
    from skbio.tree import nj
    n = len(seq_names)
    dm_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(sequences[i], sequences[j])
            dm_matrix[i][j] = d
            dm_matrix[j][i] = d
    dm = DistanceMatrix(dm_matrix, seq_names)
    tree = nj(dm)
    return tree.__str__().replace(" ", "")


def build_random_tree(seq_names):
    shuffled = seq_names[:]
    random.shuffle(shuffled)
    return "(" + ",".join(shuffled) + ");"


def evaluate_baseline(family_list, ref_data, pred_data, msa_dir, baseline_type, faa_dir):
    norm_rfs = []
    csv_rows = []
    metadata = []  # list of (vid, num_seqs, norm_rf)
    skipped = 0

    print(f"  Evaluating {baseline_type}...")
    for i, entry in enumerate(family_list):
        # family_list entries vary by context (see main() for format)
        if baseline_type == "phyla":
            vid, ref_tree_str, pred_tree_str, n_seqs = entry
        elif baseline_type == "hamming":
            vid, ref_tree_str, n_seqs = entry
            msa_path = os.path.join(msa_dir, f"{vid}.msa")
            if not os.path.exists(msa_path):
                skipped += 1
                continue
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
            seq_names = sorted(seqs.keys())
            if len(seq_names) < 4:
                skipped += 1
                continue
            seq_list = [seqs[n] for n in seq_names]
            try:
                pred_tree_str = build_nj_tree_from_hamming(seq_list, seq_names)
            except:
                skipped += 1
                continue
        elif baseline_type == "random":
            vid, ref_tree_str, ref_leaves = entry
            if len(ref_leaves) < 4:
                skipped += 1
                continue
            pred_tree_str = build_random_tree(ref_leaves)
            n_seqs = len(ref_leaves)
        else:
            continue

        metric = compute_normrf(pred_tree_str, ref_tree_str)
        if metric is None:
            skipped += 1
            continue

        norm_rfs.append(metric["norm_rf"])
        csv_rows.append([vid, baseline_type, str(metric["rf"]), str(metric["max_rf"]), f"{metric['norm_rf']:.4f}"])
        metadata.append((vid, n_seqs, metric["norm_rf"]))

        if (i + 1) % 1000 == 0:
            avg = sum(norm_rfs) / len(norm_rfs) if norm_rfs else 0
            print(f"    {baseline_type}: {i+1}/{len(family_list)}, evaluated={len(norm_rfs)}, skipped={skipped}, avg_normRF={avg:.6f}")

    return norm_rfs, csv_rows, skipped, metadata


def print_stats(norm_rfs, label):
    n = len(norm_rfs)
    if n == 0:
        print(f"  {label}: 0 families evaluated")
        return
    sorted_rfs = sorted(norm_rfs)
    mean_val = sum(norm_rfs) / n
    std_val = (sum((x - mean_val) ** 2 for x in norm_rfs) / n) ** 0.5
    # Bootstrap CI
    boot_mean, ci_lo, ci_hi = bootstrap_ci(norm_rfs, n_resamples=10000)
    print(f"  {label} ({n} families):")
    print(f"    Average normRF:  {mean_val:.6f}  [95% CI: {ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"    Median normRF:   {sorted_rfs[n//2]:.6f}")
    print(f"    Std normRF:      {std_val:.6f}")
    print(f"    Perfect (0):     {sum(1 for nv in norm_rfs if nv == 0.0)} ({sum(1 for nv in norm_rfs if nv == 0.0)/n*100:.1f}%)")
    worst = sum(1 for nv in norm_rfs if nv >= 0.98)
    print(f"    Worst  (>=0.98): {worst} ({worst/n*100:.1f}%)")
    print(f"    Distribution:")
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        cnt = sum(1 for nv in norm_rfs if lo <= nv < hi)
        print(f"      [{lo:.1f}, {hi:.1f}): {cnt:5d} ({cnt/n*100:.1f}%)")


def print_stratified(metadata, label):
    """Print normRF stratified by family size."""
    n = len(metadata)
    if n == 0:
        return
    # Stratify by number of sequences
    strata = {"small (4-10)": [], "medium (11-50)": [], "large (51+)": []}
    for vid, n_seqs, nr in metadata:
        if n_seqs <= 10:
            strata["small (4-10)"].append(nr)
        elif n_seqs <= 50:
            strata["medium (11-50)"].append(nr)
        else:
            strata["large (51+)"].append(nr)
    print(f"\n  {label} — Stratified by family size:")
    print(f"  {'Strata':<20} {'Families':>10} {'Avg normRF':>12} {'Perfect%':>10} {'Worst%':>10}")
    print(f"  {'-'*62}")
    for sname, snrs in strata.items():
        if len(snrs) == 0:
            continue
        s_avg = sum(snrs) / len(snrs)
        s_perf = sum(1 for x in snrs if x == 0.0) / len(snrs) * 100
        s_worst = sum(1 for x in snrs if x >= 0.98) / len(snrs) * 100
        print(f"  {sname:<20} {len(snrs):>10} {s_avg:>12.6f} {s_perf:>9.1f}% {s_worst:>9.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-pickle", default="virus_data/vogdb_treefam_v2.pickle")
    parser.add_argument("--pred-pickle", default="virus_data/phyla_predictions.pickle")
    parser.add_argument("--msa-dir", default="virus_data/msa")
    parser.add_argument("--faa-dir", default="virus_data/faa")
    parser.add_argument("--output-dir", default="eval_preds")
    parser.add_argument("--baselines", nargs="+", default=["phyla", "hamming", "random"])
    args = parser.parse_args()

    ref_path = os.path.join(SCRIPT_DIR, args.ref_pickle)
    pred_path = os.path.join(SCRIPT_DIR, args.pred_pickle)
    msa_dir = os.path.join(SCRIPT_DIR, args.msa_dir)
    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    ref_data = pickle.load(open(ref_path, "rb"))
    pred_data = pickle.load(open(pred_path, "rb"))
    common = sorted(set(ref_data.keys()) & set(pred_data.keys()))
    print(f"  Reference: {len(ref_data)} families")
    print(f"  Predictions: {len(pred_data)} families")
    print(f"  Common: {len(common)} families")

    # ----------------------------------------------------------------
    # Pre-process: prune trees to intersecting leaf sets
    # Instead of discarding families with mismatched leaves, we prune
    # the predicted tree (which has more leaves from FAA) to match the
    # reference tree's leaf set (which has fewer leaves from MSA).
    # In rare cases we also prune the reference tree.
    # ----------------------------------------------------------------
    print("\nPre-processing: pruning trees to intersecting leaf sets...")
    phyla_families = []   # (vid, ref_tree_str, pred_tree_str, n_seqs)
    hamming_families = [] # (vid, ref_tree_str, n_seqs)
    random_families = []  # (vid, ref_tree_str, ref_leaves)
    pruned_count = 0
    perfect_match_count = 0
    too_few_count = 0

    for vid in common:
        ref_tree_str = ref_data[vid]["tree_newick"]
        pred_tree_str = pred_data[vid]["pred_tree_newick"]
        pred_seq_names = sorted(pred_data[vid].get("seq_names", []))

        ref_leaves = get_leaf_names(ref_tree_str)
        pred_leaves = pred_seq_names

        # Compute intersection
        common_leaves = sorted(set(ref_leaves) & set(pred_leaves))
        if len(common_leaves) < 4:
            too_few_count += 1
            continue

        ref_needs_pruning = set(ref_leaves) != set(common_leaves)
        pred_needs_pruning = set(pred_leaves) != set(common_leaves)

        if ref_needs_pruning:
            pruned_ref = prune_tree_to_leaves(ref_tree_str, common_leaves)
            if pruned_ref is None:
                too_few_count += 1
                continue
            ref_tree_str_clean = pruned_ref
        else:
            ref_tree_str_clean = ref_tree_str

        if pred_needs_pruning:
            pruned_pred = prune_tree_to_leaves(pred_tree_str, common_leaves)
            if pruned_pred is None:
                too_few_count += 1
                continue
            pred_tree_str_clean = pruned_pred
        else:
            pred_tree_str_clean = pred_tree_str

        if ref_needs_pruning or pred_needs_pruning:
            pruned_count += 1
        else:
            perfect_match_count += 1

        n_seqs = len(common_leaves)
        phyla_families.append((vid, ref_tree_str_clean, pred_tree_str_clean, n_seqs))
        hamming_families.append((vid, ref_tree_str_clean, n_seqs))
        random_families.append((vid, ref_tree_str_clean, common_leaves))

    print(f"  Perfect leaf match (no pruning needed): {perfect_match_count}")
    print(f"  Pruned to intersection:                 {pruned_count}")
    print(f"  Too few leaves after pruning (<4):      {too_few_count}")
    print(f"  Total evaluable families:                {len(phyla_families)}")

    # ----------------------------------------------------------------
    # Evaluate each baseline
    # ----------------------------------------------------------------
    all_results = {}
    all_metadata = {}

    if "phyla" in args.baselines:
        print(f"\n--- Evaluating PHYLA on {len(phyla_families)} families ---")
        nrf, csv_rows, skipped, meta = evaluate_baseline(
            phyla_families, ref_data, pred_data, msa_dir, "phyla", args.faa_dir)
        all_results["phyla"] = {"norm_rfs": nrf, "csv_rows": csv_rows, "skipped": skipped}
        all_metadata["phyla"] = meta
        out_csv = os.path.join(output_dir, "virus_phyla_vs_fasttree.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows([["dataset", "model", "rf", "max_rf", "norm_rf"]])
            writer.writerows(csv_rows)
        print(f"  Saved: {out_csv} ({len(csv_rows)} rows)")

    if "hamming" in args.baselines:
        print(f"\n--- Evaluating Hamming + NJ on {len(hamming_families)} families ---")
        nrf, csv_rows, skipped, meta = evaluate_baseline(
            hamming_families, ref_data, pred_data, msa_dir, "hamming", args.faa_dir)
        all_results["hamming"] = {"norm_rfs": nrf, "csv_rows": csv_rows, "skipped": skipped}
        all_metadata["hamming"] = meta
        out_csv = os.path.join(output_dir, "virus_hamming_vs_fasttree.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows([["dataset", "model", "rf", "max_rf", "norm_rf"]])
            writer.writerows(csv_rows)
        print(f"  Saved: {out_csv} ({len(csv_rows)} rows)")

    if "random" in args.baselines:
        print(f"\n--- Evaluating Random on {len(random_families)} families ---")
        nrf, csv_rows, skipped, meta = evaluate_baseline(
            random_families, ref_data, pred_data, msa_dir, "random", args.faa_dir)
        all_results["random"] = {"norm_rfs": nrf, "csv_rows": csv_rows, "skipped": skipped}
        all_metadata["random"] = meta
        out_csv = os.path.join(output_dir, "virus_random_vs_fasttree.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows([["dataset", "model", "rf", "max_rf", "norm_rf"]])
            writer.writerows(csv_rows)
        print(f"  Saved: {out_csv} ({len(csv_rows)} rows)")

    # ----------------------------------------------------------------
    # Detailed per-baseline stats
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"   VOGDB Virus Benchmark: Multi-Baseline Comparison")
    print(f"   Trees pruned to intersecting leaf sets (recovered {pruned_count} families)")
    print(f"{'='*70}")
    for bl in args.baselines:
        if bl in all_results:
            print_stats(all_results[bl]["norm_rfs"],
                        f"{bl} (evaluated={len(all_results[bl]['norm_rfs'])}, skipped={all_results[bl]['skipped']})")
            if bl in all_metadata:
                print_stratified(all_metadata[bl], bl)

    # ----------------------------------------------------------------
    # Paired comparisons between PHYLA and Hamming
    # ----------------------------------------------------------------
    if "phyla" in all_results and "hamming" in all_results:
        print(f"\n{'='*70}")
        print(f"   Paired Comparison: PHYLA vs Hamming")
        print(f"{'='*70}")
        # Align on common families
        phyla_meta = all_metadata["phyla"]
        hamming_meta = all_metadata["hamming"]
        phyla_by_vid = {m[0]: m for m in phyla_meta}
        hamming_by_vid = {m[0]: m for m in hamming_meta}
        common_vids = sorted(set(phyla_by_vid.keys()) & set(hamming_by_vid.keys()))
        phyla_nrs = [phyla_by_vid[v][2] for v in common_vids]
        hamming_nrs = [hamming_by_vid[v][2] for v in common_vids]
        n_paired = len(common_vids)

        # Mean difference
        mean_diff = (sum(phyla_nrs) - sum(hamming_nrs)) / n_paired
        print(f"  Paired families: {n_paired}")
        print(f"  PHYLA mean:      {sum(phyla_nrs)/n_paired:.6f}")
        print(f"  Hamming mean:    {sum(hamming_nrs)/n_paired:.6f}")
        print(f"  Mean difference: {mean_diff:.6f} (PHYLA - Hamming)")

        # Paired Wilcoxon
        W, p = paired_wilcoxon(phyla_nrs, hamming_nrs)
        print(f"  Wilcoxon W:      {W:.2f}")
        print(f"  p-value:         {p:.6e}")
        print(f"  Significant?     {'YES (p<0.05)' if p < 0.05 else 'NO (p>=0.05)'}")

        # Cohen's d
        d = cohens_d(phyla_nrs, hamming_nrs)
        print(f"  Cohen's d:       {d:.4f} ({'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small' if abs(d) > 0.2 else 'negligible'})")

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  Summary Comparison Table (pruned to intersecting leaf sets):")
    print(f"  {'Baseline':<20} {'Families':>10} {'Avg_normRF':>12} {'Perfect%':>10} {'Worst%':>10}")
    print(f"  {'-'*62}")
    for bl in args.baselines:
        if bl in all_results:
            nf = len(all_results[bl]["norm_rfs"])
            avg = sum(all_results[bl]["norm_rfs"]) / nf if nf else 0
            perf_pct = sum(1 for x in all_results[bl]["norm_rfs"] if x == 0.0) / nf * 100 if nf else 0
            worst_pct = sum(1 for x in all_results[bl]["norm_rfs"] if x >= 0.98) / nf * 100 if nf else 0
            print(f"  {bl:<20} {nf:>10} {avg:>12.6f} {perf_pct:>9.1f}% {worst_pct:>9.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()