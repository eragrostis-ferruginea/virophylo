"""
Compare PHYLA-predicted trees vs FastTree reference trees with proper baselines.
Uses EXACT algorithm from the paper's evo_reasoning_eval.py rf_distance().
Critical fix: only compares trees with matching leaf sets (ete3 returns NA otherwise).
"""
import sys
import os
import pickle
import csv
import argparse
import re
import random

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
    skipped = 0

    print(f"  Evaluating {baseline_type}...")
    for i, (vid, ref_tree_str) in enumerate(family_list):
        if baseline_type == "phyla":
            pred_tree_str = pred_data[vid]["pred_tree_newick"]
        elif baseline_type == "hamming":
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
            seq_names = list(ref_data[vid].get("sequences", {}).keys())
            if len(seq_names) < 4:
                skipped += 1
                continue
            pred_tree_str = build_random_tree(seq_names)
        else:
            continue

        metric = compute_normrf(pred_tree_str, ref_tree_str)
        if metric is None:
            skipped += 1
            continue

        norm_rfs.append(metric["norm_rf"])
        csv_rows.append([vid, baseline_type, str(metric["rf"]), str(metric["max_rf"]), f"{metric['norm_rf']:.4f}"])

        if (i + 1) % 1000 == 0:
            avg = sum(norm_rfs) / len(norm_rfs) if norm_rfs else 0
            print(f"    {baseline_type}: {i+1}/{len(family_list)}, evaluated={len(norm_rfs)}, skipped={skipped}, avg_normRF={avg:.6f}")

    return norm_rfs, csv_rows, skipped


def print_stats(norm_rfs, label):
    n = len(norm_rfs)
    if n == 0:
        print(f"  {label}: 0 families evaluated")
        return
    sorted_rfs = sorted(norm_rfs)
    print(f"  {label} ({n} families):")
    print(f"    Average normRF:  {sum(norm_rfs)/n:.6f}")
    print(f"    Median normRF:   {sorted_rfs[n//2]:.6f}")
    print(f"    Std normRF:      {(sum((x-sum(norm_rfs)/n)**2 for x in norm_rfs)/n)**0.5:.6f}")
    print(f"    Perfect (0):     {sum(1 for nv in norm_rfs if nv == 0.0)} ({sum(1 for nv in norm_rfs if nv == 0.0)/n*100:.1f}%)")
    worst = sum(1 for nv in norm_rfs if nv >= 0.98)
    print(f"    Worst  (>=0.98): {worst} ({worst/n*100:.1f}%)")
    print(f"    Distribution:")
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        cnt = sum(1 for nv in norm_rfs if lo <= nv < hi)
        print(f"      [{lo:.1f}, {hi:.1f}): {cnt:5d} ({cnt/n*100:.1f}%)")


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

    print("\nIdentifying families with matching leaf sets...")
    leaf_match = []
    leaf_mismatch = []
    for vid in common:
        ref_leaves = get_leaf_names(ref_data[vid]["tree_newick"])
        pred_leaves = sorted(pred_data[vid].get("seq_names", []))
        if ref_leaves == pred_leaves:
            leaf_match.append((vid, ref_data[vid]["tree_newick"]))
        else:
            leaf_mismatch.append((vid, len(ref_leaves), len(pred_leaves)))

    n_matched = len(leaf_match)
    n_mismatched = len(leaf_mismatch)
    print(f"  Matched leaf sets:  {n_matched} ({n_matched/len(common)*100:.1f}%)")
    print(f"  Mismatched leaves:  {n_mismatched} ({n_mismatched/len(common)*100:.1f}%)")
    if leaf_mismatch:
        mismatch_by_ref = {}
        for _, rl, pl in leaf_mismatch:
            diff = rl - pl
            key = "ref_smaller" if diff < 0 else "pred_smaller" if diff > 0 else "ref_larger"
            mismatch_by_ref[key] = mismatch_by_ref.get(key, 0) + 1
        for k, v in sorted(mismatch_by_ref.items()):
            print(f"    {k}: {v}")

    all_results = {}
    for bl in args.baselines:
        print(f"\n--- Evaluating {bl} on {n_matched} matched families ---")
        norm_rfs, csv_rows, skipped = evaluate_baseline(leaf_match, ref_data, pred_data, msa_dir, bl, args.faa_dir)
        all_results[bl] = {"norm_rfs": norm_rfs, "csv_rows": csv_rows, "skipped": skipped}

        out_csv = os.path.join(output_dir, f"virus_{bl}_vs_fasttree.csv")
        header = [["dataset", "model", "rf", "max_rf", "norm_rf"]]
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(header)
            writer.writerows(csv_rows)
        print(f"  Saved: {out_csv} ({len(csv_rows)} rows)")

    print(f"\n{'='*70}")
    print(f"   VOGDB Virus Benchmark: Multi-Baseline Comparison")
    print(f"   Only families with matching leaf sets between ref and pred")
    n_total = n_matched
    print(f"{'='*70}")
    for bl in args.baselines:
        norm_rfs = all_results[bl]["norm_rfs"]
        skipped = all_results[bl]["skipped"]
        print_stats(norm_rfs, f"{bl} (evaluated={len(norm_rfs)}, skipped={skipped})")

    print(f"\n{'='*70}")
    print("  Summary Comparison Table (matched leaf sets only):")
    print(f"  {'Baseline':<20} {'Families':>10} {'Avg_normRF':>12} {'Perfect%':>10}")
    print(f"  {'-'*52}")
    for bl in args.baselines:
        nf = len(all_results[bl]["norm_rfs"])
        avg = sum(all_results[bl]["norm_rfs"]) / nf if nf else 0
        perf_pct = sum(1 for x in all_results[bl]["norm_rfs"] if x == 0.0) / nf * 100 if nf else 0
        print(f"  {bl:<20} {nf:>10} {avg:>12.6f} {perf_pct:>9.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()