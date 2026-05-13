"""
Compare PHYLA-predicted trees vs FastTree reference trees.
Uses EXACT algorithm from the paper's evo_reasoning_eval.py rf_distance():
  1. Bio.Phylo.read → set branch_length=None → Phylo.write → remove :digits
  2. ete3.Tree → compare(unrooted=True) → normRF
  3. try/except: skip families with invalid tree formats (exactly as paper does)
"""
import sys
import os
import pickle
import csv
import argparse

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
    new_str_obj.seek(0)
    new_str = new_str_obj.getvalue()

    dist_decimals = 8
    while True:
        try:
            curr_index = new_str.index(":")
            new_str = new_str[:curr_index] + new_str[curr_index + dist_decimals:]
        except:
            return new_str


def compute_normrf(pred_tree_str, ref_tree_str):
    try:
        pred_clean = remove_branch_distances(pred_tree_str)
        ref_clean = remove_branch_distances(ref_tree_str)

        t1 = Tree(pred_clean)
        t2 = Tree(ref_clean)
        result = t1.compare(t2, unrooted=True)
        return {
            "rf": int(result["rf"]),
            "max_rf": int(result["max_rf"]),
            "norm_rf": result["norm_rf"]
        }
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description="Compare PHYLA vs FastTree trees")
    parser.add_argument("--ref-pickle", default="virus_data/vogdb_treefam_v2.pickle",
                        help="FastTree reference trees pickle")
    parser.add_argument("--pred-pickle", default="virus_data/phyla_predictions.pickle",
                        help="PHYLA predictions pickle")
    parser.add_argument("--output-csv", default="eval_preds/virus_phyla_vs_fasttree.csv",
                        help="Output CSV with normRF results")
    parser.add_argument("--output-pickle", default="virus_data/virus_eval_full.pickle",
                        help="Output pickle with combined data")
    args = parser.parse_args()

    ref_path = os.path.join(SCRIPT_DIR, args.ref_pickle)
    pred_path = os.path.join(SCRIPT_DIR, args.pred_pickle)
    output_dir = os.path.dirname(os.path.join(SCRIPT_DIR, args.output_csv))
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(ref_path):
        print(f"ERROR: reference pickle not found: {ref_path}")
        sys.exit(1)
    if not os.path.exists(pred_path):
        print(f"ERROR: predictions pickle not found: {pred_path}")
        sys.exit(1)

    print("Loading reference trees...")
    ref_data = pickle.load(open(ref_path, "rb"))
    print(f"  {len(ref_data)} families")

    print("Loading PHYLA predictions...")
    pred_data = pickle.load(open(pred_path, "rb"))
    print(f"  {len(pred_data)} families")

    common = set(ref_data.keys()) & set(pred_data.keys())
    print(f"Common families: {len(common)}")

    if len(common) == 0:
        print("ERROR: no common families found between ref and pred!")
        sys.exit(1)

    csv_rows = [["dataset", "model", "rf", "max_rf", "norm_rf"]]
    norm_rfs = []
    failures = []
    skipped = 0

    for i, vid in enumerate(sorted(common)):
        ref_tree_str = ref_data[vid]["tree_newick"]
        pred_tree_str = pred_data[vid]["pred_tree_newick"]

        metric = compute_normrf(pred_tree_str, ref_tree_str)
        if metric is None:
            skipped += 1
            continue

        rf, max_rf, norm_rf = metric["rf"], metric["max_rf"], metric["norm_rf"]
        norm_rfs.append(norm_rf)
        csv_rows.append([vid, "Phyla-beta", str(rf), str(max_rf), f"{norm_rf:.4f}"])

        if (i + 1) % 2000 == 0:
            print(f"  Progress: {i+1}/{len(common)}, skipped={skipped}, "
                  f"running avg_normRF={sum(norm_rfs)/len(norm_rfs):.6f}")

    with open(os.path.join(SCRIPT_DIR, args.output_csv), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"\nCSV saved: {args.output_csv} ({len(csv_rows)-1} rows)")

    sorted_rfs = sorted(norm_rfs)
    n = len(norm_rfs)
    print(f"\n{'='*60}")
    print(f"   Virus Phylogeny: PHYLA vs FastTree Reference Trees")
    print(f"   Method: exact rf_distance from evo_reasoning_eval.py")
    print(f"{'='*60}")
    print(f"  Total common families: {len(common)}")
    print(f"  Families evaluated:    {n}")
    print(f"  Skipped (invalid fmt): {skipped} ({skipped/len(common)*100:.1f}%)")
    if n > 0:
        print(f"  Average normRF:        {sum(norm_rfs)/n:.6f}")
        print(f"  Median normRF:         {sorted_rfs[n//2]:.6f}")
        print(f"  Min normRF:            {min(norm_rfs):.6f}")
        print(f"  Max normRF:            {max(norm_rfs):.6f}")
        print(f"  Std normRF:            {(sum((x-sum(norm_rfs)/n)**2 for x in norm_rfs)/n)**0.5:.6f}")

        perfect = sum(1 for nv in norm_rfs if nv == 0.0)
        print(f"\n  Perfect (normRF=0):    {perfect} ({perfect/n*100:.1f}%)")
        worst = sum(1 for nv in norm_rfs if nv >= 0.98)
        print(f"  Worst (normRF>=0.98):  {worst} ({worst/n*100:.1f}%)")

        print(f"\n  Distribution:")
        for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
            cnt = sum(1 for nv in norm_rfs if lo <= nv < hi)
            print(f"    [{lo:.1f}, {hi:.1f}): {cnt:5d} ({cnt/n*100:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()