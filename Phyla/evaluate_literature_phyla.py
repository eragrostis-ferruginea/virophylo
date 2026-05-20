#!/usr/bin/env python3
"""
PHYLA-beta inference on Literature Reference Trees (Brown & Firth 2025 RdRp).
303 OTU-level expert phylogenies. GPU (A100) required.

Usage: sbatch run_lit_phyla_slurm.sh
"""
import sys, os, re, pickle, csv, argparse, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "phyla"))

import torch
from Bio import SeqIO
from phyla import phyla
from phyla.utils.eval_configs import Config, Mamba_ModelConfig
from ete3 import Tree
from Bio import Phylo
from io import StringIO


def remove_branch_distances(tree_str):
    t = Phylo.read(StringIO(tree_str), "newick")
    for n in t.get_nonterminals():
        n.branch_length = None
    for n in t.get_terminals():
        n.branch_length = None
    o = StringIO()
    Phylo.write(t, o, "newick")
    s = o.getvalue()
    s = re.sub(r":[^,();\n]*", "", s)
    return s.replace("'", "")


def get_leaf_names(tree_str):
    return sorted(str(x.name) for x in Phylo.read(StringIO(tree_str), "newick").get_terminals())


def prune_tree_to_leaves(tree_str, leaves):
    clean = remove_branch_distances(tree_str)
    t = Tree(clean)
    tree_leaf_names = t.get_leaf_names()
    keep_names = sorted([l for l in leaves if l in tree_leaf_names])
    if len(keep_names) < 4:
        return None
    try:
        t.prune(keep_names)
        return t.write(format=5).replace(" ", "").replace("'", "")
    except Exception as e:
        if "Ambiguous" in str(e):
            from collections import Counter
            needed = Counter(keep_names)
            keep_nodes = []
            for leaf in t.get_leaves():
                if leaf.name in needed and needed[leaf.name] > 0:
                    keep_nodes.append(leaf)
                    needed[leaf.name] -= 1
            if len(keep_nodes) >= len(keep_names):
                try:
                    t.prune(keep_nodes)
                    return t.write(format=5).replace(" ", "").replace("'", "")
                except Exception:
                    pass
        return None


def compute_normrf(pred_str, ref_str):
    try:
        t1 = Tree(remove_branch_distances(pred_str))
        t2 = Tree(remove_branch_distances(ref_str))
        r = t1.compare(t2, unrooted=True)
        if isinstance(r["norm_rf"], str):
            return None
        return {"rf": int(r["rf"]), "max_rf": int(r["max_rf"]), "norm_rf": r["norm_rf"]}
    except:
        return None


def match_names_to_tree_leaves(seq_names, ref_leaves):
    seq_set = set(seq_names)
    leaf_set = set(ref_leaves)
    exact = seq_set & leaf_set
    matched = [(n, n) for n in exact]
    used_seq = set(exact)
    used_leaf = set(exact)
    remaining_seq = seq_set - used_seq
    remaining_leaf = leaf_set - used_leaf
    for sname in list(remaining_seq):
        for lname in list(remaining_leaf):
            if lname.startswith(sname + "|") or lname.startswith(sname + ":"):
                matched.append((sname, lname))
                used_seq.add(sname); used_leaf.add(lname); break
    remaining_seq = seq_set - used_seq; remaining_leaf = leaf_set - used_leaf
    for sname in list(remaining_seq):
        base = sname.rsplit(".", 1)[0] if "." in sname else sname
        for lname in list(remaining_leaf):
            if lname.startswith(base + "|") or lname.startswith(base + ":") or lname.startswith(base + "."):
                matched.append((sname, lname)); used_seq.add(sname); used_leaf.add(lname); break
    remaining_seq = seq_set - used_seq; remaining_leaf = leaf_set - used_leaf
    for sname in list(remaining_seq):
        for lname in list(remaining_leaf):
            first_part = lname.split("|")[0].split(":")[0]
            if sname == first_part or (sname.endswith(".1") and sname[:-2] == first_part) or \
               (first_part.endswith(".1") and first_part[:-2] == sname) or sname == first_part:
                matched.append((sname, lname)); used_seq.add(sname); used_leaf.add(lname); break
    return matched, seq_set - used_seq, leaf_set - used_leaf


def rename_tree_leaves(tree_str, name_map):
    """Rename tree leaves using a mapping dict {old_name: new_name}."""
    t = Tree(tree_str)
    for leaf in t:
        old_name = leaf.name
        if old_name in name_map:
            leaf.name = name_map[old_name]
    return t.write(format=5).replace(" ", "").replace("'", "")


def encode_with_phyla(sequences, sequence_names, model, device):
    from phyla.dataset.data import Arbitrary_Sequence_Dataset
    dataset = Arbitrary_Sequence_Dataset()
    batch, names = dataset.encode_sequences(sequences, sequence_names)

    with torch.no_grad():
        preds = model(
            batch['encoded_sequences'].to(device),
            batch['sequence_mask'].to(device),
            batch['cls_positions'].bool().to(device)
        )

    tree = model.reconstruct_tree(preds, names)
    return str(tree)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default="/home/jianpinhe3/virophylo/virus_data/literature_refs/literature_dataset.pickle")
    parser.add_argument("--output-dir",
                        default="/home/jianpinhe3/virophylo/Phyla/eval_preds/literature")
    parser.add_argument("--checkpoint", default="weights/11564369")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-seqs", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print(" PHYLA-beta on LITERATURE REFERENCE TREES")
    print(" Brown & Firth 2025 RdRp — 303 OTU-level expert phylogenies")
    print(f" Device: {args.device}")
    print("=" * 65)

    # Load data
    ds_path = os.path.join(SCRIPT_DIR, args.dataset)
    with open(ds_path, "rb") as f:
        ds = pickle.load(f)

    otu_keys = sorted([k for k in ds if k.startswith("OTUs_newick_")])
    print(f"Total families: {len(ds)}")
    print(f"OTU families: {len(otu_keys)}")

    # Load model
    ckpt_path = os.path.join(SCRIPT_DIR, args.checkpoint)
    print(f"\nLoading PHYLA-beta from {ckpt_path}...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    config = Config()
    config.model = Mamba_ModelConfig()
    config.model.d_model = 256
    config.model.n_layer = 16
    config.model.vocab_size = 24
    config.model.num_blocks = 3
    config.model.model_name = 'Phyla-beta'
    config.model.bidirectional = True
    config.model.bidirectional_strategy = "add"
    config.model.bidirectional_weight_tie = True

    model = phyla(config, device=args.device).load(ckpt_path)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count/1e6:.0f}M")

    # Evaluate
    all_results = []
    skipped = 0
    t_start = time.time()

    for i, fid in enumerate(otu_keys):
        entry = ds[fid]
        seqs = entry["sequences"]
        ref_tree_str = entry["tree_newick"]
        seq_names = sorted(seqs.keys())

        if len(seq_names) < args.min_seqs:
            skipped += 1
            continue

        ref_leaves = get_leaf_names(ref_tree_str)
        matched_pairs, _, _ = match_names_to_tree_leaves(seq_names, ref_leaves)

        if len(matched_pairs) >= args.min_seqs:
            eval_seq_names = [s for s, l in matched_pairs]
            eval_leaf_names = [l for s, l in matched_pairs]
            name_map = dict(matched_pairs)  # dataset_key -> tree_leaf_name
        elif len(seq_names) >= args.min_seqs:
            eval_seq_names = seq_names
            eval_leaf_names = None
            name_map = None
        else:
            skipped += 1
            continue

        if eval_leaf_names and set(ref_leaves) != set(eval_leaf_names):
            pruned = prune_tree_to_leaves(ref_tree_str, eval_leaf_names)
            if pruned is None:
                skipped += 1; continue
            ref_tree_str = pruned

        seq_list = [seqs[n] for n in eval_seq_names]

        try:
            phyla_tree_str = encode_with_phyla(seq_list, eval_seq_names, model, device)

            # Rename prediction tree leaves to match reference tree leaf names
            if name_map:
                phyla_tree_str = rename_tree_leaves(phyla_tree_str, name_map)

            metric = compute_normrf(phyla_tree_str, ref_tree_str)

            if metric:
                all_results.append({
                    "family": fid,
                    "n_seqs": len(eval_seq_names),
                    "method": "PHYLA",
                    "normRF": metric["norm_rf"],
                    "rf": metric["rf"],
                    "max_rf": metric["max_rf"],
                })
                print(f"  [{i+1}/{len(otu_keys)}] {fid}: "
                      f"{len(eval_seq_names):>4} seqs, normRF={metric['norm_rf']:.4f}")
            else:
                skipped += 1
                print(f"  [{i+1}/{len(otu_keys)}] {fid}: normRF failed")
        except Exception as e:
            skipped += 1
            print(f"  [{i+1}/{len(otu_keys)}] {fid}: ERROR {str(e)[:80]}")

    elapsed = time.time() - t_start
    rate = len(otu_keys) / elapsed * 3600 if elapsed > 0 else 0

    # Summary
    print(f"\n{'='*65}")
    print("  PHYLA-beta vs LITERATURE REFERENCE TREES — RESULTS")
    print(f"{'='*65}")
    print(f"  Elapsed: {elapsed:.0f}s ({rate:.0f} fams/hr)")
    print(f"  Skipped: {skipped}")

    phyla_results = [r for r in all_results]
    n = len(phyla_results)
    if n > 0:
        avg = sum(r["normRF"] for r in phyla_results) / n
        sorted_nrf = sorted(r["normRF"] for r in phyla_results)
        med = sorted_nrf[n // 2]
        perf_pct = sum(1 for r in phyla_results if r["normRF"] == 0) / n * 100
        worst_pct = sum(1 for r in phyla_results if r["normRF"] >= 0.98) / n * 100
        print(f"  Families:     {n}")
        print(f"  Avg normRF:   {avg:.6f}")
        print(f"  Median normRF:{med:.6f}")
        print(f"  Perfect%:     {perf_pct:.1f}%")
        print(f"  Worst%:       {worst_pct:.1f}%")

    out_csv = os.path.join(args.output_dir, "literature_phyla.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "n_seqs", "method", "normRF", "rf", "max_rf"])
        for r in all_results:
            w.writerow([r["family"], r["n_seqs"], r["method"],
                       f"{r['normRF']:.4f}", r["rf"], r["max_rf"]])
    print(f'\n  Saved: {out_csv}')


if __name__ == "__main__":
    main()