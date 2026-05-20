#!/usr/bin/env python3
"""
Literature reference tree evaluation — Baselines (Hamming/SeqID/Random)
Brown & Firth 2025 RdRp — 303 OTU-level expert phylogenies

Parallelized MSA, no timeout, progress every family.
Usage: sbatch run_lit_baselines_slurm.sh
"""
import os, sys, re, csv, pickle, argparse, random, math, subprocess, tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "phyla"))

from ete3 import Tree
from Bio import Phylo, SeqIO
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


def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        return 1.0
    v = sum(1 for a, b in zip(s1, s2) if a != "-" and a != "." and b != "-" and b != ".")
    if v == 0:
        return 0.5
    return 1.0 - sum(1 for a, b in zip(s1, s2) if a == b and a not in "-.") / v


def seqid_distance(s1, s2):
    if len(s1) != len(s2):
        return 1.0
    ident = sum(1 for a, b in zip(s1, s2) if a == b and a not in "-.")
    align = sum(1 for a, b in zip(s1, s2) if a not in "-." and b not in "-.")
    return 1.0 - ident / align if align else 1.0


def build_nj_tree(sequences, names, dist_func):
    from skbio import DistanceMatrix
    from skbio.tree import nj
    n = len(names)
    dm = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_func(sequences[i], sequences[j])
            dm[i][j] = dm[j][i] = d
    return nj(DistanceMatrix(dm, names)).__str__().replace(" ", "")


def build_random_tree(names):
    s = names[:]
    random.shuffle(s)
    return "(" + ",".join(s) + ");"


def run_mafft_single(seq_dict, names, tmp_dir):
    raw_fa = os.path.join(tmp_dir, "raw.fa")
    ali_fa = os.path.join(tmp_dir, "aligned.fa")
    with open(raw_fa, "w") as f:
        for name in names:
            f.write(f">{name}\n{seq_dict[name]}\n")

    n_seqs = len(names)
    if n_seqs > 200:
        cmd = ["mafft", "--quiet", "--parttree", "--thread", "1", raw_fa]
    elif n_seqs > 50:
        cmd = ["mafft", "--quiet", "--retree", "1", "--maxiterate", "2", "--thread", "1", raw_fa]
    else:
        cmd = ["mafft", "--quiet", "--auto", raw_fa]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not result.stdout.strip():
        if n_seqs > 200 and result.returncode != 0:
            fallback = ["mafft", "--quiet", "--retree", "1", "--maxiterate", "2", "--thread", "1", raw_fa]
            result = subprocess.run(fallback, capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            return None, result.stderr.strip()[:200] if result.stderr else f"rc={result.returncode}"

    with open(ali_fa, "w") as f:
        f.write(result.stdout)
    ali_seqs = {}
    for rec in SeqIO.parse(ali_fa, "fasta"):
        ali_seqs[rec.id] = str(rec.seq)
    return ali_seqs, ""


def match_names_to_tree_leaves(seq_names, ref_leaves):
    """Fuzzy-match dataset sequence names to tree leaf names.

    Dataset keys are extracted accessions (e.g. 'YP_009328360.1' or 'AEF56735').
    Tree leaves are full names (e.g. 'YP_009328360.1|virus|refseq:0.12').

    Returns (matched_pairs: list of (seq_name, leaf_name), unmatched_seqs, unmatched_leaves).
    """
    seq_set = set(seq_names)
    leaf_set = set(ref_leaves)

    # 1. Exact match
    exact = seq_set & leaf_set
    matched = [(n, n) for n in exact]
    used_seq = set(exact)
    used_leaf = set(exact)

    # 2. Seq key is a prefix of leaf name (accession without version)
    remaining_seq = seq_set - used_seq
    remaining_leaf = leaf_set - used_leaf
    for sname in list(remaining_seq):
        for lname in list(remaining_leaf):
            if lname.startswith(sname + "|") or lname.startswith(sname + ":"):
                matched.append((sname, lname))
                used_seq.add(sname)
                used_leaf.add(lname)
                break

    # 3. Seq key without version matches leaf prefix
    remaining_seq = seq_set - used_seq
    remaining_leaf = leaf_set - used_leaf
    for sname in list(remaining_seq):
        base = sname.rsplit(".", 1)[0] if "." in sname else sname
        for lname in list(remaining_leaf):
            if lname.startswith(base + "|") or lname.startswith(base + ":") or \
               lname.startswith(base + "."):
                matched.append((sname, lname))
                used_seq.add(sname)
                used_leaf.add(lname)
                break

    # 4. Leaf name starts with seq key as substring
    remaining_seq = seq_set - used_seq
    remaining_leaf = leaf_set - used_leaf
    for sname in list(remaining_seq):
        for lname in list(remaining_leaf):
            first_part = lname.split("|")[0].split(":")[0]
            if sname == first_part or (sname.endswith(".1") and sname[:-2] == first_part) or \
               (first_part.endswith(".1") and first_part[:-2] == sname) or \
               sname == first_part:
                matched.append((sname, lname))
                used_seq.add(sname)
                used_leaf.add(lname)
                break

    return matched, seq_set - used_seq, leaf_set - used_leaf


def rename_tree_leaves(tree_str, name_map):
    """Rename tree leaves using a mapping dict {old_name: new_name}."""
    t = Tree(tree_str)
    for leaf in t:
        old_name = leaf.name
        if old_name in name_map:
            leaf.name = name_map[old_name]
    return t.write(format=5).replace(" ", "").replace("'", "")


def process_family(args):
    """Process a single family: MSA + all baselines."""
    fid, entry, min_seqs = args
    seqs = entry["sequences"]
    ref_tree_str = entry["tree_newick"]
    seq_names = sorted(seqs.keys())

    diag = {"fid": fid, "n_dataset_seqs": len(seq_names), "stage": "ok", "reason": ""}

    if len(seq_names) < min_seqs:
        diag["stage"] = "too_few_dataset"
        diag["reason"] = f"dataset has {len(seq_names)} < {min_seqs}"
        return [], diag

    ref_leaves = get_leaf_names(ref_tree_str)

    # Fuzzy match dataset sequence names to tree leaf names
    matched_pairs, unmatched_seqs, unmatched_leaves = match_names_to_tree_leaves(seq_names, ref_leaves)
    diag["n_matched"] = len(matched_pairs)
    diag["n_ref_leaves"] = len(ref_leaves)

    # Use ALL available sequences (dataset was built for this tree)
    # but map them to tree leaves for pruning
    if len(matched_pairs) >= min_seqs:
        eval_seq_names = [s for s, l in matched_pairs]
        eval_leaf_names = [l for s, l in matched_pairs]
        name_map = dict(matched_pairs)
    elif len(seq_names) >= min_seqs:
        eval_seq_names = seq_names
        eval_leaf_names = None  # skip pruning
        name_map = None
    else:
        diag["stage"] = "too_few_matched"
        diag["reason"] = f"matched={len(matched_pairs)} dataset={len(seq_names)} < {min_seqs}"
        return [], diag

    diag["n_eval_seqs"] = len(eval_seq_names)

    # Prune reference tree to matched leaves
    if eval_leaf_names and len(eval_leaf_names) >= min_seqs:
        if set(ref_leaves) != set(eval_leaf_names):
            pruned = prune_tree_to_leaves(ref_tree_str, eval_leaf_names)
            if pruned is None:
                diag["stage"] = "prune_failed"
                diag["reason"] = "prune returned None"
                return [], diag
            ref_tree_str = pruned

    tmp_dir = tempfile.mkdtemp(prefix=f"lit_{fid.replace('/', '_')}_")
    os.makedirs(tmp_dir, exist_ok=True)

    common_seqs = {n: seqs[n] for n in eval_seq_names}
    ali, mafft_err = run_mafft_single(common_seqs, eval_seq_names, tmp_dir)

    results = []
    if ali is None:
        diag["stage"] = "mafft_none"
        diag["reason"] = f"MAFFT failed: {mafft_err}"
        import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
        return results, diag
    if len(ali) < min_seqs:
        diag["stage"] = "mafft_too_few"
        diag["reason"] = f"MAFFT returned {len(ali)} seqs < {min_seqs}"
        import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
        return results, diag

    ali_names = sorted(set(ali.keys()) & set(eval_seq_names))
    if len(ali_names) < min_seqs:
        diag["stage"] = "overlap_too_few"
        diag["reason"] = f"alignment overlap={len(ali_names)} < {min_seqs}"
        import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
        return results, diag

    diag["n_aligned"] = len(ali_names)

    seq_list = [ali[n] for n in ali_names]

    methods = ["Hamming", "SeqIdentity", "random"]
    normrf_failures = []
    for method in methods:
        if method in ("Hamming", "SeqIdentity"):
            dist_func = hamming_distance if method == "Hamming" else seqid_distance
            try:
                pred_tree_str = build_nj_tree(seq_list, ali_names, dist_func)
            except Exception as e:
                normrf_failures.append(f"{method}_nj_failed:{e}")
                continue
        elif method == "random":
            pred_tree_str = build_random_tree(ali_names)

        # Rename prediction tree leaves to match reference tree leaf names
        if name_map:
            pred_tree_str = rename_tree_leaves(pred_tree_str, name_map)

        metric = compute_normrf(pred_tree_str, ref_tree_str)
        if metric is None:
            normrf_failures.append(f"{method}_normrf_none")
            continue

        results.append({
            "family": fid,
            "n_seqs": len(ali_names),
            "method": method,
            "normRF": metric["norm_rf"],
            "rf": metric["rf"],
            "max_rf": metric["max_rf"],
        })

    if not results and normrf_failures:
        diag["stage"] = "normrf_failed"
        diag["reason"] = "; ".join(normrf_failures)

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results, diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default="/home/jianpinhe3/virophylo/virus_data/literature_refs/literature_dataset.pickle")
    parser.add_argument("--output-dir",
                        default="/home/jianpinhe3/virophylo/Phyla/eval_preds/literature")
    parser.add_argument("--min-seqs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers for MSA (default: CPU count)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print(" LITERATURE REFERENCE TREE EVALUATION — BASELINES")
    print(" Brown & Firth 2025 RdRp — 303 OTU-level expert phylogenies")
    print(f" Workers: {args.workers or mp.cpu_count()}")
    print("=" * 65)

    with open(args.dataset, "rb") as f:
        ds = pickle.load(f)

    otu_keys = sorted([k for k in ds if k.startswith("OTUs_newick_")])
    print(f"Total families: {len(ds)}")
    print(f"OTU families: {len(otu_keys)}")

    tasks = [(fid, ds[fid], args.min_seqs) for fid in otu_keys]
    n_workers = args.workers or mp.cpu_count()

    all_results = []
    diags = []
    done_count = 0
    total = len(tasks)
    stage_counts = defaultdict(int)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_family, task): task[0] for task in tasks}
        for future in as_completed(futures):
            fid = futures[future]
            try:
                res, diag = future.result()
                all_results.extend(res)
                diags.append(diag)
                stage_counts[diag["stage"]] += 1
            except Exception as e:
                print(f"  ERROR {fid}: {e}")
                stage_counts["exception"] += 1
            done_count += 1
            if done_count % 10 == 0 or done_count == total:
                print(f"  [{done_count}/{total}] families processed, "
                      f"{len(all_results)} results so far")

    # Diagnostic summary
    print(f"\n{'='*65}")
    print("  DIAGNOSTIC SUMMARY — where families were lost")
    print(f"{'='*65}")
    for stage, count in sorted(stage_counts.items()):
        print(f"  {stage:<25} {count:>5} ({count/total*100:.1f}%)")

    # Write full diagnostics
    diag_csv = os.path.join(args.output_dir, "baselines_diagnostics.csv")
    with open(diag_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "n_dataset_seqs", "n_ref_leaves", "n_matched",
                     "n_eval_seqs", "n_aligned", "stage", "reason"])
        for d in diags:
            w.writerow([d["fid"], d.get("n_dataset_seqs", ""),
                        d.get("n_ref_leaves", ""), d.get("n_matched", ""),
                        d.get("n_eval_seqs", ""), d.get("n_aligned", ""),
                        d["stage"], d["reason"]])
    print(f"\n  Diagnostics saved: {diag_csv}")

    # Summary
    print(f"\n{'='*65}")
    print("  LITERATURE REFERENCE TREE RESULTS — BASELINES")
    print(f"{'='*65}")
    print(f'  {"Method":<15} {"n":>6} {"Avg normRF":>12} {"Median":>8} '
          f'{"Perfect%":>9} {"Worst%":>9}')
    print(f'  {"-"*61}')

    methods = ["Hamming", "SeqIdentity", "random"]
    for method in methods:
        m_results = [r for r in all_results if r["method"] == method]
        n = len(m_results)
        if n == 0:
            print(f'  {method:<15} {"-":>6} {"-":>12} {"-":>8}')
            continue
        avg = sum(r["normRF"] for r in m_results) / n
        sorted_nrf = sorted(r["normRF"] for r in m_results)
        med = sorted_nrf[n // 2]
        perf_pct = sum(1 for r in m_results if r["normRF"] == 0) / n * 100
        worst_pct = sum(1 for r in m_results if r["normRF"] >= 0.98) / n * 100
        print(f'  {method:<15} {n:>6} {avg:>12.4f} {med:>8.4f} '
              f'{perf_pct:>8.1f}% {worst_pct:>8.1f}%')

    out_csv = os.path.join(args.output_dir, "literature_baselines.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "n_seqs", "method", "normRF", "rf", "max_rf"])
        for r in all_results:
            w.writerow([r["family"], r["n_seqs"], r["method"],
                       f"{r['normRF']:.4f}", r["rf"], r["max_rf"]])
    print(f'\n  Saved: {out_csv}')


if __name__ == "__main__":
    main()