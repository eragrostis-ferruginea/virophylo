#!/usr/bin/env python3
"""
Prepare literature reference tree dataset from MAFFT-aligned data.
Uses the proven matching strategy from evaluate_full_literature.py:
  - Loads trees from supplementary_data (.tre files)
  - Matches leaves to sequences from all_sequences.faa + local FASTA files
  - Uses MAFFT-aligned .aln sequences from evaluation_data where possible

Also processes GVDB, Orthototiviridae, RdRp-scan from evaluation_data.

Usage:
    python prepare_literature_dataset_mafft.py --output eval_preds/literature_refs_dataset.pickle

Or via SLURM:
    sbatch run_prepare_lit_mafft_slurm.sh
"""
import os, sys, re, pickle, argparse, csv, glob
from pathlib import Path
from collections import defaultdict
from Bio import SeqIO, Phylo
from io import StringIO

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "virus_data" / "literature_refs"


def load_fasta(fasta_path):
    seqs = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seqs[rec.id] = str(rec.seq)
    return seqs


def get_leaf_names(tree_str):
    try:
        t = Phylo.read(StringIO(tree_str), 'newick')
        return [str(x.name) for x in t.get_terminals()]
    except:
        return []


def prune_tree(tree_str, keep_names):
    from ete3 import Tree as EteTree
    try:
        tree_str = tree_str.replace("'", '').replace(' ', '')
        t = EteTree(tree_str)
        keep = sorted([n for n in keep_names if n in set(t.get_leaf_names())])
        if len(keep) < 4:
            return None, 0
        t.prune(keep)
        return t.write(format=5).replace(' ', '').replace("'", ''), len(keep)
    except:
        return None, 0


def match_leaf(name, seqs):
    """Try multiple strategies to match a leaf name to a sequence.
    Copied from evaluate_full_literature.py (proven to work for 179 datasets)."""
    if name in seqs:
        return name
    parts = name.split('|')
    if parts[0] in seqs:
        return parts[0]
    base = parts[0].split('.')[0]
    if base in seqs:
        return base
    for p in parts:
        if p in seqs:
            return p
        pbase = p.split('.')[0]
        if pbase in seqs:
            return pbase
    return None


def load_local_sequences(tree_dir, global_seqs):
    """Load sequences from the same directory as the tree file. Falls back to global_seqs."""
    local_seqs = {}
    if tree_dir.exists():
        for f in tree_dir.iterdir():
            if f.suffix in ('.fasta', '.faa'):
                for rec in SeqIO.parse(str(f), 'fasta'):
                    local_seqs[rec.id] = str(rec.seq)
    parent = tree_dir.parent
    if parent != tree_dir and parent.exists():
        for f in parent.iterdir():
            if f.suffix in ('.fasta', '.faa') and \
               ('aligned' in f.name.lower() or 'rdrp' in f.name.lower()):
                for rec in SeqIO.parse(str(f), 'fasta'):
                    local_seqs[rec.id] = str(rec.seq)
    merged = dict(global_seqs)
    merged.update(local_seqs)
    return merged, len(local_seqs) > 0


def process_brown_firth_supplementary():
    """Process Brown & Firth RdRp dataset using supplementary_data trees.
    Follows the proven approach from evaluate_full_literature.py."""
    print("\n" + "=" * 60)
    print("[1] Brown & Firth 2025 RdRp (supplementary_data trees)")
    print("=" * 60)

    eval_data_dir = DATA_DIR / "Brown_Firth_2025_RdRp" / "evaluation_data"
    supp_dir = DATA_DIR / "Brown_Firth_2025_RdRp" / "supplementary_data"

    if not supp_dir.exists():
        print(f"  [SKIP] Directory not found: {supp_dir}")
        return []

    # Load global sequences
    all_seqs = {}
    all_seqs_path = eval_data_dir / "all_sequences.faa"
    if all_seqs_path.exists():
        all_seqs = load_fasta(str(all_seqs_path))
        print(f"  all_sequences.faa: {len(all_seqs)} sequences")

    ortho_path = DATA_DIR / "Orthototiviridae_Ghabrivirales" / "evaluation_data" / "sequences.faa"
    if ortho_path.exists():
        ortho_seqs = load_fasta(str(ortho_path))
        all_seqs.update(ortho_seqs)
        print(f"  Orthototiviridae sequences: {len(ortho_seqs)} added")

    rdrp_dir = DATA_DIR / "RdRp-scan" / "evaluation_data"
    if rdrp_dir.exists():
        for fa_file in sorted(rdrp_dir.glob("*.fasta")):
            if fa_file.name.startswith("RdRp-scan_0.9"):
                continue
            rdrp_seqs = load_fasta(str(fa_file))
            all_seqs.update(rdrp_seqs)
    print(f"  Total global sequences: {len(all_seqs)}")

    # Load MAFFT-aligned .aln files from evaluation_data
    mafft_seqs = {}
    if eval_data_dir.exists():
        for aln_file in sorted(eval_data_dir.glob("*.aln")):
            mafft_seqs[aln_file.stem] = load_fasta(str(aln_file))
    print(f"  MAFFT-aligned files available: {len(mafft_seqs)}")

    # Find all .tre files in supplementary_data
    tree_files = []
    for root, dirs, files in os.walk(supp_dir):
        for f in files:
            if f.endswith('.tre') and not f.startswith('.'):
                tree_files.append(Path(root) / f)
    print(f"  Found {len(tree_files)} tree files (.tre)")

    datasets = []
    stats = {'matched_by_local': 0, 'matched_by_global': 0, 'failed': 0,
             'mafft_aligned': 0, 'no_aln_match': 0}

    for tf in sorted(tree_files):
        try:
            tree_str = tf.read_text().strip()
        except:
            stats['failed'] += 1
            continue

        leaves = get_leaf_names(tree_str)
        if len(leaves) < 4:
            stats['failed'] += 1
            continue

        tree_dir = tf.parent
        local_seqs, has_local = load_local_sequences(tree_dir, all_seqs)

        matched = {}
        for leaf in leaves:
            key = match_leaf(leaf, local_seqs)
            if key:
                matched[leaf] = local_seqs[key]

        if len(matched) < 4:
            stats['failed'] += 1
            continue

        matched = {k: v for k, v in matched.items() if len(v) >= 10}
        if len(matched) < 4:
            stats['failed'] += 1
            continue

        # Try to replace with MAFFT-aligned sequences from evaluation_data
        eval_stem = re.sub(r'[\\/]', '_', str(tf.relative_to(supp_dir)))
        eval_stem = eval_stem.replace('.tre', '.tre')

        mafft_used = False
        for stem_var in [eval_stem, tf.stem + '.tre']:
            if stem_var in mafft_seqs:
                aln_dict = mafft_seqs[stem_var]
                replaced = 0
                for leaf_name in list(matched.keys()):
                    key = match_leaf(leaf_name, aln_dict)
                    if key:
                        matched[leaf_name] = aln_dict[key]
                        replaced += 1
                if replaced > 0:
                    mafft_used = True
                    stats['mafft_aligned'] += 1
                break

        if not mafft_used:
            stats['no_aln_match'] += 1

        pruned_tree, n = prune_tree(tree_str, list(matched.keys()))
        if pruned_tree is None or n < 4:
            stats['failed'] += 1
            continue

        if has_local:
            stats['matched_by_local'] += 1
        else:
            stats['matched_by_global'] += 1

        rel_path = str(tf.relative_to(supp_dir))
        ds_name = f"Brown2025_{rel_path.replace('/', '_').replace('.tre', '')}"
        datasets.append({
            "ds_name": ds_name,
            "tree_str": pruned_tree,
            "seqs": {k: matched[k] for k in sorted(matched.keys())},
            "n_seqs": n,
            "source": "Brown_Firth_2025_RdRp"
        })

    print(f"  Matching stats:")
    print(f"    Matched by local sequences: {stats['matched_by_local']}")
    print(f"    Matched by global sequences: {stats['matched_by_global']}")
    print(f"    Using MAFFT-aligned .aln: {stats['mafft_aligned']}")
    print(f"    No MAFFT .aln match: {stats['no_aln_match']}")
    print(f"    Failed: {stats['failed']}")
    print(f"  Usable trees: {len(datasets)}")
    return datasets


def process_gvdb():
    print("\n" + "=" * 60)
    print("[2] GVDB Giant Virus (evaluation_data)")
    print("=" * 60)
    eval_dir = DATA_DIR / "GVDB_giant_virus" / "evaluation_data"
    if not eval_dir.exists():
        print(f"  [SKIP] Directory not found")
        return []

    datasets = []
    for tre_file in sorted(eval_dir.glob("*.treefile")):
        try:
            tree_str = tre_file.read_text().strip()
        except:
            continue
        aln_name = tre_file.name.rsplit('.treefile', 1)[0]
        aln_file = eval_dir / aln_name
        if not aln_file.exists():
            continue
        seqs = load_fasta(str(aln_file))
        if len(seqs) < 4:
            continue
        tree_leaves = get_leaf_names(tree_str)
        matched = {}
        for leaf in tree_leaves:
            key = match_leaf(leaf, seqs)
            if key:
                matched[leaf] = seqs[key]
        if len(matched) < 4:
            continue
        pruned_tree, n = prune_tree(tree_str, list(matched.keys()))
        if pruned_tree is None or n < 4:
            continue
        ds_name = f"GVDB_{tre_file.stem.replace('.aln', '')}"
        datasets.append({
            "ds_name": ds_name, "tree_str": pruned_tree,
            "seqs": {k: matched[k] for k in sorted(matched.keys())},
            "n_seqs": n, "source": "GVDB_giant_virus"
        })
        print(f"  {ds_name}: {n} sequences")
    print(f"  Usable trees: {len(datasets)}")
    return datasets


def process_orthototiviridae():
    print("\n" + "=" * 60)
    print("[3] Orthototiviridae (evaluation_data)")
    print("=" * 60)
    eval_dir = DATA_DIR / "Orthototiviridae_Ghabrivirales" / "evaluation_data"
    if not eval_dir.exists():
        print(f"  [SKIP] Directory not found")
        return []

    datasets = []
    tree_file = eval_dir / "reference.nwk"
    seq_file = eval_dir / "sequences.aln"
    if not seq_file.exists():
        seq_file = eval_dir / "sequences.faa"
    if not tree_file.exists() or not seq_file.exists():
        return []

    tree_str = tree_file.read_text().strip()
    seqs = load_fasta(str(seq_file))
    tree_leaves = get_leaf_names(tree_str)
    matched = {}
    for leaf in tree_leaves:
        key = match_leaf(leaf, seqs)
        if key:
            matched[leaf] = seqs[key]
    if len(matched) >= 4:
        pruned_tree, n = prune_tree(tree_str, list(matched.keys()))
        if pruned_tree and n >= 4:
            datasets.append({
                "ds_name": "Orthototiviridae_reference", "tree_str": pruned_tree,
                "seqs": {k: matched[k] for k in sorted(matched.keys())},
                "n_seqs": n, "source": "Orthototiviridae_Ghabrivirales"
            })
            print(f"  Orthototiviridae: {n} sequences")
    print(f"  Usable trees: {len(datasets)}")
    return datasets


def process_rdrpscan():
    print("\n" + "=" * 60)
    print("[4] RdRp-scan (evaluation_data)")
    print("=" * 60)
    eval_dir = DATA_DIR / "RdRp-scan" / "evaluation_data"
    if not eval_dir.exists():
        print(f"  [SKIP] Directory not found")
        return []

    datasets = []
    fast_tree_file = eval_dir / "RdRp-scan.CLUSTALO_0.4.FAST_TREE"
    if not fast_tree_file.exists():
        return []

    tree_str = fast_tree_file.read_text().strip()
    tree_leaves = get_leaf_names(tree_str)
    if len(tree_leaves) < 4:
        return []

    all_seqs = {}
    for fa_file in sorted(eval_dir.glob("*.fasta")):
        if fa_file.name.startswith("RdRp-scan_0.9"):
            continue
        all_seqs.update(load_fasta(str(fa_file)))
    if not all_seqs:
        return []

    matched = {}
    for leaf in tree_leaves:
        key = match_leaf(leaf, all_seqs)
        if key:
            matched[leaf] = all_seqs[key]
    if len(matched) >= 4:
        pruned_tree, n = prune_tree(tree_str, list(matched.keys()))
        if pruned_tree and n >= 4:
            datasets.append({
                "ds_name": "RdRp_scan_master", "tree_str": pruned_tree,
                "seqs": {k: matched[k] for k in sorted(matched.keys())},
                "n_seqs": n, "source": "RdRp_scan"
            })
            print(f"  RdRp-scan master: {n} sequences")
    print(f"  Usable trees: {len(datasets)}")
    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Prepare literature reference dataset from MAFFT-aligned data")
    parser.add_argument("--output", default="eval_preds/literature_refs_dataset.pickle",
                        help="Output pickle path")
    args = parser.parse_args()

    print("=" * 60)
    print("LITERATURE REFERENCE TREES — DATASET PREPARATION")
    print("Using supplementary_data trees + MAFFT-aligned sequences")
    print("=" * 60)

    all_datasets = []
    all_datasets.extend(process_brown_firth_supplementary())
    all_datasets.extend(process_gvdb())
    all_datasets.extend(process_orthototiviridae())
    all_datasets.extend(process_rdrpscan())

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total usable trees: {len(all_datasets)}")

    by_source = defaultdict(list)
    for ds in all_datasets:
        by_source[ds['source']].append(ds)

    for source, dss in sorted(by_source.items()):
        n_seqs = [ds['n_seqs'] for ds in dss]
        avg_seqs = sum(n_seqs) / len(n_seqs) if n_seqs else 0
        print(f"  {source}: {len(dss)} trees, "
              f"seqs={min(n_seqs)}-{max(n_seqs)}, avg={avg_seqs:.0f}")

    out_path = SCRIPT_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(all_datasets, open(out_path, 'wb'))
    print(f"\n  Saved: {out_path} ({len(all_datasets)} datasets)")

    all_seqs_total = sum(ds['n_seqs'] for ds in all_datasets)
    print(f"  Total sequences across all trees: {all_seqs_total}")

    csv_path = out_path.with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write("dataset,n_seqs,source\n")
        for ds in sorted(all_datasets, key=lambda x: x['ds_name']):
            f.write(f"{ds['ds_name']},{ds['n_seqs']},{ds['source']}\n")
    print(f"  CSV index: {csv_path}")


if __name__ == "__main__":
    main()
