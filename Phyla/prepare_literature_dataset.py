#!/usr/bin/env python3
"""
Prepare literature reference tree dataset for evaluation.
- Performs MSA on unaligned sequences using MAFFT
- Validates tree-sequence pairs
- Creates unified pickle dataset

Usage:
    python prepare_literature_dataset.py [--dry-run] [--mafft-opts OPTS]
"""
import os, sys, re, pickle, argparse, subprocess, shutil
from pathlib import Path
from collections import defaultdict
from Bio import SeqIO, Phylo, AlignIO
from io import StringIO

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "virus_data" / "literature_refs"
OUT_DIR = DATA_DIR / "aligned_data"
MAFFT = shutil.which("mafft") or "/home/jianpinhe3/miniforge3/bin/mafft"

# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def is_aligned(fasta_path):
    """Check if FASTA file is already aligned (all sequences same length)."""
    try:
        lens = set()
        for rec in SeqIO.parse(fasta_path, "fasta"):
            lens.add(len(str(rec.seq).replace('-', '').replace('.', '')))
        return len(lens) <= 1
    except:
        return False

def get_leaf_names(tree_str):
    """Extract leaf names from Newick tree using Bio.Phylo."""
    try:
        t = Phylo.read(StringIO(tree_str), 'newick')
        return [str(x.name) for x in t.get_terminals()]
    except:
        return []

def prune_tree(tree_str, keep_names):
    """Prune tree to keep only specified leaf names."""
    from ete3 import Tree
    try:
        t = Tree(tree_str.replace("'", '').replace(' ', ''))
        keep = sorted([n for n in keep_names if n in set(t.get_leaf_names())])
        if len(keep) < 4:
            return None, 0
        t.prune(keep)
        return t.write(format=5).replace(' ', '').replace("'", ''), len(keep)
    except Exception as e:
        return None, 0

def run_mafft(fasta_in, fasta_out, mafft_opts="--auto --quiet"):
    """Run MAFFT alignment on input FASTA, save to output."""
    cmd = f"{MAFFT} {mafft_opts} {fasta_in} > {fasta_out}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"MAFFT failed: {result.stderr[:200]}")
    return fasta_out

def index_fasta(fasta_path):
    """Index FASTA file, return dict of seq_name -> sequence."""
    seqs = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        sid = rec.id
        seq = str(rec.seq)
        seqs[sid] = seq
        # Index without version
        if '.' in sid:
            seqs[sid.split('.')[0]] = seq
    return seqs

# ─────────────────────────────────────────────────────────────────────────────
# Process Brown & Firth 2025 RdRp dataset
# ─────────────────────────────────────────────────────────────────────────────

def process_brown_firth():
    """Process Brown & Firth 2025 RdRp dataset."""
    print("=" * 60)
    print("[1] Brown & Firth 2025 RdRp")
    print("=" * 60)
    
    eval_dir = DATA_DIR / "Brown_Firth_2025_RdRp" / "evaluation_data"
    aligned_dir = OUT_DIR / "Brown_Firth_2025_RdRp"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    
    # Build sequence index from all FASTA files
    print("  Building sequence index...")
    all_seqs = {}
    for faa in eval_dir.glob("*.faa"):
        if "all_" in faa.name:
            continue
        seqs = index_fasta(faa)
        all_seqs.update(seqs)
    print(f"    Indexed {len(all_seqs)} sequences")
    
    # Process each tree-seq pair
    nwk_files = sorted([f for f in eval_dir.glob("*.nwk")])
    datasets = []
    stats = {"total": 0, "aligned": 0, "msa_needed": 0, "usable": 0, "failed": 0}
    
    for nwk in nwk_files:
        faa = nwk.with_suffix('.faa')
        stats["total"] += 1
        
        if not faa.exists():
            print(f"  ⚠️  {nwk.name}: no matching .faa file")
            stats["failed"] += 1
            continue
        
        # Read tree
        tree_str = nwk.read_text().strip()
        tree_leaves = get_leaf_names(tree_str)
        
        # Read sequences and match to tree leaves
        seqs = index_fasta(faa)
        matched = {}
        for leaf in tree_leaves:
            if leaf in seqs:
                matched[leaf] = seqs[leaf]
            elif '.' in leaf and leaf.split('.')[0] in seqs:
                matched[leaf] = seqs[leaf.split('.')[0]]
        
        # Filter short/empty sequences
        matched = {k: v for k, v in matched.items() if len(v) >= 10}
        
        if len(matched) < 4:
            print(f"  ⚠️  {nwk.name}: only {len(matched)} valid sequences")
            stats["failed"] += 1
            continue
        
        # Check alignment status
        lens = set(len(v) for v in matched.values())
        if len(lens) <= 1:
            stats["aligned"] += 1
            ali_seqs = matched
        else:
            stats["msa_needed"] += 1
            # Write temp unaligned FASTA
            temp_in = OUT_DIR / f"temp_{nwk.stem}.faa"
            with open(temp_in, 'w') as f:
                for name, seq in sorted(matched.items()):
                    f.write(f">{name}\n{seq}\n")
            
            # Run MAFFT
            temp_out = OUT_DIR / f"temp_{nwk.stem}_aligned.faa"
            try:
                run_mafft(temp_in, temp_out)
                ali_seqs = index_fasta(temp_out)
                os.remove(temp_in)
                # os.remove(temp_out)  # Keep for inspection
            except Exception as e:
                print(f"  ⚠️  {nwk.name}: MAFFT failed ({e}), using unaligned")
                ali_seqs = matched
        
        # Prune tree to matched leaves
        pruned_tree, n_leaves = prune_tree(tree_str, list(ali_seqs.keys()))
        if pruned_tree is None or n_leaves < 4:
            print(f"  ⚠️  {nwk.name}: pruning failed")
            stats["failed"] += 1
            continue
        
        ds_name = f"Brown2025_{nwk.stem}"
        datasets.append({
            "ds_name": ds_name,
            "tree_str": pruned_tree,
            "seqs": ali_seqs,
            "n_seqs": n_leaves,
            "source": "Brown_Firth_2025_RdRp"
        })
        stats["usable"] += 1
    
    print(f"\n  Brown & Firth summary:")
    print(f"    Total trees: {stats['total']}")
    print(f"    Already aligned: {stats['aligned']}")
    print(f"    MSA needed: {stats['msa_needed']}")
    print(f"    Usable: {stats['usable']}")
    print(f"    Failed: {stats['failed']}")
    
    return datasets

# ─────────────────────────────────────────────────────────────────────────────
# Process GVDB giant virus dataset
# ─────────────────────────────────────────────────────────────────────────────

def process_gvdb():
    """Process GVDB giant virus dataset."""
    print("\n" + "=" * 60)
    print("[2] GVDB Giant Virus")
    print("=" * 60)
    
    eval_dir = DATA_DIR / "GVDB_giant_virus" / "evaluation_data"
    datasets = []
    
    aln_files = sorted([f for f in eval_dir.glob("*.aln")])
    for aln in aln_files:
        treefile = aln.with_suffix('.treefile')
        if not treefile.exists():
            continue
        
        try:
            # Try to parse alignment
            alignments = list(AlignIO.parse(aln, "clustal"))
            if not alignments:
                # Try reading as FASTA-like
                seqs = index_fasta(aln)
                if len(seqs) >= 4:
                    lens = set(len(v) for v in seqs.values())
                    if len(lens) > 1:
                        print(f"  ⚠️  {aln.name}: unaligned, needs special handling")
                        continue
                    tree_str = treefile.read_text().strip()
                    pruned_tree, n = prune_tree(tree_str, list(seqs.keys()))
                    if pruned_tree:
                        datasets.append({
                            "ds_name": f"GVDB_{aln.stem}",
                            "tree_str": pruned_tree,
                            "seqs": seqs,
                            "n_seqs": n,
                            "source": "GVDB_giant_virus"
                        })
                continue
            
            aln_obj = alignments[0]
            tree_str = treefile.read_text().strip()
            
            # Build seq dict from alignment
            seqs = {str(rec.id): str(rec.seq) for rec in aln_obj}
            tree_leaves = get_leaf_names(tree_str)
            
            # Match and prune
            matched = {k: v for k, v in seqs.items() if k in tree_leaves}
            if len(matched) < 4:
                continue
            
            pruned_tree, n = prune_tree(tree_str, list(matched.keys()))
            if pruned_tree:
                datasets.append({
                    "ds_name": f"GVDB_{aln.stem}",
                    "tree_str": pruned_tree,
                    "seqs": matched,
                    "n_seqs": n,
                    "source": "GVDB_giant_virus"
                })
        except Exception as e:
            print(f"  ⚠️  {aln.name}: error ({e})")
    
    print(f"  GVDB: {len(datasets)} usable trees")
    return datasets

# ─────────────────────────────────────────────────────────────────────────────
# Process RdRp-scan dataset
# ─────────────────────────────────────────────────────────────────────────────

def process_rdrpscan():
    """Process RdRp-scan dataset."""
    print("\n" + "=" * 60)
    print("[3] RdRp-scan")
    print("=" * 60)
    
    eval_dir = DATA_DIR / "RdRp-scan" / "evaluation_data"
    datasets = []
    
    # Main FAST_TREE file
    fast_tree = eval_dir / "RdRp-scan.CLUSTALO_0.4.FAST_TREE"
    if fast_tree.exists():
        try:
            tree_str = fast_tree.read_text().strip()
            tree_leaves = get_leaf_names(tree_str)
            print(f"  Master tree: {len(tree_leaves)} leaves")
            
            # Try to find sequences
            seqs = {}
            for fa in eval_dir.glob("*.fasta"):
                if "0.90" in fa.name:
                    seqs.update(index_fasta(fa))
            
            matched = {k: v for k, v in seqs.items() if k in tree_leaves}
            if len(matched) >= 4:
                pruned_tree, n = prune_tree(tree_str, list(matched.keys()))
                if pruned_tree:
                    datasets.append({
                        "ds_name": "RdRp_scan_master",
                        "tree_str": pruned_tree,
                        "seqs": matched,
                        "n_seqs": n,
                        "source": "RdRp_scan"
                    })
                    print(f"  RdRp-scan master: {n} matched leaves")
        except Exception as e:
            print(f"  ⚠️  RdRp-scan: error ({e})")
    
    print(f"  RdRp-scan: {len(datasets)} usable trees")
    return datasets

# ─────────────────────────────────────────────────────────────────────────────
# Process Orthototiviridae dataset
# ─────────────────────────────────────────────────────────────────────────────

def process_orthototiviridae():
    """Process Orthototiviridae dataset."""
    print("\n" + "=" * 60)
    print("[4] Orthototiviridae")
    print("=" * 60)
    
    eval_dir = DATA_DIR / "Orthototiviridae_Ghabrivirales" / "evaluation_data"
    datasets = []
    
    tree_file = eval_dir / "reference.nwk"
    seq_file = eval_dir / "sequences.faa"
    
    if tree_file.exists() and seq_file.exists():
        try:
            tree_str = tree_file.read_text().strip()
            tree_leaves = get_leaf_names(tree_str)
            
            seqs = index_fasta(seq_file)
            matched = {k: v for k, v in seqs.items() if k in tree_leaves}
            
            if len(matched) >= 4:
                # Check alignment
                lens = set(len(v) for v in matched.values())
                if len(lens) > 1:
                    # MSA needed
                    temp_in = OUT_DIR / "temp_ortho.faa"
                    with open(temp_in, 'w') as f:
                        for name, seq in sorted(matched.items()):
                            f.write(f">{name}\n{seq}\n")
                    
                    temp_out = OUT_DIR / "temp_ortho_aligned.faa"
                    try:
                        run_mafft(temp_in, temp_out)
                        matched = index_fasta(temp_out)
                        os.remove(temp_in)
                    except:
                        pass
                
                pruned_tree, n = prune_tree(tree_str, list(matched.keys()))
                if pruned_tree:
                    datasets.append({
                        "ds_name": "Orthototiviridae_reference",
                        "tree_str": pruned_tree,
                        "seqs": matched,
                        "n_seqs": n,
                        "source": "Orthototiviridae_Ghabrivirales"
                    })
                    print(f"  Orthototiviridae: {n} matched leaves")
        except Exception as e:
            print(f"  ⚠️  Orthototiviridae: error ({e})")
    
    print(f"  Orthototiviridae: {len(datasets)} usable trees")
    return datasets

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare literature reference dataset")
    parser.add_argument("--dry-run", action="store_true", help="Don't run MAFFT, just report")
    parser.add_argument("--mafft-opts", default="--auto --quiet", help="MAFFT options")
    parser.add_argument("--output", default="eval_preds/literature_refs_dataset.pickle",
                        help="Output pickle path")
    args = parser.parse_args()
    
    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process all datasets
    all_datasets = []
    all_datasets.extend(process_brown_firth())
    all_datasets.extend(process_gvdb())
    all_datasets.extend(process_rdrpscan())
    all_datasets.extend(process_orthototiviridae())
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total usable trees: {len(all_datasets)}")
    
    # Group by source
    by_source = defaultdict(list)
    for ds in all_datasets:
        by_source[ds['source']].append(ds)
    
    for source, dss in sorted(by_source.items()):
        n_seqs = [ds['n_seqs'] for ds in dss]
        print(f"  {source}: {len(dss)} trees, "
              f"seqs={min(n_seqs)}-{max(n_seqs)}, avg={sum(n_seqs)/len(n_seqs):.0f}")
    
    # Save pickle
    out_path = SCRIPT_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(all_datasets, open(out_path, 'wb'))
    print(f"\n  Saved: {out_path} ({len(all_datasets)} datasets)")
    
    # Quick stats
    all_seqs = sum(ds['n_seqs'] for ds in all_datasets)
    print(f"  Total sequences across all trees: {all_seqs}")

if __name__ == "__main__":
    main()
