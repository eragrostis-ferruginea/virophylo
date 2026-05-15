#!/usr/bin/env python3
"""
Prepare literature reference tree dataset for evaluation.
- Index ALL sequence sources (NCBI downloads + supplementary data)
- Match tree leaves to sequences
- Perform MSA on unaligned sequences
- Create unified pickle dataset

Usage:
    # Build dataset (CPU with MAFFT for MSA)
    sbatch run_prepare_literature_slurm.sh
    
    # Or run directly:
    python prepare_literature_dataset.py
"""
import os, sys, re, pickle, argparse, subprocess, shutil
from pathlib import Path
from collections import defaultdict
from Bio import SeqIO, Phylo
from io import StringIO

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "virus_data" / "literature_refs"
OUT_DIR = DATA_DIR / "aligned_data"

MAFFT = shutil.which("mafft") or "mafft"

# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def index_fasta(fasta_path):
    """Index FASTA file, return dict of seq_name -> sequence."""
    seqs = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        sid = rec.id
        seq = str(rec.seq)
        seqs[sid] = seq
        # Index without version (for NCBI accessions like YP_009328360.1)
        if '.' in sid:
            seqs[sid.split('.')[0]] = seq
        # Index by first token (for pipe-delimited names)
        if '|' in sid:
            for part in sid.split('|'):
                if part.strip():
                    seqs[part.strip()] = seq
    return seqs

def build_sequence_index(data_dir):
    """Build comprehensive sequence index from ALL sources."""
    print("Building sequence index from all sources...")
    all_seqs = {}
    
    sources = [
        # NCBI downloads (primary source for most trees)
        DATA_DIR / "Brown_Firth_2025_RdRp" / "downloaded_seqs" / "genbank_rdrp.fasta",
        DATA_DIR / "Brown_Firth_2025_RdRp" / "downloaded_seqs" / "genbank_rdrp_missing.fasta",
        # Supplementary data
        DATA_DIR / "Brown_Firth_2025_RdRp" / "supplementary_data" / "SF1_NC_sequences.fasta",
        DATA_DIR / "Brown_Firth_2025_RdRp" / "supplementary_data" / "SF2_NM_sequences.fasta",
    ]
    
    # Also index all .fasta files in supplementary_data subdirectories
    supp_dir = DATA_DIR / "Brown_Firth_2025_RdRp" / "supplementary_data"
    for fasta in supp_dir.rglob("*.fasta"):
        sources.append(fasta)
    
    for src in sources:
        if src.exists():
            n_before = len(all_seqs)
            seqs = index_fasta(str(src))
            all_seqs.update(seqs)
            n_added = len(all_seqs) - n_before
            print(f"  {src.name}: +{n_added} sequences")
    
    print(f"  Total unique sequences indexed: {len(all_seqs)}")
    return all_seqs

def get_leaf_names(tree_str):
    """Extract leaf names from Newick tree."""
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
    except:
        return None, 0

def run_mafft(fasta_in, fasta_out, mafft_opts="--auto --quiet"):
    """Run MAFFT alignment."""
    cmd = f"{MAFFT} {mafft_opts} {fasta_in} > {fasta_out} 2>/dev/null"
    result = subprocess.run(cmd, shell=True, capture_output=True, timeout=300)
    return result.returncode == 0

def is_aligned(seqs_dict):
    """Check if sequences are aligned (same length)."""
    if not seqs_dict:
        return True
    lens = set(len(v) for v in seqs_dict.values())
    return len(lens) <= 1

# ─────────────────────────────────────────────────────────────────────────────
# Process Brown & Firth 2025 RdRp dataset
# ─────────────────────────────────────────────────────────────────────────────

def process_brown_firth(all_seqs):
    """Process Brown & Firth 2025 RdRp dataset."""
    print("\n" + "=" * 60)
    print("[1] Brown & Firth 2025 RdRp")
    print("=" * 60)
    
    supp_dir = DATA_DIR / "Brown_Firth_2025_RdRp" / "supplementary_data"
    aligned_dir = OUT_DIR / "Brown_Firth_2025_RdRp"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = []
    stats = {"total": 0, "usable": 0, "aligned": 0, "msa_done": 0, "failed": 0}
    
    # Find all .tre files
    for tre_file in sorted(supp_dir.rglob("*.tre")):
        if tre_file.name.startswith('.'):
            continue
        
        stats["total"] += 1
        tree_str = tre_file.read_text().strip()
        tree_leaves = get_leaf_names(tree_str)
        
        if len(tree_leaves) < 4:
            stats["failed"] += 1
            continue
        
        # Match leaves to sequences
        matched = {}
        for leaf in tree_leaves:
            if leaf in all_seqs:
                matched[leaf] = all_seqs[leaf]
            elif '.' in leaf and leaf.split('.')[0] in all_seqs:
                matched[leaf] = all_seqs[leaf.split('.')[0]]
            elif '|' in leaf:
                for part in leaf.split('|'):
                    p = part.strip()
                    if p in all_seqs:
                        matched[leaf] = all_seqs[p]
                        break
        
        # Filter short/empty sequences
        matched = {k: v for k, v in matched.items() if len(v) >= 10}
        
        if len(matched) < 4:
            stats["failed"] += 1
            continue
        
        # Check alignment status
        if not is_aligned(matched):
            # Write temp FASTA and run MAFFT
            temp_in = aligned_dir / f"temp_{tre_file.stem}.faa"
            with open(temp_in, 'w') as f:
                for name, seq in sorted(matched.items()):
                    f.write(f">{name}\n{seq}\n")
            
            temp_out = aligned_dir / f"temp_{tre_file.stem}_aligned.faa"
            if run_mafft(str(temp_in), str(temp_out)):
                matched = index_fasta(str(temp_out))
                stats["msa_done"] += 1
            else:
                stats["aligned"] += 1
            
            # Clean up temp files
            if temp_in.exists():
                os.remove(temp_in)
        
        # Prune tree
        pruned_tree, n = prune_tree(tree_str, list(matched.keys()))
        if pruned_tree is None or n < 4:
            stats["failed"] += 1
            continue
        
        ds_name = f"Brown2025_{tre_file.stem}"
        datasets.append({
            "ds_name": ds_name,
            "tree_str": pruned_tree,
            "seqs": matched,
            "n_seqs": n,
            "source": "Brown_Firth_2025_RdRp"
        })
        stats["usable"] += 1
        
        if stats["usable"] <= 10 or stats["total"] % 50 == 0:
            print(f"  {ds_name[:50]:<50} n={n}")
    
    print(f"\n  Brown & Firth summary:")
    print(f"    Total trees: {stats['total']}")
    print(f"    Usable: {stats['usable']}")
    print(f"    Already aligned: {stats['aligned']}")
    print(f"    MSA performed: {stats['msa_done']}")
    print(f"    Failed: {stats['failed']}")
    
    return datasets

# ─────────────────────────────────────────────────────────────────────────────
# Process other datasets (GVDB, RdRp-scan, Orthototiviridae)
# ─────────────────────────────────────────────────────────────────────────────

def process_gvdb(all_seqs):
    """Process GVDB giant virus dataset."""
    print("\n" + "=" * 60)
    print("[2] GVDB Giant Virus")
    print("=" * 60)
    
    eval_dir = DATA_DIR / "GVDB_giant_virus" / "evaluation_data"
    datasets = []
    
    for aln_file in sorted(eval_dir.glob("*.aln")):
        tree_file = aln_file.with_suffix('.treefile')
        if not tree_file.exists():
            continue
        
        try:
            # GVDB files may have non-standard format, try FASTA-like parsing
            seqs = {}
            with open(aln_file) as f:
                current_name = None
                current_seq = []
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_name:
                            seqs[current_name] = ''.join(current_seq)
                        current_name = line[1:].split()[0]
                        current_seq = []
                    else:
                        current_seq.append(line)
                if current_name:
                    seqs[current_name] = ''.join(current_seq)
            
            if len(seqs) < 4:
                continue
            
            tree_str = tree_file.read_text().strip()
            pruned_tree, n = prune_tree(tree_str, list(seqs.keys()))
            
            if pruned_tree:
                datasets.append({
                    "ds_name": f"GVDB_{aln_file.stem}",
                    "tree_str": pruned_tree,
                    "seqs": seqs,
                    "n_seqs": n,
                    "source": "GVDB_giant_virus"
                })
                print(f"  GVDB_{aln_file.stem}: n={n}")
        except Exception as e:
            print(f"  Error: {aln_file.name}: {e}")
    
    print(f"  GVDB: {len(datasets)} usable trees")
    return datasets

def process_orthototiviridae(all_seqs):
    """Process Orthototiviridae dataset."""
    print("\n" + "=" * 60)
    print("[3] Orthototiviridae")
    print("=" * 60)
    
    eval_dir = DATA_DIR / "Orthototiviridae_Ghabrivirales" / "evaluation_data"
    datasets = []
    
    tree_file = eval_dir / "reference.nwk"
    seq_file = eval_dir / "sequences.faa"
    
    if tree_file.exists() and seq_file.exists():
        seqs = index_fasta(str(seq_file))
        tree_str = tree_file.read_text().strip()
        pruned_tree, n = prune_tree(tree_str, list(seqs.keys()))
        
        if pruned_tree and n >= 4:
            datasets.append({
                "ds_name": "Orthototiviridae_reference",
                "tree_str": pruned_tree,
                "seqs": seqs,
                "n_seqs": n,
                "source": "Orthototiviridae_Ghabrivirales"
            })
            print(f"  Orthototiviridae: n={n}")
    
    print(f"  Orthototiviridae: {len(datasets)} usable trees")
    return datasets

def process_rdrpscan(all_seqs):
    """Process RdRp-scan dataset."""
    print("\n" + "=" * 60)
    print("[4] RdRp-scan")
    print("=" * 60)
    
    eval_dir = DATA_DIR / "RdRp-scan" / "evaluation_data"
    datasets = []
    
    # Main tree
    fast_tree = eval_dir / "RdRp-scan.CLUSTALO_0.4.FAST_TREE"
    if fast_tree.exists():
        tree_str = fast_tree.read_text().strip()
        tree_leaves = get_leaf_names(tree_str)
        
        matched = {k: v for k, v in all_seqs.items() if k in tree_leaves}
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
                print(f"  RdRp-scan master: n={n}")
    
    print(f"  RdRp-scan: {len(datasets)} usable trees")
    return datasets

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare literature reference dataset")
    parser.add_argument("--output", default="eval_preds/literature_refs_dataset.pickle",
                        help="Output pickle path")
    args = parser.parse_args()
    
    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build sequence index
    all_seqs = build_sequence_index(DATA_DIR)
    
    # Process all datasets
    all_datasets = []
    all_datasets.extend(process_brown_firth(all_seqs))
    all_datasets.extend(process_gvdb(all_seqs))
    all_datasets.extend(process_rdrpscan(all_seqs))
    all_datasets.extend(process_orthototiviridae(all_seqs))
    
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
    all_seqs_total = sum(ds['n_seqs'] for ds in all_datasets)
    print(f"  Total sequences across all trees: {all_seqs_total}")

if __name__ == "__main__":
    main()
