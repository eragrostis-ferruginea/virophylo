#!/usr/bin/env python3
"""
Preprocess literature reference trees: extract tree-sequence pairs for evaluation.
Scans all .tre files, matches leaf names to FASTA sequences,
builds combined dataset for all-method evaluation.

Output: eval_preds/literature_eval_dataset.pickle
  - list of (dataset_name, tree_str, seq_dict)
"""
import os, sys, re, pickle, argparse
from collections import defaultdict
from Bio import SeqIO
from Bio import Phylo
from io import StringIO
from ete3 import Tree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'virus_data/literature_refs')

def get_leaf_names_newick(tree_str):
    try:
        t = Phylo.read(StringIO(tree_str), 'newick')
        return sorted(str(x.name) for x in t.get_terminals())
    except:
        return []

def prune_tree(tree_str, keep_leaves):
    clean = tree_str.replace("'", '').replace(' ', '')
    t = Tree(clean)
    keep = [l for l in keep_leaves if l in set(t.get_leaf_names())]
    if len(keep) < 4:
        return None
    t.prune(keep)
    return t.write(format=5).replace(' ', '').replace("'", '')

def load_brown_firth_refs():
    """Extract tree-sequence pairs from Brown & Firth 2025 data.
    Loads ALL available sequence sources for maximum coverage."""
    supp_dir = os.path.join(DATA_DIR, 'Brown_Firth_2025_RdRp/supplementary_data')
    
    # ── Load ALL sequence sources ──
    all_seqs = {}
    
    # 1. RdRp-scan master database (90k+ seqs)
    rdrp_scan = os.path.join(DATA_DIR, 'RdRp-scan/RdRp-scan_0.90.fasta')
    rdrp_scan2 = os.path.join(os.path.dirname(DATA_DIR), 'Phyla/virus_data/literature_refs/RdRp-scan/RdRp-scan_0.90.fasta')
    for rp in [rdrp_scan, rdrp_scan2]:
        if os.path.exists(rp):
            for rec in SeqIO.parse(rp, 'fasta'):
                all_seqs[rec.id] = str(rec.seq)
            sz = os.path.getsize(rp)
            print(f'  Loaded RdRp-scan DB: {sz//1024//1024}MB')
    
    # 2. Supplementary FASTA files (SF1, SF2, per-family)
    sf_count = 0
    for root, dirs, files in os.walk(supp_dir):
        for f in files:
            if f.endswith('.fasta'):
                fpath = os.path.join(root, f)
                for rec in SeqIO.parse(fpath, 'fasta'):
                    all_seqs[rec.id] = str(rec.seq)
                    sf_count += 1
    print(f'  Loaded supplementary FASTA: {sf_count} sequences')
    
    # 3. Downloaded NCBI GenBank sequences (5,198 RdRp seqs)
    gb_fasta = os.path.join(DATA_DIR, 'Brown_Firth_2025_RdRp/downloaded_seqs/genbank_rdrp.fasta')
    gb_count = 0
    if os.path.exists(gb_fasta):
        for rec in SeqIO.parse(gb_fasta, 'fasta'):
            all_seqs[rec.id] = str(rec.seq)
            gb_count += 1
    print(f'  Loaded GenBank downloads: {gb_count} sequences')
    
    print(f'  Total unique sequences loaded: {len(all_seqs)}')

    # Build accession index for fuzzy matching
    acc_index = defaultdict(list)
    for sid in all_seqs:
        base = sid.split('|')[0].split()[0].split('.')[0]
        acc_index[base].append(sid)

    # ── Scan all .tre files and match leaves ──
    import glob
    datasets = []
    matched_total = 0
    unmatched_total = 0

    for tf in sorted(glob.glob(f'{supp_dir}/**/*.tre', recursive=True)):
        rel_path = os.path.relpath(tf, supp_dir)
        with open(tf) as fh:
            tree_str = fh.read().strip()

        try:
            t = Phylo.read(StringIO(tree_str), 'newick')
            leaves = sorted(str(x.name) for x in t.get_terminals())
        except:
            continue

        if len(leaves) < 4:
            continue

        # Match leaves to sequences using multiple strategies
        seqs = {}
        for leaf in leaves:
            seq = None
            
            # Strategy 1: exact match
            if leaf in all_seqs:
                seq = all_seqs[leaf]
            
            # Strategy 2: split by '|' (common format: ACC.VER|name|ictv)
            if seq is None and '|' in leaf:
                core = leaf.split('|')[0]
                if core in all_seqs:
                    seq = all_seqs[core]
                elif '.' in core:
                    base = core.split('.')[0]
                    if base in all_seqs:
                        seq = all_seqs[base]
            
            # Strategy 3: remove version number
            if seq is None and '.' in leaf:
                base = leaf.split('.')[0]
                if base in all_seqs:
                    seq = all_seqs[base]
            
            # Strategy 4: accession index lookup
            if seq is None:
                acc = leaf.split('|')[0].split()[0].split('.')[0]
                if acc in acc_index:
                    seq = all_seqs[acc_index[acc][0]]
            
            if seq:
                seqs[leaf] = seq

        matched_total += len(seqs)
        unmatched_total += (len(leaves) - len(seqs))

        if len(seqs) < 4:
            continue

        # Prune tree to matched leaves
        common = sorted(seqs.keys())
        clean = tree_str.replace("'", '').replace(' ', '')
        pt = Tree(clean)
        keep = [l for l in common if l in set(pt.get_leaf_names())]
        if len(keep) < 4:
            continue
        pt.prune(keep)
        pruned_str = pt.write(format=5).replace(' ', '').replace("'", '')

        ds_name = f'Brown2025_{rel_path.replace("/","_").replace(".tre","")}'
        datasets.append({
            'ds_name': ds_name,
            'tree_str': pruned_str,
            'seqs': {k: seqs[k] for k in keep},
            'n_seqs': len(keep),
            'source': 'Brown_Firth_2025'
        })

    total = matched_total + unmatched_total
    print(f'  Leaf matching: {matched_total}/{total} ({matched_total/total*100:.1f}%)')
    return datasets

def load_gvdb_refs():
    """Extract tree-sequence pairs from GVDB data."""
    ind_dir = os.path.join(DATA_DIR, 'GVDB_giant_virus/phylogenies_alignments/individual')
    datasets = []

    # GVDB has per-gene FAA files but concatenated trees
    # We'll use individual trees if available, otherwise skip
    concat_tree_dir = os.path.join(DATA_DIR, 'GVDB_giant_virus/phylogenies_alignments/concatenated')

    # For each concatenated tree, try to match with individual FAA
    for tf in os.listdir(concat_tree_dir):
        if tf.endswith('.treefile'):
            tf_path = os.path.join(concat_tree_dir, tf)
            with open(tf_path) as fh:
                tree_str = fh.read().strip()
            leaves = get_leaf_names_newick(tree_str)
            if len(leaves) < 4:
                continue
            datasets.append({
                'ds_name': f'GVDB_{tf.replace(".treefile","").replace(".","_")}',
                'tree_str': tree_str,
                'seqs': {},
                'n_seqs': len(leaves),
                'source': 'GVDB',
                'notes': 'Sequences need to be extracted from GVDB FAA files'
            })

    return datasets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='eval_preds/literature_eval_dataset.pickle')
    args = ap.parse_args()

    out_path = os.path.join(SCRIPT_DIR, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_datasets = []

    # Brown & Firth (325 trees)
    print('Processing Brown & Firth 2025...')
    bf = load_brown_firth_refs()
    print(f'  Found {len(bf)} usable datasets')
    all_datasets.extend(bf)

    # GVDB
    print('Processing GVDB...')
    gv = load_gvdb_refs()
    print(f'  Found {len(gv)} GVDB datasets')
    all_datasets.extend(gv)

    print(f'\nTotal: {len(all_datasets)} datasets')
    seq_counts = [d['n_seqs'] for d in all_datasets if d['n_seqs']]
    if seq_counts:
        print(f'  Min seqs: {min(seq_counts)}, Max: {max(seq_counts)}, Mean: {sum(seq_counts)/len(seq_counts):.0f}')

    # Short summary
    print(f'\n  {"Source":<20} {"Count":>6}')
    sources = defaultdict(int)
    for d in all_datasets:
        sources[d['source']] += 1
    for s, c in sorted(sources.items()):
        print(f'  {s:<20} {c:>6}')

    pickle.dump(all_datasets, open(out_path, 'wb'))
    print(f'\nSaved: {out_path}')
    print(f'Total datasets with sequences: {sum(1 for d in all_datasets if len(d["seqs"]) >= 4)}')
    print(f'Total datasets without sequences: {sum(1 for d in all_datasets if len(d["seqs"]) < 4)}')


if __name__ == '__main__':
    main()
