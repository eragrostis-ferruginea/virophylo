#!/usr/bin/env python3
"""Properly parse all 325 Brown & Firth trees + sequence databases.
Uses Bio.Phylo for correct leaf name extraction (not regex).
"""
import os, sys, re, pickle, glob
from Bio import SeqIO, Phylo
from io import StringIO
from collections import defaultdict, Counter
from ete3 import Tree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'virus_data/literature_refs')
bf_dir = os.path.join(DATA_DIR, 'Brown_Firth_2025_RdRp/supplementary_data')

def get_leaf_names(tree_str):
    try:
        t = Phylo.read(StringIO(tree_str), 'newick')
        return [str(x.name) for x in t.get_terminals()]
    except:
        return []

def prune_tree(tree_str, keep):
    try:
        t = Tree(tree_str.replace("'", '').replace(' ', ''))
        keep_in = [l for l in keep if l in set(t.get_leaf_names())]
        if len(keep_in) < 4:
            return None
        t.prune(keep_in)
        return t.write(format=5).replace(' ', '').replace("'", '')
    except:
        return None

def normrf(p, r):
    from ete3 import Tree as T2
    try:
        a = T2(p); b = T2(r)
        x = a.compare(b, unrooted=True)
        if isinstance(x.get('norm_rf'), str): return None
        return {'rf': int(x['rf']), 'max_rf': int(x['max_rf']), 'norm_rf': x['norm_rf']}
    except:
        return None

# Step 1: Build comprehensive sequence index
print("Building sequence index...")
all_seqs = {}  # seq_name -> seq
all_name_aliases = defaultdict(set)  # alias -> set of canonical names

def index_fasta(fpath):
    count = 0
    for rec in SeqIO.parse(fpath, 'fasta'):
        sid = rec.id
        seq = str(rec.seq)
        all_seqs[sid] = seq
        # Also index by accession without version
        if '.' in sid:
            all_seqs[sid.split('.')[0]] = seq
        # Index by first part before space
        name_base = sid.split()[0]
        all_seqs[name_base] = seq
        count += 1
    return count

# Index all FASTA files
total = 0
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith('.fasta') or f.endswith('.faa'):
            total += index_fasta(os.path.join(root, f))
print(f"  Indexed {total} sequences from {len(all_seqs)} unique names")

# Step 2: Process each tree
print("\nProcessing trees...")
datasets = []
matched_leaves_total = 0
total_leaves = 0
skipped_no_seq = 0
skipped_small = 0

for root, dirs, files in os.walk(bf_dir):
    for f in sorted(files):
        if not f.endswith('.tre'):
            continue
        
        fpath = os.path.join(root, f)
        rel = os.path.relpath(fpath, bf_dir)
        ds_name = f'Brown2025_{rel.replace("/","_").replace(".tre","")}'
        
        with open(fpath) as fh:
            tree_str = fh.read().strip()
        
        all_leaves = get_leaf_names(tree_str)
        total_leaves += len(all_leaves)
        
        if len(all_leaves) < 4:
            continue
        
        # Match each leaf to sequences using multiple strategies
        matched_seqs = {}
        for leaf in all_leaves:
            if leaf in all_seqs:
                matched_seqs[leaf] = all_seqs[leaf]
                continue
            # Try splitting on '|'
            if '|' in leaf:
                for part in leaf.split('|'):
                    p = part.strip()
                    if p in all_seqs:
                        matched_seqs[leaf] = all_seqs[p]
                        break
            if leaf in matched_seqs:
                continue
            # Try accession without version
            if '.' in leaf:
                base = leaf.split('.')[0]
                if base in all_seqs:
                    matched_seqs[leaf] = all_seqs[base]
                    continue
            # Try first part before space/slash
            base = re.split(r'[\s/\|]', leaf)[0]
            if base in all_seqs:
                matched_seqs[leaf] = all_seqs[base]
        
        matched_leaves_total += len(matched_seqs)
        
        # Skip numeric-only leaves (internal node labels)
        valid_seqs = {k: v for k, v in matched_seqs.items() 
                     if not re.match(r'^[\d\.\-eE]+$', k) and len(v) >= 10}
        
        if len(valid_seqs) < 4:
            skipped_small += 1
            continue
        
        # Prune tree to matched leaves
        common = sorted(valid_seqs.keys())
        pruned_tree = prune_tree(tree_str, common)
        if pruned_tree is None:
            skipped_no_seq += 1
            continue
        
        datasets.append({
            'ds_name': ds_name,
            'tree_str': pruned_tree,
            'seqs': valid_seqs,
            'n_seqs': len(common),
            'source': 'Brown_Firth_2025'
        })

print(f"\n  Total leaves across all trees: {total_leaves}")
print(f"  Total matched leaf occurrences: {matched_leaves_total}")
print(f"  Datasets with >=4 seqs: {len(datasets)}")
print(f"  Skipped (<4 after match): {skipped_small}")
print(f"  Skipped (prune fail): {skipped_no_seq}")

# Summary
seq_counts = [d['n_seqs'] for d in datasets]
print(f"\n  Sequence stats:")
print(f"    Min: {min(seq_counts)}, Max: {max(seq_counts)}, Mean: {sum(seq_counts)/len(seq_counts):.0f}")

# Save
out_path = os.path.join(SCRIPT_DIR, 'eval_preds/literature_eval_dataset_v2.pickle')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
pickle.dump(datasets, open(out_path, 'wb'))
print(f"\n  Saved: {out_path} ({len(datasets)} datasets)")

# Quick test: run Hamming on all datasets
def ham_dist(s1, s2):
    if len(s1) != len(s2): return 1.0
    v = sum(1 for a,b in zip(s1,s2) if a!='-' and a!='.' and b!='-' and b!='.')
    if v == 0: return 0.5
    return 1.0 - sum(1 for a,b in zip(s1,s2) if a==b and a not in '-.X')/v

def nj_tree(seqs, names, dfunc):
    from skbio import DistanceMatrix; from skbio.tree import nj
    n = len(names); dm = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = dfunc(seqs[i], seqs[j])
            dm[i][j] = dm[j][i] = d
    return nj(DistanceMatrix(dm, names)).__str__().replace(' ', '')

import random
hamming_vals = []
for idx, ds in enumerate(datasets):
    names = sorted(ds['seqs'].keys())
    sl = [ds['seqs'][n] for n in names]
    try:
        ht = nj_tree(sl, names, ham_dist)
        m = normrf(ht, ds['tree_str'])
        if m: hamming_vals.append(m['norm_rf'])
    except:
        pass
    if (idx+1) % 50 == 0:
        print(f"  Progress: {idx+1}/{len(datasets)}, Hamming avg so far: {sum(hamming_vals)/len(hamming_vals):.4f}" % (idx+1, len(datasets)))

if hamming_vals:
    print(f"\n  QUICK HAMMING TEST (all datasets):")
    print(f"    n={len(hamming_vals)}, avg={sum(hamming_vals)/len(hamming_vals):.4f}")

# Random test
random.seed(42)
rand_vals = []
for ds in datasets:
    names = sorted(ds['seqs'].keys())
    s = names[:]; random.shuffle(s)
    rt = '(' + ','.join(s) + ');'
    m = normrf(rt, ds['tree_str'])
    if m: rand_vals.append(m['norm_rf'])
if rand_vals:
    print(f"    Random: n={len(rand_vals)}, avg={sum(rand_vals)/len(rand_vals):.4f}")
