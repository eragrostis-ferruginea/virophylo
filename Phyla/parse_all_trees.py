#!/usr/bin/env python3
"""Parse all 331 literature trees with ALL sequence sources."""
import os, glob, re, pickle, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'phyla'))

from Bio import SeqIO, Phylo
from ete3 import Tree

base = os.path.join(os.path.dirname(SCRIPT_DIR), 'virus_data/literature_refs/Brown_Firth_2025_RdRp/supplementary_data')
gb_dir = os.path.join(os.path.dirname(base), 'downloaded_seqs')

def prune_tree(tree_str, keep):
    clean = tree_str.replace("'", '').replace(' ', '')
    try:
        t = Tree(clean)
        k = [l for l in keep if l in set(t.get_leaf_names())]
        if len(k) < 4: return None
        t.prune(k)
        return t.write(format=5).replace(' ', '').replace("'", '')
    except: return None

print("Loading FASTA files...")
all_seqs = {}
for f in [os.path.join(gb_dir, 'genbank_rdrp.fasta'),
          os.path.join(gb_dir, 'genbank_rdrp_missing.fasta')]:
    if os.path.exists(f):
        n = 0
        for rec in SeqIO.parse(f, 'fasta'):
            all_seqs[rec.id] = str(rec.seq)
            a = rec.id.split('.')[0]
            if a not in all_seqs: all_seqs[a] = str(rec.seq)
            n += 1
        print(f'  Loaded {n} from {os.path.basename(f)}')

for f in glob.glob(os.path.join(base, 'SF*_sequences.fasta')):
    if not os.path.exists(f): continue
    n = 0
    for rec in SeqIO.parse(f, 'fasta'):
        all_seqs[rec.id] = str(rec.seq)
        all_seqs[rec.id.split('.')[0]] = str(rec.seq)
        n += 1
    print(f'  Loaded {n} from SF files')

for f in glob.glob(os.path.join(base, '**/*.fasta'), recursive=True) + \
        glob.glob(os.path.join(base, '**/*.faa'), recursive=True):
    n = 0
    for rec in SeqIO.parse(f, 'fasta'):
        r = rec.id.split()[0]
        if r not in all_seqs: all_seqs[r] = str(rec.seq)
        if '.' in r: all_seqs[r.split('.')[0]] = str(rec.seq)
        n += 1

print(f'  Total unique IDs: {len(all_seqs)}')

tree_files = glob.glob(os.path.join(base, '**/*.tre'), recursive=True)
print(f'Processing {len(tree_files)} trees...')

datasets = []
too_few = 0
for tf in sorted(tree_files):
    try:
        t = Phylo.read(tf, 'newick')
    except: continue
    leaves_raw = [str(x.name).strip().strip("'\"") for x in t.get_terminals() if x.name]
    if len(leaves_raw) < 4: continue

    seqs = {}
    for leaf in leaves_raw:
        acc = leaf.split('|')[0].strip()
        for c in [acc, acc.split('.')[0], leaf.strip(), leaf]:
            if c in all_seqs:
                seqs[leaf] = all_seqs[c]
                break
    if len(seqs) < 4: too_few += 1; continue

    pruned = prune_tree(open(tf).read(), list(seqs.keys()))
    if pruned is None: too_few += 1; continue

    datasets.append({
        'ds_name': f'Brown2025_{os.path.relpath(tf, base).replace("/","_").replace(".tre","")}',
        'tree_str': pruned, 'seqs': seqs,
        'n_seqs': len(seqs), 'source': 'Brown_Firth_2025'
    })

print(f'  Datasets: {len(datasets)}  Skipped: {too_few}')
if datasets:
    sc = [d['n_seqs'] for d in datasets]
    print(f'  Seqs: min={min(sc)} max={max(sc)} mean={sum(sc)/len(sc):.0f}')

out = os.path.join(SCRIPT_DIR, 'eval_preds/literature_eval_dataset_v3.pickle')
pickle.dump(datasets, open(out, 'wb'))
print(f'Saved: {out}')
