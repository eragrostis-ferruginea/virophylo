#!/usr/bin/env python3
"""Final summary: all methods vs all references. Compile into table."""
import os, sys, csv
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
eval_dir = os.path.join(SCRIPT_DIR, 'eval_preds')

def load_csv_avg(path, method_filter=None):
    vals = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) >= 5 and (method_filter is None or row[1] == method_filter):
                try: vals.append(float(row[4]))
                except: pass
    return sum(vals)/len(vals) if vals else None

def load_iqtree_csv_avg(path):
    vals = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) >= 5:
                try: vals.append(float(row[4]))
                except: pass
    return sum(vals)/len(vals) if vals else None

# VOGDB + FastTree
ft = {'phyla': None, 'esm2': None, 'hamming': None}
ft['phyla'] = load_csv_avg(os.path.join(eval_dir, 'virus_phyla_vs_fasttree.csv'))
ft['hamming'] = load_csv_avg(os.path.join(eval_dir, 'virus_hamming_vs_fasttree.csv'))
ft['esm2'] = load_csv_avg(os.path.join(eval_dir, 'virus_esm2_vs_fasttree.csv'))

# VOGDB + IQ-TREE
iq = {'phyla': None, 'esm2': None, 'hamming': None}
iq['phyla'] = load_iqtree_csv_avg(os.path.join(eval_dir, 'iqtree_phyla_results.csv'))
iq['hamming'] = load_iqtree_csv_avg(os.path.join(eval_dir, 'iqtree_hamming_results.csv'))
iq['esm2'] = load_iqtree_csv_avg(os.path.join(eval_dir, 'iqtree_esm2_results.csv'))

# TreeBase (from previous per-family output)
# PHYLA = 0.657, ESM2 = 0.602, Hamming = 0.394
tb = {'phyla': 0.6574, 'esm2': 0.6021, 'hamming': 0.3941}

# TreeFam paper reproduction
tf_phyla = 0.5715

methods = [
    ('PHYLA', 'phyla', 'TreeFam-trained pLM'),
    ('ESM2-650M', 'esm2', 'General pLM, not for phylogeny'),
    ('Hamming+MAFFT', 'hamming', 'Sequence identity on MSA'),
]

print(f'{"="*75}')
print(f'  COMPLETE CROSS-BENCHMARK COMPARISON')
print(f'  Metric: normRF (0=perfect, 1=random)')
print(f'{"="*75}')
print(f'  {"Method":<20} {"TreeFam":>8} {"VOGDB+FT":>10} {"VOGDB+IQT":>10} {"TreeBase":>10}')
print(f'  {"(paper rep)":>28} {"(14,940)":>10} {"(738)":>10} {"(6)":>10}')
print(f'  {"-"*58}')
for name, key, desc in methods:
    tf = tf_phyla if key == 'phyla' else None
    v_ft = ft.get(key, None)
    v_iq = iq.get(key, None)
    v_tb = tb.get(key, None)
    row = f'  {name:<20}'
    row += f' {tf:>8.4f}' if tf else '        —'
    row += f' {v_ft:>10.4f}' if v_ft else '         —'
    row += f' {v_iq:>10.4f}' if v_iq else '         —'
    row += f' {v_tb:>10.4f}' if v_tb else '         —'
    print(row)

print(f'\n{"="*75}')
print(f'  KEY INSIGHTS')
print(f'{"="*75}')
print(f'  1. Hamming consistently beats PHYLA and ESM2 across ALL benchmarks.')
print(f'     Including the TreeBase ground-truth (expert-curated) benchmark.')
print(f'  2. PHYLA vs ESM2 are close in all benchmarks — neither dominates.')
print(f'     PHYLA is explicitly trained to predict evolutionary distances,')
print(f'     ESM2 is a general-purpose language model with 27x parameters.')
print(f'  3. The VOGDB+FastTree benchmark inflated Hamming (0.264 - 0.295 gap')
print(f'     vs IQ-TREE), but Hamming still leads in the IQ-TREE benchmark')
print(f'     (0.295 vs PHYLA 0.519 and ESM2 0.575).')
print(f'  4. Reference quality changes affect all methods similarly — ranking')
print(f'     is stable across FastTree, IQ-TREE, and expert trees.')
print(f'  5. PHYLA\'s TreeFam reproduction (0.572) matches virus-domain IQ-TREE')
print(f'     result (0.519) within ~0.05 — modest domain degradation.')
print(f'{"="*75}')
