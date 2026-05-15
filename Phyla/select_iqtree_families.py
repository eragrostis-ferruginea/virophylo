#!/usr/bin/env python3
"""
Select representative VOGDB families for IQ-TREE re-evaluation.
Stratifies by family size to ensure coverage across all size ranges,
then picks families that have both MSA and PHYLA prediction available.

Output: Saves selected family list as a pickle for SLURM array job.
"""
import os, sys, pickle, random
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

random.seed(42)

REF_PICKLE = os.path.join(SCRIPT_DIR, 'virus_data/vogdb_treefam_v2.pickle')
MSA_DIR = os.path.join(SCRIPT_DIR, 'virus_data/msa')
OUTPUT = os.path.join(SCRIPT_DIR, 'virus_data/iqtree_selected_families.pickle')
N_TOTAL = 1000

print(f'Loading ref data from {REF_PICKLE}...')
ref_data = pickle.load(open(REF_PICKLE, 'rb'))

# Build family size bins
BINS = [(4, 10), (11, 20), (21, 50), (51, 100), (101, 500), (501, 99999)]
BIN_NAMES = ['4-10', '11-20', '21-50', '51-100', '101-500', '501+']

families_by_bin = defaultdict(list)
for vid in ref_data:
    n = len(ref_data[vid]['sequences'])
    for (lo, hi), bname in zip(BINS, BIN_NAMES):
        if lo <= n <= hi and os.path.exists(os.path.join(MSA_DIR, f'{vid}.msa')):
            families_by_bin[bname].append(vid)
            break

print(f'\nAvailable families with MSA by size bin:')
for bname in BIN_NAMES:
    print(f'  {bname}: {len(families_by_bin[bname])}')

# Stratified sampling: proportional to bin size, min 20 per bin
total_available = sum(len(v) for v in families_by_bin.values())
selected = []

for bname in BIN_NAMES:
    n = len(families_by_bin[bname])
    target = max(20, int(N_TOTAL * n / total_available))
    target = min(target, n)  # can't exceed available
    sampled = random.sample(families_by_bin[bname], target)
    selected.extend(sampled)
    print(f'  Selecting {target}/{n} from {bname}')

# If we undershoot, fill from largest bin
shortfall = N_TOTAL - len(selected)
if shortfall > 0 and families_by_bin['501+']:
    extra = families_by_bin['501+'][:shortfall]
    # Remove any already selected
    extra = [v for v in extra if v not in selected][:shortfall]
    selected.extend(extra)

final_count = len(selected)
print(f'\nTotal selected: {final_count}')
print(f'Saving to {OUTPUT}')

# Save as flat list
pickle.dump(sorted(selected), open(OUTPUT, 'wb'))

# Also print the list for SLURM array
with open(os.path.join(SCRIPT_DIR, 'virus_data/iqtree_family_list.txt'), 'w') as f:
    for vid in sorted(selected):
        f.write(f'{vid}\n')

print(f'Family list saved to virus_data/iqtree_family_list.txt')
print(f'\nSize distribution of selected:')
for bname in BIN_NAMES:
    cnt = sum(1 for v in selected if bname == 
              [b for (lo,hi),b in zip(BINS, BIN_NAMES) 
               if lo <= len(ref_data[v]['sequences']) <= hi][0])
    print(f'  {bname}: {cnt}')
