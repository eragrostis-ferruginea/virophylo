#!/usr/bin/env python3
"""
Analyze all 331 Brown & Firth trees, count unique leaf names,
check match rates with existing FASTA, report gaps.
Run on login node (needs internet for NCBI).
"""
import os, re, sys, glob, pickle, time
from Bio import SeqIO, Phylo, Entrez
from io import StringIO
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'virus_data/literature_refs')
Entrez.email = "phyla_eval@example.com"

bf_dir = os.path.join(DATA_DIR, 'Brown_Firth_2025_RdRp/supplementary_data')

# Step 1: Build master sequence index from ALL FASTA files
print("Building sequence index from all FASTA files...")
all_seqs = {}
all_accs = set()
fasta_count = 0
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith('.fasta') or f.endswith('.faa'):
            fasta_count += 1
            fpath = os.path.join(root, f)
            for rec in SeqIO.parse(fpath, 'fasta'):
                # Multiple name variants
                seq_id = rec.id
                all_seqs[seq_id] = str(rec.seq)
                # accession without version
                acc = seq_id.split('.')[0] if '.' in seq_id else seq_id
                all_accs.add(acc)
                # for '|' separated names
                if '|' in seq_id:
                    for part in seq_id.split('|'):
                        all_accs.add(part)
                        if '_' in part:
                            all_accs.add(part.split('_')[0])

print(f"  Parsed {fasta_count} FASTA files")
print(f"  Total sequences indexed: {len(all_seqs)}")
print(f"  Total accession variants: {len(all_accs)}")

# Step 2: Collect all unique leaf names from ALL trees
print("\nCollecting all tree leaf names...")
all_leaves = set()  # unique leaf names across all trees
leaf_trees = defaultdict(list)  # leaf -> which trees it appears in
tree_count = 0

for root, dirs, files in os.walk(bf_dir):
    for f in files:
        if f.endswith('.tre'):
            tree_count += 1
            fpath = os.path.join(root, f)
            try:
                tree_str = open(fpath).read().strip()
                # Parse using regex for speed (avoid Bio.Phylo issues)
                leaf_matches = re.findall(r'([\w\.\-]+)(?=:)', tree_str)
                for name in leaf_matches:
                    all_leaves.add(name)
                    leaf_trees[name].append(f)
            except:
                continue

print(f"  Total trees: {tree_count}")
print(f"  Total unique leaf names: {len(all_leaves)}")

# Step 3: Check match rates
matched = set()
unmatched = set()
for leaf in all_leaves:
    if leaf in all_seqs:
        matched.add(leaf)
    elif leaf.split('.')[0] in all_accs:
        matched.add(leaf)
    elif '|' in leaf:
        parts = leaf.split('|')
        if any(p in all_seqs or p in all_accs for p in parts):
            matched.add(leaf)
        else:
            unmatched.add(leaf)
    else:
        unmatched.add(leaf)

print(f"\n  Match results:")
print(f"    Matched:   {len(matched)} ({len(matched)/len(all_leaves)*100:.1f}%)")
print(f"    Unmatched: {len(unmatched)} ({len(unmatched)/len(all_leaves)*100:.1f}%)")

# Step 4: Show unmatched sample
if unmatched:
    print(f"\n  Sample unmatched names:")
    for n in sorted(unmatched)[:20]:
        trees_for_leaf = leaf_trees.get(n, ['?'])
        print(f"    {n:<60} appears in {len(trees_for_leaf)} tree(s)")
    print(f"  ... and {len(unmatched)-20} more")

    # Step 5: Categorize unmatched by type
    genbank = set()
    ipg = set()
    other = set()
    gb_pattern = re.compile(r'^[A-Z][A-Z0-9]+\d+\.\d+$')
    single_pattern = re.compile(r'^[A-Z][A-Z0-9]+\d+$')
    for n in unmatched:
        if gb_pattern.match(n) or single_pattern.match(n):
            genbank.add(n)
        elif '|' in n:
            ipg.add(n)
        else:
            other.add(n)
    
    print(f"\n  Unmatched categories:")
    print(f"    GenBank accessions (downloadable): {len(genbank)}")
    print(f"    Pipe-delimited names:              {len(ipg)}")
    print(f"    Other:                             {len(other)}")
    print(f"    TOTAL downloadable:                {len(genbank) + len(ipg)}")

# Step 6: How many trees become usable if all downloaded?
print(f"\n  Tree coverage analysis:")
usable_now = 0
usable_after = 0
for root, dirs, files in os.walk(bf_dir):
    for f in files:
        if f.endswith('.tre'):
            fpath = os.path.join(root, f)
            try:
                tree_str = open(fpath).read().strip()
                leaves_in_tree = set(re.findall(r'([\w\.\-]+)(?=:)', tree_str))
                now_match = sum(1 for l in leaves_in_tree if l in matched)
                if now_match >= 4 and now_match >= len(leaves_in_tree) * 0.5:
                    usable_now += 1
                after_match = now_match + sum(1 for l in leaves_in_tree if l in unmatched)
                if after_match >= 4:
                    usable_after += 1
            except:
                continue

print(f"    Trees usable now (>=4 matched leaves):  {usable_now}")
print(f"    Trees usable after download:             {usable_after}")

# Save unmatched list
unmatched_path = os.path.join(SCRIPT_DIR, 'virus_data/missing_seqs.txt')
os.makedirs(os.path.join(SCRIPT_DIR, 'virus_data'), exist_ok=True)
with open(unmatched_path, 'w') as f:
    for n in sorted(unmatched):
        if re.match(r'^[A-Z][A-Z0-9]+\d+(\.\d+)?$', n):
            f.write(f"{n}\n")
print(f"\n  Saved downloadable missing list: {unmatched_path}")
gb_pattern = re.compile(r'^[A-Z][A-Z0-9]+\d+(\.\d+)?$')
downloadable = [n for n in unmatched if gb_pattern.match(n)]
print(f"  Lines: {len(downloadable)}")
