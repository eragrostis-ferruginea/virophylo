#!/usr/bin/env python3
"""
Download missing GenBank protein sequences for Brown & Firth 2025 tree leaf names.
Run on login node (requires internet access).
"""
import os, re, sys, glob, time
from Bio import SeqIO, Phylo, Entrez
from io import StringIO
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
Entrez.email = "phyla_eval@example.com"

bf_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'virus_data/literature_refs/Brown_Firth_2025_RdRp/supplementary_data')
output_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'virus_data/literature_refs/Brown_Firth_2025_RdRp/downloaded_seqs')
os.makedirs(output_dir, exist_ok=True)

# Step 1: Build existing sequence index
print("Building existing sequence index...")
existing_seqs = {}
existing_accs = set()

# From RdRp-scan master DB
rdrp_scan = os.path.join(os.path.dirname(SCRIPT_DIR), 'virus_data/literature_refs/RdRp-scan/RdRp-scan_0.90.fasta')
if os.path.exists(rdrp_scan):
    for rec in SeqIO.parse(rdrp_scan, 'fasta'):
        existing_seqs[rec.id] = str(rec.seq)
        existing_accs.add(rec.id.split('|')[0])
        existing_accs.add(rec.id.split('.')[0])

# From all supplementary FASTAs
for f in sorted(glob.glob(f"{bf_dir}/**/*.fasta", recursive=True)):
    for rec in SeqIO.parse(f, 'fasta'):
        existing_seqs[rec.id] = str(rec.seq)
        existing_accs.add(rec.id.split('|')[0])
        existing_accs.add(rec.id.split()[0].split('.')[0])

print(f"  Existing sequences: {len(existing_seqs)}")

# Step 2: Collect all needed accessions from tree leaves
print("Collecting missing accessions from tree leaves...")
needed = set()
for tf in sorted(glob.glob(f"{bf_dir}/**/*.tre", recursive=True)):
    try:
        t = Phylo.read(StringIO(open(tf).read()), 'newick')
        for leaf in t.get_terminals():
            name = str(leaf.name).strip()
            if re.match(r'^[\d.+\-eE]+$', name): continue
            acc = name.split('|')[0].split()[0].strip()
            if re.match(r'^[A-Za-z]+\d+\.\d+$', acc) and acc not in existing_accs:
                needed.add(acc)
    except:
        pass

print(f"  Accessions to download: {len(needed)}")

# Step 3: Download in batches
print(f"Downloading from NCBI GenBank...")
downloaded = {}
batch = []
batch_size = 100
failures = []

for i, acc in enumerate(sorted(needed)):
    batch.append(acc)
    
    if len(batch) >= batch_size or i == len(needed) - 1:
        try:
            handle = Entrez.efetch(db="protein", id=",".join(batch), rettype="fasta", retmode="text")
            for rec in SeqIO.parse(handle, "fasta"):
                downloaded[rec.id.split('|')[0]] = str(rec.seq)
            handle.close()
        except Exception as e:
            failures.extend(batch)
        
        print(f"  [{i+1}/{len(needed)}] batch downloaded, total so far: {len(downloaded)}")
        batch = []
        time.sleep(1)  # NCBI rate limit

print(f"\nDownloaded: {len(downloaded)} sequences")
print(f"Failed: {len(failures)}")

# Step 4: Save
output_fa = os.path.join(output_dir, 'genbank_rdrp.fasta')
with open(output_fa, 'w') as f:
    for acc, seq in downloaded.items():
        f.write(f'>{acc}\n{seq}\n')
print(f"Saved: {output_fa}")

# Step 5: Final count
total = len(existing_seqs) + len(downloaded)
print(f"\nTotal available sequences: {total}")

# Save list of accessions per tree file for the evaluation script
tree_seq_map = defaultdict(set)
for tf in sorted(glob.glob(f"{bf_dir}/**/*.tre", recursive=True)):
    try:
        t = Phylo.read(StringIO(open(tf).read()), 'newick')
        for leaf in t.get_terminals():
            name = str(leaf.name).strip()
            if re.match(r'^[\d.+\-eE]+$', name): continue
            acc = name.split('|')[0].split()[0].strip()
            if acc in downloaded or acc in existing_accs:
                tree_seq_map[tf].add(name.split('|')[0])
    except:
        pass

with open(os.path.join(output_dir, 'tree_coverage.txt'), 'w') as f:
    for tf, accs in sorted(tree_seq_map.items()):
        f.write(f"{os.path.basename(tf)}: {len(accs)} matched\n")

print(f"Tree coverage report saved.")
PYEOF
