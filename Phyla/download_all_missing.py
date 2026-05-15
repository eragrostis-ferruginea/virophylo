#!/usr/bin/env python3
import os, re, glob, time
from Bio import SeqIO, Phylo, Entrez
from io import StringIO
Entrez.email = "phyla_eval@example.com"
ROOT = '/home/jianpinhe3/virophylo'
bf_dir = ROOT + '/virus_data/literature_refs/Brown_Firth_2025_RdRp/supplementary_data'
out_dir = ROOT + '/virus_data/literature_refs/Brown_Firth_2025_RdRp/downloaded_seqs'
os.makedirs(out_dir, exist_ok=True)
print("Building index...")
existing = set()
for src in [ROOT + '/virus_data/literature_refs/RdRp-scan/RdRp-scan_0.90.fasta',
            out_dir + '/genbank_rdrp.fasta']:
    if os.path.exists(src):
        for r in SeqIO.parse(src, 'fasta'):
            existing.add(r.id.split('|')[0].split()[0].split('.')[0])
for f in glob.glob(bf_dir + '/**/*.fasta', recursive=True):
    for r in SeqIO.parse(f, 'fasta'):
        existing.add(r.id.split('|')[0].split()[0].split('.')[0])
print("Existing:", len(existing))
needed = set()
for tf in glob.glob(bf_dir + '/**/*.tre', recursive=True):
    try:
        for leaf in Phylo.read(StringIO(open(tf).read()), 'newick').get_terminals():
            n = str(leaf.name).strip()
            if re.match(r'^[\d.+\-eE]+$', n): continue
            a = n.split('|')[0].split()[0].strip()
            if re.match(r'^[A-Za-z_]+\d+', a) and a.split('.')[0] not in existing:
                needed.add(a.split('.')[0])
    except: pass
print("Needed:", len(needed))
if not needed: print("None."); exit(0)
nl = sorted(needed); done = 0
for i in range(0, len(nl), 200):
    batch = nl[i:i+200]
    try:
        h = Entrez.efetch(db="protein", id=",".join(batch), rettype="fasta", retmode="text")
        with open(out_dir + '/genbank_rdrp.fasta', 'a') as f: f.write(h.read())
        h.close(); done += len(batch)
    except Exception as e: print("Fail at", i, e)
    time.sleep(0.34)
    if (i+200) % 5000 == 0: print(" ", i+200, "/", len(nl), done)
print("Done!", done, "sequences")
