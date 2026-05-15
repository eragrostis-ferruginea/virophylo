#!/usr/bin/env python3
"""ESM2-650M on IQ-TREE reference families. Same FAA→embedding→NJ→normRF pipeline."""
import os, sys, re, csv, argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR); sys.path.insert(0, os.path.join(SCRIPT_DIR, 'phyla'))

import torch
from Bio import SeqIO, Phylo
from io import StringIO
from transformers import AutoModel, AutoTokenizer
from skbio import DistanceMatrix
from skbio.tree import nj
from ete3 import Tree

def rm_bl(s):
    t = Phylo.read(StringIO(s), 'newick')
    for n in t.get_nonterminals(): n.branch_length = None
    for n in t.get_terminals(): n.branch_length = None
    o = StringIO(); Phylo.write(t, o, 'newick')
    return re.sub(r':[^,();\n]*', '', o.getvalue()).replace("'", '')

def leaves(s):
    return sorted(str(x.name) for x in Phylo.read(StringIO(s), 'newick').get_terminals())

def prune(s, keep):
    clean = rm_bl(s); t = Tree(clean)
    k = sorted([l for l in keep if l in set(t.get_leaf_names())])
    if len(k) < 4: return None
    t.prune(k); return t.write(format=5).replace(' ', '').replace("'", '')

def normrf(p, r):
    try:
        a = Tree(rm_bl(p)); b = Tree(rm_bl(r))
        x = a.compare(b, unrooted=True)
        if isinstance(x['norm_rf'], str): return None
        return {'rf': int(x['rf']), 'max_rf': int(x['max_rf']), 'norm_rf': x['norm_rf']}
    except: return None

@torch.no_grad()
def embed(model, tok, seqs, dev):
    inp = tok(seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    inp = {k: v.to(dev) for k, v in inp.items()}
    h = model(**inp).last_hidden_state
    m = inp['attention_mask'].unsqueeze(-1).float()
    return ((h * m).sum(1) / m.sum(1)).cpu()

def esm2_nj(model, tok, seqs, names, dev):
    emb = embed(model, tok, seqs, dev)
    n = len(names)
    dm = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = torch.cdist(emb[i:i+1], emb[j:j+1]).item()
            dm[i][j] = dm[j][i] = d
    return nj(DistanceMatrix(dm, names)).__str__().replace(' ', '')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--fam-list', default='virus_data/iqtree_family_list.txt')
    ap.add_argument('--faa-dir', default='virus_data/faa')
    ap.add_argument('--iqtree-dir', default='virus_data/iqtree_trees')
    ap.add_argument('--output', default='eval_preds/iqtree_esm2_results.csv')
    args = ap.parse_args()

    with open(os.path.join(SCRIPT_DIR, args.fam_list)) as f:
        families = [l.strip() for l in f if l.strip()]

    iqdir = os.path.join(SCRIPT_DIR, args.iqtree_dir)
    faadir = os.path.join(SCRIPT_DIR, args.faa_dir)

    print('Loading ESM2-650M...')
    tok = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
    model = AutoModel.from_pretrained('facebook/esm2_t33_650M_UR50D')
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(dev).eval()
    print(f'  Device: {dev}  Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M')

    results = []
    for vid in families:
        iq_nwk = os.path.join(iqdir, f'{vid}.iqtree.nwk')
        faa = os.path.join(faadir, f'{vid}.faa')
        if not os.path.exists(iq_nwk) or not os.path.exists(faa):
            continue

        with open(iq_nwk) as f: iq_str = f.read().strip()
        seqs = {r.id: str(r.seq) for r in SeqIO.parse(faa, 'fasta')}
        iq_leaves = set(leaves(iq_str))
        common = sorted(iq_leaves & set(seqs.keys()))
        if len(common) < 4: continue

        if iq_leaves != set(common):
            iq_str = prune(iq_str, common)
            if iq_str is None: continue

        sl = [seqs[n] for n in common]
        try:
            tree = esm2_nj(model, tok, sl, common, dev)
            m = normrf(tree, iq_str)
            if m:
                results.append({'vid': vid, 'n': len(common), **m})
        except: pass

        if len(results) % 100 == 0 and results:
            avg = sum(r['norm_rf'] for r in results)/len(results)
            print(f'  [{len(results)}] avg={avg:.4f}')

    # Report
    n = len(results)
    if n:
        avg = sum(r['norm_rf'] for r in results)/n
        sv = sorted(r['norm_rf'] for r in results)
        med = sv[n//2]
        perf = sum(1 for r in results if r['norm_rf']==0)/n*100
        worst = sum(1 for r in results if r['norm_rf']>=0.98)/n*100
        print(f'\nESM2 vs IQ-TREE: n={n} avg={avg:.4f} median={med:.4f} perfect={perf:.1f}% worst={worst:.1f}%')
        print(f'\n  {"Method":<15} {"n":>6} {"Avg normRF":>12} {"Median":>8} {"Perfect%":>10} {"Worst%":>10}')
        print(f'  {"-"*61}')
        print(f'  {"esm2":<15} {n:>6} {avg:>12.4f} {med:>8.4f} {perf:>9.1f}% {worst:>9.1f}%')

        with open(os.path.join(SCRIPT_DIR, args.output), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows([['vfam', 'method', 'rf', 'max_rf', 'norm_rf', 'n_seqs']])
            for r in results:
                w.writerow([r['vid'], 'esm2', r['rf'], r['max_rf'], f"{r['norm_rf']:.4f}", r['n']])
        print(f'Saved: {args.output}')

if __name__ == '__main__':
    main()
