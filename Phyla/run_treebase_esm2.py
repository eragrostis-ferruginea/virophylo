#!/usr/bin/env python3
"""ESM2-650M inference on TreeBase virus protein families. Compares to expert ground truth."""
import sys, os, re, csv, argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR); sys.path.insert(0, os.path.join(SCRIPT_DIR, 'phyla'))

import torch
from Bio import SeqIO, Phylo
from io import StringIO
from transformers import AutoModel, AutoTokenizer
from skbio import DistanceMatrix
from skbio.tree import nj
from ete3 import Tree

# ── Helpers ──
def rm_bl(s):
    t = Phylo.read(StringIO(s), 'newick')
    for n in t.get_nonterminals(): n.branch_length = None
    for n in t.get_terminals():   n.branch_length = None
    o = StringIO(); Phylo.write(t, o, 'newick')
    return re.sub(r':[^,();\n]*', '', o.getvalue()).replace("'", '')

def leaves(s): return sorted(str(x.name) for x in Phylo.read(StringIO(s), 'newick').get_terminals())

def prune(s, keep):
    c = rm_bl(s); t = Tree(c)
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
def esm2_embed(model, tok, seqs, dev='cuda:0'):
    inp = tok(seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    inp = {k: v.to(dev) for k, v in inp.items()}
    h = model(**inp).last_hidden_state
    m = inp['attention_mask'].unsqueeze(-1).float()
    return ((h * m).sum(1) / m.sum(1)).cpu()

def build_nj_esm2(model, tok, seqs, names, dev='cuda:0'):
    emb = esm2_embed(model, tok, seqs, dev)
    n = len(names)
    dm = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = torch.cdist(emb[i:i+1], emb[j:j+1]).item()
            dm[i][j] = dm[j][i] = d
    return nj(DistanceMatrix(dm, names)).__str__().replace(' ', '')

# ── Main ──
TBFAMS = [
    ('TB2:S10171Taxa1', 'Phage terminase'),
    ('TB2:S10521', 'Poxvirus protein'),
    ('TB2:S12857Taxa1', 'Viral metagenomics'),
    ('TB2:S13909Taxa1', 'Fungal virus capsid'),
    ('TB2:S13955Taxa5', 'Plant virus polyprotein'),
    ('TB2:S1458', 'Plant potyvirus'),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    sd = os.path.join(SCRIPT_DIR, 'treebase_benchmark', 'sequences')
    td = os.path.join(SCRIPT_DIR, 'treebase_benchmark', 'trees')
    out = os.path.join(SCRIPT_DIR, 'eval_preds', 'treebase_esm2_gt.csv')

    print('Loading ESM2-650M...')
    tok = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
    model = AutoModel.from_pretrained('facebook/esm2_t33_650M_UR50D').to(args.device).eval()
    print(f'  Device: {args.device}  Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M')

    results = []
    for fid, desc in TBFAMS:
        fp = os.path.join(sd, f'{fid}_processed.fa')
        tp = os.path.join(td, f'{fid}_processed_tree.nh')
        if not os.path.exists(fp) or not os.path.exists(tp):
            print(f'  SKIP {fid}')
            continue

        with open(tp) as fh: ref = fh.read().strip()
        seqs = {r.id: str(r.seq) for r in SeqIO.parse(fp, 'fasta')}
        common = sorted(set(leaves(ref)) & set(seqs.keys()))
        if len(common) < 4: continue
        if set(leaves(ref)) != set(common):
            ref = prune(ref, common)
            if ref is None: continue

        sl = [seqs[n] for n in common]
        print(f'  {fid} ({len(common)} seqs)...')
        try:
            tree = build_nj_esm2(model, tok, sl, common, args.device)
            m = normrf(tree, ref)
            if m:
                print(f'    ESM2 normRF={m["norm_rf"]:.4f}')
                results.append({'family': fid, 'n_seqs': len(common), 'method': 'ESM2',
                                'normRF': m['norm_rf'], 'rf': m['rf'], 'max_rf': m['max_rf']})
            else: print('    normRF FAILED')
        except Exception as e: print(f'    ERROR: {e}')

    esm2r = [r for r in results if r['method'] == 'ESM2']
    n = len(esm2r)
    print(f'\n{"="*50}')
    print(f'  ESM2-650M vs Expert Trees')
    print(f'  Families: {n}')
    if n:
        avg = sum(r['normRF'] for r in esm2r) / n
        med = sorted(r['normRF'] for r in esm2r)[n//2]
        print(f'  Avg normRF: {avg:.4f}  Median: {med:.4f}')
        for r in results:
            print(f'    {r["family"]:<20} {r["n_seqs"]:>4} seqs  normRF={r["normRF"]:.4f}')
        # Comparison
        print(f'\n  PHYLA avg:   0.6574')
        print(f'  Hamming avg: 0.3941')
        print(f'  ESM2 avg:    {avg:.4f}')
        print(f'  Random avg:  1.0000')

    if results:
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows([['family', 'n_seqs', 'method', 'normRF', 'rf', 'max_rf']])
            for r in results:
                w.writerow([r['family'], r['n_seqs'], r['method'], f"{r['normRF']:.4f}", r['rf'], r['max_rf']])
        print(f'\n  Saved: {out}')

if __name__ == '__main__':
    main()
