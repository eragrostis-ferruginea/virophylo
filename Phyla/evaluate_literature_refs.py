#!/usr/bin/env python3
"""Evaluate all literature reference datasets against PHYLA, ESM2, Hamming."""
import os, sys, re, pickle, csv, argparse, random, subprocess
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR); sys.path.insert(0, os.path.join(SCRIPT_DIR, 'phyla'))
from ete3 import Tree; from Bio import Phylo, SeqIO; from io import StringIO

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

def ham_dist(s1, s2):
    if len(s1) != len(s2): return 1.0
    v = sum(1 for a,b in zip(s1,s2) if a!='-' and a!='.' and b!='-' and b!='.')
    if v == 0: return 0.5
    return 1.0 - sum(1 for a,b in zip(s1,s2) if a==b and a not in '-.X')/v

def sid_dist(s1, s2):
    if len(s1) != len(s2): return 1.0
    idn = sum(1 for a,b in zip(s1,s2) if a==b and a not in '-.X')
    alg = sum(1 for a,b in zip(s1,s2) if a not in '-.X' and b not in '-.X')
    return 1.0 - idn/alg if alg else 1.0

def nj_tree(seqs, names, dfunc):
    from skbio import DistanceMatrix; from skbio.tree import nj
    n = len(names); dm = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = dfunc(seqs[i], seqs[j])
            dm[i][j] = dm[j][i] = d
    return nj(DistanceMatrix(dm, names)).__str__().replace(' ', '')

def rand_tree(names):
    s = names[:]; random.shuffle(s)
    return '(' + ','.join(s) + ');'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='eval_preds/literature_eval_dataset_v2.pickle')
    ap.add_argument('--output', default='eval_preds/literature_results_v2.csv')
    ap.add_argument('--mafft-tmp', default='virus_data/lit_mafft_tmp')
    args = ap.parse_args()

    import pickle
    datasets = pickle.load(open(os.path.join(SCRIPT_DIR, args.dataset), 'rb'))
    print(f'Loaded {len(datasets)} datasets')
    tmp_dir = os.path.join(SCRIPT_DIR, args.mafft_tmp)
    os.makedirs(tmp_dir, exist_ok=True)

    results = []
    for idx, ds in enumerate(datasets):
        ds_name = ds['ds_name']
        ref_tree = ds['tree_str']
        seqs = ds.get('seqs', {})
        n_seqs = len(seqs)

        if n_seqs < 4:
            print(f'  [{idx+1}/{len(datasets)}] {ds_name[:60]:<60} SKIP (<4 seqs)')
            continue

        names = sorted(seqs.keys())
        seq_list = [seqs[n] for n in names]

        # Prune reference tree
        ref_pruned = prune(ref_tree, names)
        if ref_pruned is None:
            print(f'  [{idx+1}/{len(datasets)}] {ds_name[:60]:<60} SKIP (prune fail)')
            continue

        print(f'  [{idx+1}/{len(datasets)}] {ds_name[:60]:<60} {n_seqs:>4} seqs', end='')

        # --- RANDOM ---
        rt = rand_tree(names)
        m = normrf(rt, ref_pruned)
        if m: results.append({'ds': ds_name, 'n': n_seqs, 'method': 'random', **m})

        # Sequences in Brown & Firth are already aligned — use directly
        ali_seqs = seqs
        ali_n = names
        ali_sl = seq_list

        # Prune ref to aligned names
        ref_ali = prune(ref_pruned, ali_n)
        if ref_ali is None:
            print(' PRUNE_FAIL', end='')
            print()
            continue

        # --- HAMMING ---
        try:
            ht = nj_tree(ali_sl, ali_n, ham_dist)
            m = normrf(ht, ref_ali)
            if m: results.append({'ds': ds_name, 'n': len(ali_n), 'method': 'hamming', **m})
        except:
            print(' HAM_FAIL', end='')

        # --- SEQIDENTITY ---
        try:
            st = nj_tree(ali_sl, ali_n, sid_dist)
            m = normrf(st, ref_ali)
            if m: results.append({'ds': ds_name, 'n': len(ali_n), 'method': 'seqidentity', **m})
        except:
            print(' SID_FAIL', end='')

        print()

    # Summary
    print(f'\n{"="*60}')
    print(f'LITERATURE REFERENCE EVALUATION RESULTS')
    print(f'{"="*60}')
    methods = ['random', 'hamming', 'seqidentity']
    for m in methods:
        m_r = [r for r in results if r['method'] == m]
        if not m_r: continue
        vals = [r['norm_rf'] for r in m_r]
        avg = sum(vals)/len(vals)
        sv = sorted(vals); med = sv[len(vals)//2]
        perf = sum(1 for v in vals if v==0)/len(vals)*100
        worst = sum(1 for v in vals if v>=0.98)/len(vals)*100
        print(f'  {m:<15}: n={len(vals):>3}  avg={avg:.4f}  med={med:.4f}  perfect={perf:.1f}%  worst={worst:.1f}%')

    # Save CSV
    if results:
        with open(os.path.join(SCRIPT_DIR, args.output), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows([['dataset', 'method', 'n_seqs', 'normRF', 'rf', 'max_rf']])
            for r in results:
                w.writerow([r['ds'], r['method'], r['n'],
                           f"{r['norm_rf']:.4f}", r['rf'], r['max_rf']])
        print(f'\nSaved: {args.output} ({len(results)} rows)')

if __name__ == '__main__':
    main()
