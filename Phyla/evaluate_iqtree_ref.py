#!/usr/bin/env python3
"""
Evaluate PHYLA, ESM2, Hamming, SeqIdentity against IQ-TREE reference trees.
Compares with original FastTree-based results.

Usage (sbatch only):
  python evaluate_iqtree_ref.py \
    --ref-pickle virus_data/vogdb_treefam_v2.pickle \
    --pred-pickle virus_data/phyla_predictions.pickle \
    --iqtree-dir virus_data/iqtree_trees \
    --family-list virus_data/iqtree_family_list.txt \
    --msa-dir virus_data/msa \
    --output-dir eval_preds
"""
import os, sys, re, csv, pickle, argparse, random
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'phyla'))

from ete3 import Tree
from Bio import Phylo
from io import StringIO


# ── Tree utilities ──

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


# ── Distance functions + NJ ──

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
    from skbio import DistanceMatrix
    from skbio.tree import nj
    n = len(names)
    dm = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = dfunc(seqs[i], seqs[j])
            dm[i][j] = dm[j][i] = d
    return nj(DistanceMatrix(dm, names)).__str__().replace(' ', '')

def rand_tree(names):
    s = names[:]; random.shuffle(s)
    return '(' + ','.join(s) + ');'

def load_msa(p):
    seqs, name = {}, None
    with open(p) as f:
        for line in f:
            l = line.strip()
            if not l: continue
            if l.startswith('>'): name, seqs[l[1:].split()[0]] = l[1:].split()[0], ''
            elif name: seqs[name] += l
    return seqs


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-pickle', default='virus_data/vogdb_treefam_v2.pickle')
    ap.add_argument('--pred-pickle', default='virus_data/phyla_predictions.pickle')
    ap.add_argument('--iqtree-dir', default='virus_data/iqtree_trees')
    ap.add_argument('--family-list', default='virus_data/iqtree_family_list.txt')
    ap.add_argument('--msa-dir', default='virus_data/msa')
    ap.add_argument('--output-dir', default='eval_preds')
    args = ap.parse_args()

    ref_path = os.path.join(SCRIPT_DIR, args.ref_pickle)
    pred_path = os.path.join(SCRIPT_DIR, args.pred_pickle)
    iq_dir = os.path.join(SCRIPT_DIR, args.iqtree_dir)
    msa_dir = os.path.join(SCRIPT_DIR, args.msa_dir)
    out_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load family list
    with open(os.path.join(SCRIPT_DIR, args.family_list)) as f:
        fam_list = [l.strip() for l in f if l.strip()]

    print(f'Target families: {len(fam_list)}')

    # Load PHYLA predictions
    print('Loading PHYLA predictions...')
    pred = pickle.load(open(pred_path, 'rb'))

    # Load (optional) ESM2 results — read from CSV if available
    esm2_csv = os.path.join(out_dir, 'virus_esm2_vs_fasttree.csv')
    esm2_results = {}
    if os.path.exists(esm2_csv):
        with open(esm2_csv) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 5:
                    vid = row[0]
                    try: esm2_results[vid] = float(row[4])
                    except: pass
        print(f'  Loaded {len(esm2_results)} ESM2 results')

    methods = ['phyla', 'hamming', 'seqidentity', 'random']
    iq_results = {m: [] for m in methods}
    ft_results = {m: [] for m in methods}  # FastTree comparison for reference

    evaluated = 0
    skipped_no_iq = 0
    skipped_small = 0

    for vid in fam_list:
        iq_nwk = os.path.join(iq_dir, f'{vid}.iqtree.nwk')
        if not os.path.exists(iq_nwk):
            skipped_no_iq += 1
            continue

        with open(iq_nwk) as f: iq_tree_str = f.read().strip()
        iq_leaves = set(leaves(iq_tree_str))

        # Get PHYLA prediction
        if vid not in pred: continue
        phyla_tree_str = pred[vid]['pred_tree_newick']
        phyla_leaves = set(pred[vid].get('seq_names', []))

        # Intersection
        common = sorted(iq_leaves & phyla_leaves)
        if len(common) < 4:
            skipped_small += 1
            continue

        # Prune IQ-TREE tree
        if iq_leaves != set(common):
            iq_p = prune(iq_tree_str, common)
            if iq_p is None: continue
        else:
            iq_p = iq_tree_str

        # Prune PHYLA tree
        if phyla_leaves != set(common):
            phy_p = prune(phyla_tree_str, common)
            if phy_p is None: continue
        else:
            phy_p = phyla_tree_str

        # --- PHYLA ---
        m = normrf(phy_p, iq_p)
        if m: iq_results['phyla'].append({'vid': vid, 'n': len(common), **m})

        # --- Hamming + SeqIdentity ---
        msa_path = os.path.join(msa_dir, f'{vid}.msa')
        if os.path.exists(msa_path):
            seqs = load_msa(msa_path)
            msa_names = sorted(set(seqs.keys()) & set(common))
            if len(msa_names) >= 4:
                sl = [seqs[n] for n in msa_names]
                try:
                    ht = nj_tree(sl, msa_names, ham_dist)
                    m = normrf(ht, iq_p)
                    if m: iq_results['hamming'].append({'vid': vid, 'n': len(msa_names), **m})

                    st = nj_tree(sl, msa_names, sid_dist)
                    m = normrf(st, iq_p)
                    if m: iq_results['seqidentity'].append({'vid': vid, 'n': len(msa_names), **m})
                except: pass

        # --- Random ---
        if len(common) >= 4:
            rt = rand_tree(common)
            m = normrf(rt, iq_p)
            if m: iq_results['random'].append({'vid': vid, 'n': len(common), **m})

        evaluated += 1
        if evaluated % 100 == 0:
            def avg(r): return sum(x['norm_rf'] for x in r)/len(r) if r else 0
            print(f'  [{evaluated}/{len(fam_list)}] '
                  f'P={avg(iq_results["phyla"]):.4f} '
                  f'H={avg(iq_results["hamming"]):.4f} '
                  f'R={avg(iq_results["random"]):.4f}')

    # ── Report ──
    print(f'\n{"="*70}')
    print(f'  IQ-TREE REFERENCE BENCHMARK')
    print(f'{"="*70}')
    print(f'  Aimed: {len(fam_list)} | Evaluated: {evaluated} | '
          f'No IQ-TREE: {skipped_no_iq} | <4 seqs: {skipped_small}')

    print(f'\n  {"Method":<15} {"n":>6} {"Avg normRF":>12} {"Median":>8} {"Perfect%":>10} {"Worst%":>10}')
    print(f'  {"-"*61}')
    for m in methods:
        r = iq_results[m]
        n = len(r)
        if n:
            vals = [x['norm_rf'] for x in r]
            avg = sum(vals)/n
            sv = sorted(vals)
            med = sv[n//2]
            perf = sum(1 for v in vals if v==0)/n*100
            worst = sum(1 for v in vals if v>=0.98)/n*100
            print(f'  {m:<15} {n:>6} {avg:>12.4f} {med:>8.4f} {perf:>9.1f}% {worst:>9.1f}%')

    # Compare with FastTree-based results
    print(f'\n  {"="*61}')
    print(f'  COMPARISON: IQ-TREE vs FastTree reference')
    print(f'  {"="*61}')
    print(f'  {"Method":<15} {"vs FastTree":>12} {"vs IQ-TREE":>12} {"Delta":>10}')
    print(f'  {"-"*49}')
    for m in methods:
        r_iq = iq_results[m]
        if not r_iq: continue
        iq_avg = sum(x['norm_rf'] for x in r_iq)/len(r_iq)
        # FastTree reference comes from the original eval
        ft_avg = None
        if m in ('phyla', 'hamming', 'seqidentity', 'random'):
            # Load original FastTree-based CSV
            ft_csv = os.path.join(out_dir, f'virus_{m}_vs_fasttree.csv')
            if os.path.exists(ft_csv):
                with open(ft_csv) as f:
                    reader = csv.reader(f)
                    next(reader)
                    ft_vals = []
                    for row in reader:
                        if len(row) >= 5:
                            try: ft_vals.append(float(row[4]))
                            except: pass
                    if ft_vals:
                        ft_avg = sum(ft_vals)/len(ft_vals)
        if ft_avg is not None:
            delta = iq_avg - ft_avg
            print(f'  {m:<15} {ft_avg:>12.4f} {iq_avg:>12.4f} {delta:>+10.4f}')
        else:
            print(f'  {m:<15} {"N/A":>12} {iq_avg:>12.4f} {"":>10}')

    # Save CSV
    for m in methods:
        if iq_results[m]:
            csv_path = os.path.join(out_dir, f'iqtree_{m}_results.csv')
            with open(csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerows([['vfam', 'method', 'rf', 'max_rf', 'norm_rf', 'n_seqs']])
                for r in iq_results[m]:
                    w.writerow([r['vid'], m, r['rf'], r['max_rf'],
                               f"{r['norm_rf']:.4f}", r['n']])
            print(f'  Saved: {csv_path}')

    print(f'\n{"="*70}')
    print(f'  INTERPRETATION')
    print(f'  {"="*70}')
    print(f'  If normRF values are LOWER against IQ-TREE than against FastTree')
    print(f'  (negative Delta), the methods agree MORE with the better reference.')
    print(f'  A smaller Delta for one method vs another suggests it was less')
    print(f'  affected by the FastTree → IQ-TREE reference change.')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
