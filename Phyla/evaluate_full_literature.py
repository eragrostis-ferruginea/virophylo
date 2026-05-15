#!/usr/bin/env python3
"""
Full evaluation: all 325 Brown & Firth trees + GVDB + Orthototiviridae + RdRp-scan
Runs Hamming/SeqIdentity/Random against all available trees.
"""
import os, sys, re, pickle, csv, argparse, random, glob
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

def get_leaves(s):
    return sorted(str(x.name) for x in Phylo.read(StringIO(s), 'newick').get_terminals())

def prune_tree(s, keep):
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

def load_sequences(faa_path):
    seqs = {}
    for rec in SeqIO.parse(faa_path, 'fasta'):
        seqs[rec.id] = str(rec.seq)
    return seqs

def load_local_sequences(tree_dir, global_seqs):
    """Load aligned sequences from the same directory as the tree file.
    Also falls back to global_seqs."""
    local_seqs = {}
    if os.path.exists(tree_dir):
        for f in os.listdir(tree_dir):
            if f.endswith('.fasta') or f.endswith('.faa'):
                fpath = os.path.join(tree_dir, f)
                for rec in SeqIO.parse(fpath, 'fasta'):
                    local_seqs[rec.id] = str(rec.seq)

    # Check parent directory too
    parent = os.path.dirname(tree_dir)
    if parent and parent != tree_dir:
        for f in os.listdir(parent):
            if f.endswith('.fasta') or f.endswith('.faa'):
                if 'aligned' in f.lower() or 'rdrp' in f.lower():
                    fpath = os.path.join(parent, f)
                    for rec in SeqIO.parse(fpath, 'fasta'):
                        local_seqs[rec.id] = str(rec.seq)

    # Merge: local takes priority (these are usually aligned correctly)
    merged = dict(global_seqs)
    merged.update(local_seqs)
    return merged, len(local_seqs) > 0

def match_leaf(name, seqs):
    """Try multiple strategies to match a leaf name to a sequence."""
    # Direct match
    if name in seqs: return name
    # Split on | (pipe-separated format)
    parts = name.split('|')
    if parts[0] in seqs: return parts[0]
    # Try without version
    base = parts[0].split('.')[0]
    if base in seqs: return base
    # Fallback: try any part
    for p in parts:
        if p in seqs: return p
        pbase = p.split('.')[0]
        if pbase in seqs: return pbase
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='/home/jianpinhe3/virophylo/virus_data/literature_refs')
    ap.add_argument('--output', default='eval_preds/literature_full_results.csv')
    args = ap.parse_args()

    data_dir = args.data_dir
    output_dir = os.path.join(SCRIPT_DIR, 'eval_preds')
    os.makedirs(output_dir, exist_ok=True)

    # Load all sequences
    all_seqs = {}
    for f in ['Brown_Firth_2025_RdRp/evaluation_data/all_sequences.faa',
              'Orthototiviridae_Ghabrivirales/evaluation_data/sequences.faa']:
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            s = load_sequences(p)
            all_seqs.update(s)
            print(f'{f}: {len(s)} sequences')

    # RdRp-scan FASTA files
    rdrp_dir = os.path.join(data_dir, 'RdRp-scan/evaluation_data')
    if os.path.exists(rdrp_dir):
        for f in glob.glob(os.path.join(rdrp_dir, '*.fasta')):
            s = load_sequences(f)
            all_seqs.update(s)
            print(f'RdRp-scan/{os.path.basename(f)}: {len(s)} sequences')

    print(f'Total sequences loaded: {len(all_seqs)}')

    # Find all tree files
    tree_files = []
    # Brown & Firth .tre files
    bf_dir = os.path.join(data_dir, 'Brown_Firth_2025_RdRp/supplementary_data')
    for root, dirs, files in os.walk(bf_dir):
        for f in files:
            if f.endswith('.tre'):
                tree_files.append(os.path.join(root, f))

    # RdRp-scan trees — skip FASTA_TREE (not Newick format)
    # GVDB .treefile
    gvdb_dir = os.path.join(data_dir, 'GVDB_giant_virus/evaluation_data')
    for f in glob.glob(os.path.join(gvdb_dir, '*.treefile')):
        tree_files.append(f)

    # Orthototiviridae
    ortho_nwk = os.path.join(data_dir, 'Orthototiviridae_Ghabrivirales/evaluation_data/reference.nwk')
    if os.path.exists(ortho_nwk):
        tree_files.append(ortho_nwk)

    print(f'Total tree files: {len(tree_files)}')

    # Evaluate
    results = []
    total_with_seqs = 0
    total_no_seqs = 0

    for tf in sorted(tree_files):
        try:
            with open(tf) as f: tree_str = f.read().strip()
        except:
            continue

        try:
            leaves = get_leaves(tree_str)
        except:
            continue
        if len(leaves) < 4: continue

        # Load per-directory aligned sequences (priority over global)
        tree_dir = os.path.dirname(tf)
        local_seqs, has_local = load_local_sequences(tree_dir, all_seqs)

        # Match leaves to sequences
        matched_seqs = {}
        for leaf in leaves:
            key = match_leaf(leaf, local_seqs)
            if key:
                matched_seqs[leaf] = local_seqs[key]

        n_matched = len(matched_seqs)
        if n_matched < 4:
            total_no_seqs += 1
            continue

        names = sorted(matched_seqs.keys())
        seq_list = [matched_seqs[n] for n in names]

        # Prune reference tree
        ref_pruned = prune_tree(tree_str, names)
        if ref_pruned is None: continue

        total_with_seqs += 1
        ds_name = os.path.relpath(tf, data_dir).replace('.tre','').replace('.treefile','').replace('/','_')[:80]

        # Random
        rt = rand_tree(names)
        m = normrf(rt, ref_pruned)
        if m: results.append({'ds': ds_name, 'n': n_matched, 'method': 'random', **m})

        # Hamming (sequences are already aligned)
        try:
            ht = nj_tree(seq_list, names, ham_dist)
            m = normrf(ht, ref_pruned)
            if m: results.append({'ds': ds_name, 'n': n_matched, 'method': 'hamming', **m})
        except: pass

        # SeqIdentity
        try:
            st = nj_tree(seq_list, names, sid_dist)
            m = normrf(st, ref_pruned)
            if m: results.append({'ds': ds_name, 'n': n_matched, 'method': 'seqidentity', **m})
        except: pass

        if total_with_seqs % 50 == 0:
            h_vals = [r['norm_rf'] for r in results if r['method']=='hamming']
            print(f'  [{total_with_seqs}/{len(tree_files)}] evaluated, hamming_avg={sum(h_vals)/len(h_vals):.4f}' if h_vals else f'  [{total_with_seqs}/{len(tree_files)}]')

    # Report
    print(f'\n{"="*60}')
    print(f'FULL LITERATURE REFERENCE EVALUATION')
    print(f'{"="*60}')
    print(f'Total trees: {len(tree_files)}')
    print(f'Trees with ≥4 matched seqs: {total_with_seqs}')
    print(f'Trees with insufficient seqs: {total_no_seqs}')

    for m in ['random', 'hamming', 'seqidentity']:
        m_r = [r for r in results if r['method'] == m]
        if not m_r: continue
        vals = [r['norm_rf'] for r in m_r]
        avg = sum(vals)/len(vals)
        sv = sorted(vals); med = sv[len(vals)//2]
        perf = sum(1 for v in vals if v==0)/len(vals)*100
        worst = sum(1 for v in vals if v>=0.98)/len(vals)*100
        print(f'  {m:<15}: n={len(vals):>4}  avg={avg:.4f}  med={med:.4f}  perfect={perf:.1f}%  worst={worst:.1f}%')

    # Save
    if results:
        csv_path = os.path.join(SCRIPT_DIR, args.output)
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows([['dataset', 'method', 'n_seqs', 'normRF', 'rf', 'max_rf']])
            for r in results:
                w.writerow([r['ds'], r['method'], r['n'],
                           f"{r['norm_rf']:.4f}", r['rf'], r['max_rf']])
        print(f'\nSaved: {csv_path} ({len(results)} rows)')

if __name__ == '__main__':
    main()
