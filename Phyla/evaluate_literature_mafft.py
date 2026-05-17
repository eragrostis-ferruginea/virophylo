#!/usr/bin/env python3
"""
Full evaluation: 318 MAFFT-aligned literature reference trees.
Uses the matching strategy from evaluate_full_literature.py (179 datasets)
but replaces sequences with MAFFT-aligned .aln versions from evaluation_data.

Outputs: CSV with normRF for Hamming, SeqIdentity, Random baselines.
PHYLA and ESM2 require separate GPU runs.

Usage (SLURM):
    sbatch run_literature_mafft_slurm.sh
"""
import os, sys, re, csv, argparse, random, glob
from pathlib import Path
from collections import defaultdict
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from ete3 import Tree
from Bio import Phylo, SeqIO
from io import StringIO


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

def load_fasta(fpath):
    return {rec.id: str(rec.seq) for rec in SeqIO.parse(fpath, 'fasta')}

def match_leaf(name, seqs):
    """Try multiple strategies to match a leaf name to a sequence."""
    if name in seqs: return name
    parts = name.split('|')
    if parts[0] in seqs: return parts[0]
    base = parts[0].split('.')[0]
    if base in seqs: return base
    for p in parts:
        if p in seqs: return p
        pbase = p.split('.')[0]
        if pbase in seqs: return pbase
    return None

def load_local_sequences(tree_dir, global_seqs):
    """Load sequences from tree directory + parent. Falls back to global_seqs."""
    local_seqs = {}
    if os.path.exists(tree_dir):
        for f in os.listdir(tree_dir):
            if f.endswith('.fasta') or f.endswith('.faa'):
                for rec in SeqIO.parse(os.path.join(tree_dir, f), 'fasta'):
                    local_seqs[rec.id] = str(rec.seq)
    parent = os.path.dirname(tree_dir)
    if parent and parent != tree_dir:
        for f in os.listdir(parent):
            if f.endswith('.fasta') or f.endswith('.faa'):
                if 'aligned' in f.lower() or 'rdrp' in f.lower():
                    for rec in SeqIO.parse(os.path.join(parent, f), 'fasta'):
                        local_seqs[rec.id] = str(rec.seq)
    merged = dict(global_seqs)
    merged.update(local_seqs)
    return merged, len(local_seqs) > 0


def find_mafft_aln(aln_dir, tree_rel_path):
    """Find the MAFFT-aligned .aln file for a given supplementary_data tree.
    
    Tree from: supplementary_data/OTUs/newick/Amarillovirales_1.tre
    .aln file: evaluation_data/OTUs_newick_Amarillovirales_1.tre.aln
    """
    stem = tree_rel_path.replace('/', '_').replace('.tre', '.tre')
    for ext in ['.aln', '.faa']:
        fpath = os.path.join(aln_dir, stem + ext)
        if os.path.exists(fpath):
            return fpath
    # Also try with just the filename
    fname = os.path.basename(tree_rel_path)
    for ext in ['.aln', '.faa']:
        base = fname.replace('.tre', '.tre')
        for candidate in [base, base + ext]:
            fpath = os.path.join(aln_dir, candidate)
            if os.path.exists(fpath):
                return fpath
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', 
        default='/home/jianpinhe3/virophylo/virus_data/literature_refs')
    ap.add_argument('--output', 
        default='eval_preds/literature_mafft_results.csv')
    args = ap.parse_args()

    data_dir = args.data_dir
    output_dir = os.path.join(SCRIPT_DIR, 'eval_preds')
    os.makedirs(output_dir, exist_ok=True)

    # ── Load all sequences ──
    all_seqs = {}
    for f in ['Brown_Firth_2025_RdRp/evaluation_data/all_sequences.faa',
              'Orthototiviridae_Ghabrivirales/evaluation_data/sequences.faa']:
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            s = load_fasta(p)
            all_seqs.update(s)
            print(f'{f}: {len(s)} sequences')

    rdrp_dir = os.path.join(data_dir, 'RdRp-scan/evaluation_data')
    if os.path.exists(rdrp_dir):
        for f in sorted(glob.glob(os.path.join(rdrp_dir, '*.fasta'))):
            # Include the main RdRp-scan DB (15k seqs) — many tree leaves match here
            s = load_fasta(f)
            all_seqs.update(s)
            print(f'RdRp-scan/{os.path.basename(f)}: {len(s)} sequences')
    print(f'Total sequences loaded: {len(all_seqs)}')

    # ── Find all tree files ──
    tree_files = []
    bf_dir = os.path.join(data_dir, 'Brown_Firth_2025_RdRp/supplementary_data')
    for root, dirs, files in os.walk(bf_dir):
        for f in files:
            if f.endswith('.tre'):
                tree_files.append(os.path.join(root, f))

    gvdb_dir = os.path.join(data_dir, 'GVDB_giant_virus/evaluation_data')
    for f in glob.glob(os.path.join(gvdb_dir, '*.treefile')):
        tree_files.append(f)

    ortho_nwk = os.path.join(data_dir, 
        'Orthototiviridae_Ghabrivirales/evaluation_data/reference.nwk')
    if os.path.exists(ortho_nwk):
        tree_files.append(ortho_nwk)

    print(f'Total tree files: {len(tree_files)}')

    # Directory containing MAFFT-aligned .aln files
    bf_aln_dir = os.path.join(data_dir, 
        'Brown_Firth_2025_RdRp/evaluation_data')
    supp_dir = os.path.join(data_dir, 
        'Brown_Firth_2025_RdRp/supplementary_data')

    # ── Evaluate ──
    results = []
    n_total = 0
    n_with_mafft = 0
    n_without = 0

    for tf in sorted(tree_files):
        try:
            with open(tf) as f: tree_str = f.read().strip()
        except:
            continue
        try:
            leaves = get_leaves(tree_str)
        except:
            continue
        if len(leaves) < 4:
            continue

        # Load local + global sequences
        tree_dir = os.path.dirname(tf)
        local_seqs, has_local = load_local_sequences(tree_dir, all_seqs)

        # Match leaves
        matched = {}
        for leaf in leaves:
            key = match_leaf(leaf, local_seqs)
            if key:
                matched[leaf] = local_seqs[key]

        if len(matched) < 4:
            continue

        # Try to use MAFFT-aligned sequences from evaluation_data
        try:
            rel_path = os.path.relpath(tf, supp_dir)
        except:
            rel_path = os.path.basename(tf)
        
        aln_path = find_mafft_aln(bf_aln_dir, rel_path)
        mafft_used = False
        
        if aln_path:
            mafft_seqs = load_fasta(aln_path)
            replaced = 0
            for leaf_name in list(matched.keys()):
                key = match_leaf(leaf_name, mafft_seqs)
                if key and len(mafft_seqs[key]) >= 10:
                    # Check if it's actually aligned (all same length)
                    matched[leaf_name] = mafft_seqs[key]
                    replaced += 1
            if replaced > 0:
                mafft_used = True

        # Prune tree
        ref_pruned = prune_tree(tree_str, list(matched.keys()))
        if ref_pruned is None:
            continue
        
        pruned_leaves = get_leaves(ref_pruned)
        seq_list = [matched[n] for n in pruned_leaves]

        # Check if sequences are aligned (all same length)
        seq_lens = set(len(s) for s in seq_list)
        aligned = len(seq_lens) <= 1

        n_total += 1
        if mafft_used:
            n_with_mafft += 1
        else:
            n_without += 1

        # Random baseline
        try:
            rt = rand_tree(pruned_leaves)
            m = normrf(rt, ref_pruned)
            if m:
                results.append({
                    'ds': os.path.relpath(tf, data_dir).replace('.tre', ''),
                    'n': len(pruned_leaves), 'method': 'random', 'aligned': aligned, **m
                })
        except:
            pass

        # Hamming
        try:
            ht = nj_tree(seq_list, pruned_leaves, ham_dist)
            m = normrf(ht, ref_pruned)
            if m:
                results.append({
                    'ds': os.path.relpath(tf, data_dir).replace('.tre', ''),
                    'n': len(pruned_leaves), 'method': 'hamming', 'aligned': aligned, **m
                })
        except:
            pass

        # SeqIdentity
        try:
            st = nj_tree(seq_list, pruned_leaves, sid_dist)
            m = normrf(st, ref_pruned)
            if m:
                results.append({
                    'ds': os.path.relpath(tf, data_dir).replace('.tre', ''),
                    'n': len(pruned_leaves), 'method': 'seqidentity', 'aligned': aligned, **m
                })
        except:
            pass

        if n_total % 25 == 0:
            h_vals = [r['norm_rf'] for r in results if r['method'] == 'hamming']
            print(f'  [{n_total}/{len(tree_files)}] evaluated, '
                  f'hamming_avg={sum(h_vals)/len(h_vals):.4f}' if h_vals else '')

    # ── Summary ──
    print(f'\n{"="*60}')
    print(f'FULL LITERATURE REFERENCE EVALUATION (MAFFT)')
    print(f'{"="*60}')
    print(f'Total trees: {len(tree_files)}')
    print(f'Trees with ≥4 matched seqs: {n_total}')
    print(f'  Using MAFFT-aligned: {n_with_mafft}')
    print(f'  Without MAFFT alignment: {n_without}')
    print()

    methods = ['random', 'hamming', 'seqidentity']
    for method in methods:
        m_results = [r for r in results if r['method'] == method]
        if not m_results:
            continue
        vals = [r['norm_rf'] for r in m_results]
        aligned_cnt = sum(1 for r in m_results if r.get('aligned'))
        avg = sum(vals) / len(vals)
        sv = sorted(vals)
        med = sv[len(vals) // 2]
        perf = sum(1 for v in vals if v == 0) / len(vals) * 100
        worst = sum(1 for v in vals if v >= 0.98) / len(vals) * 100
        print(f'  {method:<15}: n={len(vals):>3}  avg={avg:.4f}  med={med:.4f}  '
              f'perfect={perf:.1f}%  worst={worst:.1f}%  aligned={aligned_cnt}')

    # Save
    out_path = os.path.join(SCRIPT_DIR, args.output)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'method', 'n_seqs', 'aligned', 'normRF', 'rf', 'max_rf'])
        for r in sorted(results, key=lambda x: (x['ds'], x['method'])):
            writer.writerow([
                r['ds'], r['method'], r['n'], r.get('aligned', False),
                f"{r['norm_rf']:.4f}", r['rf'], r['max_rf']
            ])
    print(f'\nSaved: {out_path} ({len(results)} rows)')


if __name__ == '__main__':
    main()
