#!/usr/bin/env python3
"""
Ground-Truth Virus Phylogeny Evaluation: TreeBase Expert Trees.

Runs Hamming+MSA, SeqIdentity+MSA, and Random baselines against 6 published
expert-curated virus protein phylogenetic trees from TreeBase.

PHYLA requires separate GPU inference (see run_treebase_phyla.py).
Results from this script establish the baselines; PHYLA comparison follows.

Usage:
  sbatch run_treebase_gt_slurm.sh
"""
import os, sys, re, csv, pickle, argparse, random, math
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'phyla'))

from ete3 import Tree
from Bio import Phylo
from Bio import SeqIO
from io import StringIO


# ── Tree utilities ──────────────────────────────────────────────

def remove_branch_distances(tree_str):
    t = Phylo.read(StringIO(tree_str), 'newick')
    for n in t.get_nonterminals():
        n.branch_length = None
    for n in t.get_terminals():
        n.branch_length = None
    o = StringIO()
    Phylo.write(t, o, 'newick')
    s = o.getvalue()
    s = re.sub(r':[^,();\n]*', '', s)
    return s.replace("'", '')


def get_leaf_names(tree_str):
    return sorted(str(x.name) for x in Phylo.read(StringIO(tree_str), 'newick').get_terminals())


def prune_tree_to_leaves(tree_str, leaves):
    clean = remove_branch_distances(tree_str)
    t = Tree(clean)
    keep = sorted([l for l in leaves if l in set(t.get_leaf_names())])
    if len(keep) < 4:
        return None
    t.prune(keep)
    return t.write(format=5).replace(' ', '').replace("'", '')


def compute_normrf(pred_str, ref_str):
    try:
        t1 = Tree(remove_branch_distances(pred_str))
        t2 = Tree(remove_branch_distances(ref_str))
        r = t1.compare(t2, unrooted=True)
        if isinstance(r['norm_rf'], str):
            return None
        return {'rf': int(r['rf']), 'max_rf': int(r['max_rf']), 'norm_rf': r['norm_rf']}
    except:
        return None


# ── Distance functions ──────────────────────────────────────────

def hamming_distance(s1, s2):
    if len(s1) != len(s2): return 1.0
    v = sum(1 for a,b in zip(s1,s2) if a!='-' and a!='.' and b!='-' and b!='.')
    if v == 0: return 0.5
    return 1.0 - sum(1 for a,b in zip(s1,s2) if a==b and a not in '-.X')/v

def seqid_distance(s1, s2):
    if len(s1) != len(s2): return 1.0
    ident = sum(1 for a,b in zip(s1,s2) if a==b and a not in '-.X')
    align = sum(1 for a,b in zip(s1,s2) if a not in '-.X' and b not in '-.X')
    return 1.0 - ident/align if align else 1.0

def build_nj_tree(sequences, names, dist_func):
    from skbio import DistanceMatrix
    from skbio.tree import nj
    n = len(names)
    dm = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = dist_func(sequences[i], sequences[j])
            dm[i][j] = dm[j][i] = d
    return nj(DistanceMatrix(dm, names)).__str__().replace(' ', '')

def build_random_tree(names):
    s = names[:]; random.shuffle(s)
    return '(' + ','.join(s) + ');'


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='eval_preds')
    args = parser.parse_args()

    # Define TreeBase virus families with expert trees
    TBFAMILIES = [
        ('TB2:S10171Taxa1', 'Phage/virus terminase large subunit'),
        ('TB2:S10521', 'Poxvirus protein'),
        ('TB2:S12677Taxa1', 'RHDV calicivirus (Lagovirus)'),
        ('TB2:S12677Taxa2', 'RHDV calicivirus (full set)'),
        ('TB2:S12857Taxa1', 'Viral metagenomics protein'),
        ('TB2:S13909Taxa1', 'Fungal virus capsid protein'),
        ('TB2:S13955Taxa5', 'Plant virus polyprotein'),
        ('TB2:S1458', 'Plant potyvirus'),
    ]

    seq_dir = os.path.join(SCRIPT_DIR, 'treebase_benchmark', 'sequences')
    tree_dir = os.path.join(SCRIPT_DIR, 'treebase_benchmark', 'trees')
    tmp_dir = os.path.join(SCRIPT_DIR, 'treebase_benchmark', 'tmp_align')
    os.makedirs(tmp_dir, exist_ok=True)

    print('=' * 60)
    print('GROUND-TRUTH VIRUS EVALUATION: TreeBase Expert Trees')
    print('=' * 60)
    print(f'Families: {len(TBFAMILIES)} curated virus protein families')
    print(f'Reference: published, expert-validated trees')

    all_results = []
    methods = ['Hamming', 'SeqIdentity', 'random']

    for family_id, description in TBFAMILIES:
        fa_path = os.path.join(seq_dir, f'{family_id}_processed.fa')
        tree_path = os.path.join(tree_dir, f'{family_id}_processed_tree.nh')

        if not os.path.exists(fa_path) or not os.path.exists(tree_path):
            print(f'\n  SKIP {family_id}: missing data')
            continue

        # Load expert reference tree
        with open(tree_path) as f:
            ref_tree_str = f.read().strip()

        # Load raw sequences
        raw_seqs = {}
        for record in SeqIO.parse(fa_path, 'fasta'):
            raw_seqs[record.id] = str(record.seq)

        ref_leaves = get_leaf_names(ref_tree_str)
        seq_names = sorted(raw_seqs.keys())
        common_leaves = sorted(set(ref_leaves) & set(seq_names))

        if len(common_leaves) < 4:
            print(f'\n  SKIP {family_id}: <4 common leaves ({len(common_leaves)})')
            continue

        # Prune reference tree
        if set(ref_leaves) != set(common_leaves):
            ref_tree_str = prune_tree_to_leaves(ref_tree_str, common_leaves)
            if ref_tree_str is None:
                continue

        # Step 1: Run MAFFT on the raw sequences to produce MSA
        print(f'\n{"─"*55}')
        print(f'  {family_id} ({len(common_leaves)} sequences)')
        print(f'  Description: {description}')
        print(f'  Aligning with MAFFT...')

        raw_fa = os.path.join(tmp_dir, f'{family_id}_raw.fa')
        ali_fa = os.path.join(tmp_dir, f'{family_id}_aligned.fa')
        with open(raw_fa, 'w') as f:
            for name in common_leaves:
                f.write(f'>{name}\n{raw_seqs[name]}\n')

        import subprocess
        result = subprocess.run(
            ['mafft', '--quiet', '--auto', raw_fa],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f'    MAFFT FAILED: {result.stderr[:100]}')
            continue
        with open(ali_fa, 'w') as f:
            f.write(result.stdout)

        # Load aligned sequences
        ali_seqs = {}
        for record in SeqIO.parse(ali_fa, 'fasta'):
            ali_seqs[record.id] = str(record.seq)

        ali_names = sorted(set(ali_seqs.keys()) & set(common_leaves))
        if len(ali_names) < 4:
            print(f'    Too few aligned sequences')
            continue

        seq_list = [ali_seqs[n] for n in ali_names]

        # Step 2: Evaluate baselines
        for method in ['Hamming', 'SeqIdentity', 'random']:
            if method in ('Hamming', 'SeqIdentity'):
                dist_func = hamming_distance if method == 'Hamming' else seqid_distance
                try:
                    pred_tree_str = build_nj_tree(seq_list, ali_names, dist_func)
                    metric = compute_normrf(pred_tree_str, ref_tree_str)
                except Exception as e:
                    print(f'    {method:<15} ERROR: {str(e)[:80]}')
                    metric = None
            elif method == 'random':
                pred_tree_str = build_random_tree(ali_names)
                metric = compute_normrf(pred_tree_str, ref_tree_str)

            if metric:
                print(f'    {method:<15} normRF={metric["norm_rf"]:.4f}  '
                      f'(RF={metric["rf"]}/{metric["max_rf"]})')
                all_results.append({
                    'family': family_id,
                    'n_seqs': len(ali_names),
                    'method': method,
                    'normRF': metric['norm_rf'],
                    'rf': metric['rf'],
                    'max_rf': metric['max_rf'],
                })

    # Summary
    print(f'\n{"=" * 60}')
    print('  TREE BASE GROUND-TRUTH RESULTS')
    print(f'{"=" * 60}')
    print(f'  {"Method":<15} {"n":>6} {"Avg normRF":>12} {"Median":>8}')
    print(f'  {"-"*41}')

    for method in methods:
        m_results = [r for r in all_results if r['method'] == method]
        n = len(m_results)
        if n == 0:
            print(f'  {method:<15} {"—":>6} {"—":>12} {"—":>8}')
            continue
        avg = sum(r['normRF'] for r in m_results) / n
        sorted_nrf = sorted(r['normRF'] for r in m_results)
        med = sorted_nrf[n // 2]
        print(f'  {method:<15} {n:>6} {avg:>12.4f} {med:>8.4f}')

    # Save CSV
    if all_results:
        out = os.path.join(SCRIPT_DIR, args.output_dir, 'treebase_groundtruth.csv')
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows([['family', 'n_seqs', 'method', 'normRF', 'rf', 'max_rf']])
            for r in all_results:
                w.writerow([r['family'], r['n_seqs'], r['method'],
                           f"{r['normRF']:.4f}", r['rf'], r['max_rf']])
        print(f'\n  Saved: {out}')

    print(f'\n  IMPORTANT: These results compare against EXPERT-CURATED')
    print(f'  reference trees — real published phylogenies, not algorithmic')
    print(f'  approximations. This is the closest to "ground truth" available.')
    print(f'  PHYLA + ESM2 pending GPU inference on these families.')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
