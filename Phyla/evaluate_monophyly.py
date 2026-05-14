#!/usr/bin/env python3
"""
ICTV Ground-Truth Monophyly Evaluation for Virus Phylogeny.

For each VOGDB family with ICTV taxonomic labels, builds a NJ tree from each method
and evaluates: does the tree correctly group sequences by their ICTV genus/family?

Metric: Monophyly Score = fraction of ICTV clades that are monophyletic in the tree.
A clade is monophyletic if all members form an exclusive subtree (no outsiders mixed in).

Methods compared:
  - PHYLA (CLS embedding + NJ, from FAA)
  - ESM2-650M (embedding + NJ, from FAA)  [if available]
  - Hamming + NJ (from MSA)
  - SeqIdentity + NJ (from MSA)
  - Random (shuffle tree)

Output: per-baseline monophyly scores, stratified by taxonomic level.
"""
import os, sys, re, csv, pickle, argparse, random
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'phyla'))

from ete3 import Tree
from Bio import Phylo
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
    s = s.replace("'", '')
    return s


def get_leaf_names(tree_str):
    t = Phylo.read(StringIO(tree_str), 'newick')
    return sorted(str(x.name) for x in t.get_terminals())


def prune_tree_to_leaves(tree_str, leaves_to_keep):
    clean = remove_branch_distances(tree_str)
    t = Tree(clean)
    tree_leaves = set(t.get_leaf_names())
    keep = sorted([l for l in leaves_to_keep if l in tree_leaves])
    if len(keep) < 4:
        return None
    t.prune(keep)
    return t.write(format=5).replace(' ', '').replace("'", '')


def load_msa_sequences(msa_path):
    seqs = {}
    name = None
    with open(msa_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                name = line[1:].split()[0]
                seqs[name] = ''
            elif name:
                seqs[name] += line
    return seqs


# ── Distance functions ──────────────────────────────────────────

def hamming_distance(seq1, seq2):
    if len(seq1) != len(seq2):
        return 1.0
    valid = sum(1 for a, b in zip(seq1, seq2) if a != '-' and a != '.' and b != '-' and b != '.')
    if valid == 0:
        return 0.5
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a not in ('-', '.', 'X'))
    return 1.0 - matches / valid


def seqid_distance(seq1, seq2):
    if len(seq1) != len(seq2):
        return 1.0
    identities = sum(1 for a,b in zip(seq1, seq2) if a == b and a not in ('-','.','X'))
    aligned = sum(1 for a,b in zip(seq1, seq2) if a not in ('-','.','X') and b not in ('-','.','X'))
    if aligned == 0:
        return 1.0
    return 1.0 - identities / aligned


def build_nj_tree(sequences, seq_names, distance_func):
    from skbio import DistanceMatrix
    from skbio.tree import nj
    n = len(seq_names)
    dm = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = distance_func(sequences[i], sequences[j])
            dm[i][j] = d
            dm[j][i] = d
    dm_obj = DistanceMatrix(dm, seq_names)
    tree = nj(dm_obj)
    return tree.__str__().replace(' ', '')


def build_random_tree(seq_names):
    shuffled = seq_names[:]
    random.shuffle(shuffled)
    return '(' + ','.join(shuffled) + ');'


# ── Monophyly evaluation ────────────────────────────────────────

def check_monophyly(tree_str, group_members, all_leaves):
    """
    Check if a set of leaves forms a monophyletic group in the tree.
    Monophyletic = all members are in one exclusive subtree, no outsiders.
    """
    clean = remove_branch_distances(tree_str)
    t = Tree(clean)

    leaves_in_tree = set(t.get_leaf_names())
    members_present = sorted([m for m in group_members if m in leaves_in_tree])
    if len(members_present) < 2:
        return None  # insufficient to test

    # Find the MRCA of all members, then check if all its descendants are members
    try:
        ancestor = t.get_common_ancestor(members_present)
    except:
        return False

    descendant_leaves = set(ancestor.get_leaf_names())
    # Monophyletic if all descendants are in the group
    outsiders = descendant_leaves - set(members_present)
    return len(outsiders) == 0


def evaluate_tree_monophyly(tree_str, seq_to_tax, tax_rank):
    """
    For each taxonomic group (genus/family) with >=2 members in the tree,
    check if it's monophyletic. Returns (n_monophyletic, n_total, score).
    """
    leaves = get_leaf_names(tree_str)
    # Build groups
    groups = defaultdict(list)
    for leaf in leaves:
        tax = seq_to_tax.get(leaf)
        if tax:
            if tax_rank == 'genus':
                key = tax[0]  # genus name
            elif tax_rank == 'family':
                key = tax[1]  # family name
            elif tax_rank == 'order':
                key = tax[2]  # order name
            else:
                continue
            if key:
                groups[key].append(leaf)

    n_mono = 0
    n_total = 0
    for key, members in groups.items():
        if len(members) >= 2:
            n_total += 1
            result = check_monophyly(tree_str, members, leaves)
            if result is True:
                n_mono += 1

    if n_total == 0:
        return None
    return n_mono, n_total, n_mono / n_total


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref-pickle', default='virus_data/vogdb_treefam_v2.pickle')
    parser.add_argument('--pred-pickle', default='virus_data/phyla_predictions.pickle')
    parser.add_argument('--ictv-map', default='virus_data/vogdb_ictv_map.pickle')
    parser.add_argument('--esm2-csv', default='eval_preds/virus_esm2_vs_fasttree.csv',
                        help='ESM2 results CSV (if available)')
    parser.add_argument('--faa-dir', default='virus_data/faa')
    parser.add_argument('--msa-dir', default='virus_data/msa')
    parser.add_argument('--output-dir', default='eval_preds')
    parser.add_argument('--max-families', type=int, default=0)
    parser.add_argument('--tax-rank', default='genus',
                        choices=['genus', 'family'],
                        help='Taxonomic level for monophyly test')
    args = parser.parse_args()

    ref_path = os.path.join(SCRIPT_DIR, args.ref_pickle)
    pred_path = os.path.join(SCRIPT_DIR, args.pred_pickle)
    ictv_path = os.path.join(SCRIPT_DIR, args.ictv_map)
    faa_dir = os.path.join(SCRIPT_DIR, args.faa_dir)
    msa_dir = os.path.join(SCRIPT_DIR, args.msa_dir)
    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print(f'ICTV Monophyly Evaluation — Rank: {args.tax_rank}')
    print('=' * 60)

    # Load data
    print('\nLoading data...')
    ref_data = pickle.load(open(ref_path, 'rb'))
    pred_data = pickle.load(open(pred_path, 'rb'))
    ictv_map = pickle.load(open(ictv_path, 'rb'))
    common = sorted(set(ref_data.keys()) & set(pred_data.keys()) & set(ictv_map.keys()))
    print(f'  Reference families: {len(ref_data)}')
    print(f'  PHYLA families:     {len(pred_data)}')
    print(f'  ICTV mapped families: {len(ictv_map)}')
    print(f'  Common intersection:  {len(common)}')

    if args.max_families > 0:
        common = common[:args.max_families]

    # Evaluate
    methods = ['phyla', 'hamming', 'seqidentity', 'random']
    all_scores = {m: [] for m in methods}  # list of (n_mono, n_total, score)
    csv_rows = {m: [] for m in methods}
    families_evaluated = 0
    families_skipped = 0

    print(f'\nEvaluating monophyly on {len(common)} families...')
    for idx, vid in enumerate(common):
        seq_to_tax = ictv_map[vid]
        # Filter to sequences that have ICTV labels
        labeled_seqs = {s: t for s, t in seq_to_tax.items() if t is not None}
        if len(labeled_seqs) < 4:
            families_skipped += 1
            continue

        # Get reference tree and prune to common leaves
        ref_tree_str = ref_data[vid]['tree_newick']
        ref_leaves = get_leaf_names(ref_tree_str)
        pred_leaves = sorted(pred_data[vid].get('seq_names', []))
        common_leaves = sorted(set(ref_leaves) & set(pred_leaves))
        if len(common_leaves) < 4:
            families_skipped += 1
            continue

        # Prune ref tree
        if set(ref_leaves) != set(common_leaves):
            pruned_ref = prune_tree_to_leaves(ref_tree_str, common_leaves)
            if pruned_ref is None:
                families_skipped += 1
                continue
            ref_tree_str = pruned_ref

        # --- PHYLA ---
        phyla_tree_str = pred_data[vid]['pred_tree_newick']
        if set(pred_leaves) != set(common_leaves):
            phyla_tree_str = prune_tree_to_leaves(phyla_tree_str, common_leaves)
            if phyla_tree_str is None:
                families_skipped += 1
                continue

        result = evaluate_tree_monophyly(phyla_tree_str, labeled_seqs, args.tax_rank)
        if result:
            all_scores['phyla'].append(result)
            csv_rows['phyla'].append([vid, 'phyla', args.tax_rank,
                                      result[0], result[1], f'{result[2]:.4f}'])

        # --- Hamming & SeqIdentity (from MSA) ---
        msa_path = os.path.join(msa_dir, f'{vid}.msa')
        if os.path.exists(msa_path):
            seqs = load_msa_sequences(msa_path)
            msa_names = sorted(set(seqs.keys()) & set(common_leaves))
            if len(msa_names) >= 4:
                msa_seqs = [seqs[n] for n in msa_names]
                try:
                    ham_tree = build_nj_tree(msa_seqs, msa_names, hamming_distance)
                    result = evaluate_tree_monophyly(ham_tree, labeled_seqs, args.tax_rank)
                    if result:
                        all_scores['hamming'].append(result)
                        csv_rows['hamming'].append([vid, 'hamming', args.tax_rank,
                                                     result[0], result[1], f'{result[2]:.4f}'])

                    seqid_tree = build_nj_tree(msa_seqs, msa_names, seqid_distance)
                    result = evaluate_tree_monophyly(seqid_tree, labeled_seqs, args.tax_rank)
                    if result:
                        all_scores['seqidentity'].append(result)
                        csv_rows['seqidentity'].append([vid, 'seqidentity', args.tax_rank,
                                                         result[0], result[1], f'{result[2]:.4f}'])
                except:
                    pass

        # --- Random ---
        if len(common_leaves) >= 4:
            rand_tree = build_random_tree(common_leaves)
            result = evaluate_tree_monophyly(rand_tree, labeled_seqs, args.tax_rank)
            if result:
                all_scores['random'].append(result)
                csv_rows['random'].append([vid, 'random', args.tax_rank,
                                            result[0], result[1], f'{result[2]:.4f}'])

        families_evaluated += 1
        if (idx + 1) % 2000 == 0:
            print(f'  [{idx+1}/{len(common)}] evaluated={families_evaluated}, '
                  f'scores so far: phyla={sum(s[2] for s in all_scores["phyla"])/max(1,len(all_scores["phyla"])):.3f}, '
                  f'hamming={sum(s[2] for s in all_scores["hamming"])/max(1,len(all_scores["hamming"])):.3f}')

    # Report
    print(f'\n{"=" * 60}')
    print(f'  ICTV MONOPHYLY RESULTS — Rank: {args.tax_rank}')
    print(f'{"=" * 60}')
    print(f'  Families evaluated: {families_evaluated}')
    print(f'  Families skipped:   {families_skipped}')
    print()

    for method in methods:
        scores = all_scores[method]
        n = len(scores)
        if n == 0:
            continue
        total_mono = sum(s[0] for s in scores)
        total_groups = sum(s[1] for s in scores)
        mean_score = sum(s[2] for s in scores) / n
        print(f'  {method:<15}: n={n:>5}  '
              f'total_groups={total_groups:>6}  '
              f'monophyletic={total_mono:>6}  '
              f'mean_score={mean_score:.4f}')

    # Save CSVs
    for method in methods:
        if csv_rows[method]:
            out = os.path.join(output_dir, f'monophyly_{args.tax_rank}_{method}.csv')
            with open(out, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows([['vfam', 'method', 'rank', 'n_mono', 'n_groups', 'score']])
                writer.writerows(csv_rows[method])
            print(f'  Saved: {out}')

    print(f'\n{"=" * 60}')
    print(f'  INTERPRETATION:')
    print(f'  Monophyly Score = fraction of ICTV {args.tax_rank} groups')
    print(f'  that form exclusive subtrees in the predicted tree.')
    print(f'  Higher = better agreement with known virus taxonomy.')
    print(f'  THIS IS A TRUE BIOLOGICAL GROUND TRUTH.')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
