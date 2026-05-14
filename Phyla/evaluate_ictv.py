#!/usr/bin/env python3
"""
ICTV Ground-Truth Virus Phylogeny Evaluation — 4-Tier Framework.

Tier 1: Monophyly Score — baseline: can the tree separate ICTV genera/families?
Tier 2: Triplet Concordance — correct inter-genus branching order?
Tier 3: Divergence Arbitration — when PHYLA disagrees with Hamming, who's closer to ICTV?
Tier 4: ESM2 Fair Comparison — both FAA-based pLMs, who aligns better with ICTV?

All methods: PHYLA, ESM2 (when available), Hamming+MSA, SeqIdentity+MSA, Random.

Usage (sbatch only):
  python evaluate_ictv.py --ictv-map virus_data/vogdb_ictv_map.pickle ...
"""
import os, sys, re, csv, pickle, argparse, random, math
from collections import defaultdict, Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'phyla'))

from ete3 import Tree
from Bio import Phylo
from io import StringIO


# ═══════════════════════════════════════════════════════════════
# Tree utilities
# ═══════════════════════════════════════════════════════════════

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


def load_msa_sequences(msa_path):
    seqs, name = {}, None
    with open(msa_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'): name, seqs[line[1:].split()[0]] = line[1:].split()[0], ''
            elif name: seqs[name] += line
    return seqs


# ═══════════════════════════════════════════════════════════════
# Distance functions + NJ
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Tier 1: Monophyly (baseline validation)
# ═══════════════════════════════════════════════════════════════

def evaluate_monophyly(tree_str, seq_to_tax, rank='genus'):
    clean = remove_branch_distances(tree_str)
    t = Tree(clean)
    leaves_in_tree = set(t.get_leaf_names())

    groups = defaultdict(list)
    for leaf in leaves_in_tree:
        tax = seq_to_tax.get(leaf)
        if tax:
            key = tax[0] if rank == 'genus' else tax[1] if rank == 'family' else None
            if key: groups[key].append(leaf)

    n_mono, n_total = 0, 0
    for members in groups.values():
        members_in_tree = [m for m in members if m in leaves_in_tree]
        if len(members_in_tree) < 2: continue
        n_total += 1
        try:
            anc = t.get_common_ancestor(members_in_tree)
            outsiders = set(anc.get_leaf_names()) - set(members_in_tree)
            if len(outsiders) == 0: n_mono += 1
        except: pass

    if n_total == 0: return None
    return {'n_mono': n_mono, 'n_total': n_total, 'score': n_mono/n_total}


# ═══════════════════════════════════════════════════════════════
# Tier 2: Triplet Concordance
# ═══════════════════════════════════════════════════════════════

def evaluate_triplet_concordance(tree_str, seq_to_tax, rank='genus'):
    """
    For all triples of taxonomic groups (A, B, C) with >=2 members each,
    check if the tree's branching order matches ICTV genus co-membership.
    
    "Correct" = in the tree, A and B are closer to each other than either is to C,
    IFF A and B belong to the SAME higher-level group (e.g., same family)
    while C belongs to a DIFFERENT higher-level group.
    
    Returns (n_correct, n_total, score).
    """
    clean = remove_branch_distances(tree_str)
    t = Tree(clean)
    leaves_in_tree = set(t.get_leaf_names())

    # Build groups
    groups = defaultdict(list)
    for leaf in leaves_in_tree:
        tax = seq_to_tax.get(leaf)
        if not tax: continue
        if rank == 'genus':
            key = tax[0]        # genus
            higher = tax[1]      # family
        elif rank == 'family':
            key = tax[1]
            higher = tax[2]      # order
        else:
            continue
        if key:
            groups[key] = (higher, [l for l in t.get_leaf_names() if l in set(groups.get(key, ('', []))[1]) or l == leaf])
            groups[key] = (higher, groups[key][1] + [leaf])
    
    # Only keep groups with >=2 members in tree
    valid_groups = {k: v for k, v in groups.items() if len(set(v[1]) & leaves_in_tree) >= 2}
    group_keys = sorted(valid_groups.keys())
    
    if len(group_keys) < 3:
        return None  # need >=3 groups for triplets
    
    # Precompute pairwise tree distances between group pairs
    # Distance = average pairwise tree distance between members
    # Use MRCA depth as proxy: shallower MRCA = closer
    # For each leaf pair, compute tree distance = nodes between them / max possible
    
    # Simpler approach: for each group, define a "representative" and compute pairwise
    # tree distances between group representatives using MRCA depth
    n_correct = 0
    n_total = 0
    
    for i in range(len(group_keys)):
        for j in range(i+1, len(group_keys)):
            for k in range(j+1, len(group_keys)):
                gi, gj, gk = group_keys[i], group_keys[j], group_keys[k]
                hi, hj, hk = valid_groups[gi][0], valid_groups[gj][0], valid_groups[gk][0]
                mem_i = list(set(valid_groups[gi][1]) & leaves_in_tree)[:5]
                mem_j = list(set(valid_groups[gj][1]) & leaves_in_tree)[:5]
                mem_k = list(set(valid_groups[gk][1]) & leaves_in_tree)[:5]
                
                # Compute average MRCA depth for each pair
                def avg_mrca_depth(members_a, members_b):
                    depths = []
                    for a in members_a:
                        for b in members_b:
                            try:
                                anc = t.get_common_ancestor([a, b])
                                depths.append(t.get_distance(anc, t.get_tree_root()))
                            except:
                                pass
                    return sum(depths)/len(depths) if depths else float('inf')
                
                d_ij = avg_mrca_depth(mem_i, mem_j)
                d_ik = avg_mrca_depth(mem_i, mem_k)
                d_jk = avg_mrca_depth(mem_j, mem_k)
                
                # Which pair is closest in the tree?
                pair_closest = min(
                    (d_ij, 'ij'), (d_ik, 'ik'), (d_jk, 'jk'),
                    key=lambda x: x[0]
                )[1]
                
                # Which pair should be closest based on ICTV taxonomy?
                # Same higher group = should be closer
                same_ij = (hi == hj and hi != '')
                same_ik = (hi == hk and hi != '')
                same_jk = (hj == hk and hj != '')
                
                n_same = same_ij + same_ik + same_jk
                if n_same == 0: continue  # all different higher groups — skip
                if n_same == 3: continue  # all same higher group — skip
                if n_same == 2: continue  # logically impossible if hi,hj,hk are from 2 groups
                
                # Exactly one pair shares higher group — that pair SHOULD be closest
                expected_closest = 'ij' if same_ij else 'ik' if same_ik else 'jk'
                
                n_total += 1
                if pair_closest == expected_closest:
                    n_correct += 1
    
    if n_total == 0: return None
    return {'n_correct': n_correct, 'n_total': n_total, 'score': n_correct/n_total}


# ═══════════════════════════════════════════════════════════════
# Tier 3: Divergence Arbitration
# ═══════════════════════════════════════════════════════════════

def evaluate_arbitration(phyla_result, hamming_result, seq_to_tax, rank='genus'):
    """
    On families where PHYLA and Hamming disagree significantly on ICTV clustering,
    which model's tree is more correct?
    
    Returns: {'phyla_wins': N, 'hamming_wins': N, 'tie': N}
    """
    phyla_wins = 0
    hamming_wins = 0
    tie = 0
    
    for vid in sorted(set(phyla_result.keys()) & set(hamming_result.keys())):
        pr = phyla_result[vid]
        hr = hamming_result[vid]
        if pr is None or hr is None: continue
        
        diff = abs(pr['score'] - hr['score'])
        if diff < 0.1: continue  # not enough divergence
        
        if pr['score'] > hr['score']:
            phyla_wins += 1
        elif hr['score'] > pr['score']:
            hamming_wins += 1
        else:
            tie += 1
    
    return {'phyla_wins': phyla_wins, 'hamming_wins': hamming_wins, 'tie': tie,
            'total': phyla_wins + hamming_wins + tie}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='ICTV 4-Tier Evaluation')
    parser.add_argument('--ref-pickle', default='virus_data/vogdb_treefam_v2.pickle')
    parser.add_argument('--pred-pickle', default='virus_data/phyla_predictions.pickle')
    parser.add_argument('--ictv-map', default='virus_data/vogdb_ictv_map.pickle')
    parser.add_argument('--faa-dir', default='virus_data/faa')
    parser.add_argument('--msa-dir', default='virus_data/msa')
    parser.add_argument('--output-dir', default='eval_preds')
    parser.add_argument('--max-families', type=int, default=0)
    parser.add_argument('--tiers', default='1,2,3', help='Comma-separated: 1=mono, 2=triplet, 3=arbitration')
    args = parser.parse_args()

    tiers = [int(t) for t in args.tiers.split(',')]

    ref_path = os.path.join(SCRIPT_DIR, args.ref_pickle)
    pred_path = os.path.join(SCRIPT_DIR, args.pred_pickle)
    ictv_path = os.path.join(SCRIPT_DIR, args.ictv_map)
    faa_dir = os.path.join(SCRIPT_DIR, args.faa_dir)
    msa_dir = os.path.join(SCRIPT_DIR, args.msa_dir)
    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print('ICTV Ground-Truth Virus Phylogeny Evaluation')
    print(f'Tiers active: {",".join(f"T{t}" for t in tiers)}')
    print('=' * 60)

    # Load data
    print('\nLoading data...')
    ref_data = pickle.load(open(ref_path, 'rb'))
    pred_data = pickle.load(open(pred_path, 'rb'))
    ictv_map = pickle.load(open(ictv_path, 'rb'))
    common = sorted(set(ref_data.keys()) & set(pred_data.keys()) & set(ictv_map.keys()))
    print(f'  Reference: {len(ref_data)}, PHYLA: {len(pred_data)}, ICTV map: {len(ictv_map)}')
    print(f'  Common intersection: {len(common)}')
    if args.max_families > 0:
        common = common[:args.max_families]
        print(f'  Limited to {len(common)}')

    # ── Pre-processing: prune trees ──
    print('\nPre-processing: pruning trees...')
    family_list = []  # (vid, phyla_tree_str, ref_tree_str, common_leaves)
    too_few = 0
    
    for vid in common:
        seq_to_tax = ictv_map[vid]
        labeled = [s for s,t in seq_to_tax.items() if t]
        if len(labeled) < 4:
            too_few += 1; continue
        
        ref_tree = ref_data[vid]['tree_newick']
        phyla_tree = pred_data[vid]['pred_tree_newick']
        ref_leaves = get_leaf_names(ref_tree)
        pred_leaves = sorted(pred_data[vid].get('seq_names', []))
        common_leaves = sorted(set(ref_leaves) & set(pred_leaves))
        if len(common_leaves) < 4:
            too_few += 1; continue
        
        if set(ref_leaves) != set(common_leaves):
            ref_tree = prune_tree_to_leaves(ref_tree, common_leaves)
            if ref_tree is None: too_few += 1; continue
        if set(pred_leaves) != set(common_leaves):
            phyla_tree = prune_tree_to_leaves(phyla_tree, common_leaves)
            if phyla_tree is None: too_few += 1; continue
        
        family_list.append((vid, phyla_tree, ref_tree, common_leaves, seq_to_tax))
    
    print(f'  Evaluable families: {len(family_list)} (too_few={too_few})')

    # ── Evaluate ──
    methods = ['phyla', 'hamming', 'seqidentity', 'random']
    mono_results = {m: {} for m in methods}  # {vid: mono_dict}
    triplet_results = {m: {} for m in methods}
    agg_mono = {m: {'scores': []} for m in methods}
    agg_triplet = {m: {'scores': []} for m in methods}

    for idx, (vid, phyla_tree, ref_tree, common_leaves, seq_to_tax) in enumerate(family_list):
        # PHYLA
        if 1 in tiers:
            r = evaluate_monophyly(phyla_tree, seq_to_tax, 'genus')
            if r: mono_results['phyla'][vid] = r; agg_mono['phyla']['scores'].append(r['score'])
        if 2 in tiers:
            r = evaluate_triplet_concordance(phyla_tree, seq_to_tax, 'genus')
            if r: triplet_results['phyla'][vid] = r; agg_triplet['phyla']['scores'].append(r['score'])

        # Hamming + SeqIdentity
        msa_path = os.path.join(msa_dir, f'{vid}.msa')
        if os.path.exists(msa_path):
            seqs = load_msa_sequences(msa_path)
            msa_names = sorted(set(seqs.keys()) & set(common_leaves))
            if len(msa_names) >= 4:
                msa_seqs = [seqs[n] for n in msa_names]
                try:
                    ham_tree = build_nj_tree(msa_seqs, msa_names, hamming_distance)
                    if 1 in tiers:
                        r = evaluate_monophyly(ham_tree, seq_to_tax, 'genus')
                        if r: mono_results['hamming'][vid] = r; agg_mono['hamming']['scores'].append(r['score'])
                    if 2 in tiers:
                        r = evaluate_triplet_concordance(ham_tree, seq_to_tax, 'genus')
                        if r: triplet_results['hamming'][vid] = r; agg_triplet['hamming']['scores'].append(r['score'])

                    sid_tree = build_nj_tree(msa_seqs, msa_names, seqid_distance)
                    if 1 in tiers:
                        r = evaluate_monophyly(sid_tree, seq_to_tax, 'genus')
                        if r: mono_results['seqidentity'][vid] = r; agg_mono['seqidentity']['scores'].append(r['score'])
                    if 2 in tiers:
                        r = evaluate_triplet_concordance(sid_tree, seq_to_tax, 'genus')
                        if r: triplet_results['seqidentity'][vid] = r; agg_triplet['seqidentity']['scores'].append(r['score'])
                except: pass

        # Random
        if len(common_leaves) >= 4:
            rand_tree = build_random_tree(common_leaves)
            if 1 in tiers:
                r = evaluate_monophyly(rand_tree, seq_to_tax, 'genus')
                if r: mono_results['random'][vid] = r; agg_mono['random']['scores'].append(r['score'])

        if (idx+1) % 2000 == 0:
            ms = lambda m: sum(agg_mono[m]['scores'])/max(1,len(agg_mono[m]['scores']))
            ts = lambda m: sum(agg_triplet[m]['scores'])/max(1,len(agg_triplet[m]['scores']))
            print(f'  [{idx+1}/{len(family_list)}] '
                  f'mono: P={ms("phyla"):.3f} H={ms("hamming"):.3f} R={ms("random"):.3f}  '
                  f'triplet: P={ts("phyla"):.3f} H={ts("hamming"):.3f}')

    # ── Report ──
    print(f'\n{"="*60}')
    print(f'  RESULTS')
    print(f'{"="*60}')

    if 1 in tiers:
        print(f'\n── Tier 1: Monophyly (genus-level) ──')
        print(f'  {"Method":<15} {"n":>6} {"Score":>8} {"Mono":>8} {"Total":>8}')
        print(f'  {"-"*45}')
        for m in methods:
            scores = agg_mono[m]['scores']
            n = len(scores)
            if n: print(f'  {m:<15} {n:>6} {sum(scores)/n:>8.4f} {"—":>8} {"—":>8}')

    if 2 in tiers:
        print(f'\n── Tier 2: Triplet Concordance (genus branching order) ──')
        print(f'  {"Method":<15} {"n":>6} {"Score":>8}')
        print(f'  {"-"*29}')
        for m in methods:
            scores = agg_triplet[m]['scores']
            n = len(scores)
            if n: print(f'  {m:<15} {n:>6} {sum(scores)/n:>8.4f}')

    if 3 in tiers and 'phyla' in mono_results and 'hamming' in mono_results:
        print(f'\n── Tier 3: Divergence Arbitration ──')
        arb = evaluate_arbitration(mono_results['phyla'], mono_results['hamming'],
                                    ictv_map, 'genus')
        if arb['total'] > 0:
            print(f'  Disagreeing families (delta>0.1): {arb["total"]}')
            print(f'  PHYLA wins:   {arb["phyla_wins"]} ({arb["phyla_wins"]/max(1,arb["total"])*100:.1f}%)')
            print(f'  Hamming wins: {arb["hamming_wins"]} ({arb["hamming_wins"]/max(1,arb["total"])*100:.1f}%)')
            print(f'  Tie:          {arb["tie"]}')
            if arb['phyla_wins'] > arb['hamming_wins']:
                print(f'  → PHYLA is MORE ICTV-congruent on hard cases!')
            elif arb['hamming_wins'] > arb['phyla_wins']:
                print(f'  → Hamming is MORE ICTV-congruent on hard cases.')
            else:
                print(f'  → No clear winner on hard cases.')

    print(f'\n{"="*60}')
    print(f'  INTERPRETATION GUIDE')
    print(f'{"="*60}')
    print(f'  T1 (Monophyly):    Baseline — can the tree separate ICTV genera?')
    print(f'                      Easy test; all methods including Random should')
    print(f'                      score >0 since same-genus seqs are similar.')
    print(f'  T2 (Triplet):     Hard test — does the tree get inter-genus')
    print(f'                      branching order right? Scores > random')
    print(f'                      indicate genuine phylogenetic signal.')
    print(f'  T3 (Arbitration): When PHYLA & Hamming disagree, which aligns')
    print(f'                      with ICTV? Key evidence for/against PHYLA.')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
