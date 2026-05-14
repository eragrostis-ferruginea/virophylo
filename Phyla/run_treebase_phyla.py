#!/usr/bin/env python3
"""
PHYLA inference on TreeBase virus protein families.
Runs PHYLA-beta encoding + NJ tree reconstruction on 6 TreeBase families
with expert-curated reference trees. Outputs normRF against ground truth.

Usage (sbatch): sbatch run_treebase_phyla_slurm.sh
"""
import sys, os, re, pickle, csv, argparse
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'phyla'))

import torch
from Bio import SeqIO
from phyla import phyla
from phyla.utils.eval_configs import Config, Mamba_ModelConfig
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


# ── PHYLA inference ─────────────────────────────────────────────

def encode_sequences_from_memory(sequences, sequence_names, model, device):
    """Encode protein sequences with PHYLA and build NJ tree."""
    from phyla.dataset.data import Arbitrary_Sequence_Dataset
    dataset = Arbitrary_Sequence_Dataset()
    batch, names = dataset.encode_sequences(sequences, sequence_names)

    with torch.no_grad():
        preds = model(
            batch['encoded_sequences'].to(device),
            batch['sequence_mask'].to(device),
            batch['cls_positions'].bool().to(device)
        )

    tree = model.reconstruct_tree(preds, names)
    return str(tree)


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='weights/11564369')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    TBFAMILIES = [
        ('TB2:S10171Taxa1', 'Phage terminase'),
        ('TB2:S10521', 'Poxvirus protein'),
        ('TB2:S12857Taxa1', 'Viral metagenomics'),
        ('TB2:S13909Taxa1', 'Fungal virus capsid'),
        ('TB2:S13955Taxa5', 'Plant virus polyprotein'),
        ('TB2:S1458', 'Plant potyvirus'),
    ]

    seq_dir = os.path.join(SCRIPT_DIR, 'treebase_benchmark', 'sequences')
    tree_dir = os.path.join(SCRIPT_DIR, 'treebase_benchmark', 'trees')
    output_dir = os.path.join(SCRIPT_DIR, 'eval_preds')
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print('PHYLA on TreeBase Ground-Truth Virus Families')
    print('=' * 60)

    # Load model
    print(f'\nLoading PHYLA-beta from {args.checkpoint}...')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')

    config = Config()
    config.model = Mamba_ModelConfig()
    config.model.d_model = 256
    config.model.n_layer = 16
    config.model.vocab_size = 24
    config.model.num_blocks = 3
    config.model.model_name = 'Phyla-beta'
    config.model.bidirectional = True
    config.model.bidirectional_strategy = 'add'
    config.model.bidirectional_weight_tie = True

    model = phyla(config, device=args.device).load(
        os.path.join(SCRIPT_DIR, args.checkpoint))
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {param_count/1e6:.0f}M')

    # Evaluate
    all_results = []

    for family_id, description in TBFAMILIES:
        fa_path = os.path.join(seq_dir, f'{family_id}_processed.fa')
        tree_path = os.path.join(tree_dir, f'{family_id}_processed_tree.nh')

        if not os.path.exists(fa_path) or not os.path.exists(tree_path):
            print(f'\n  SKIP {family_id}: missing data')
            continue

        # Load expert reference tree
        with open(tree_path) as f:
            ref_tree_str = f.read().strip()

        # Load sequences
        sequences = {}
        for record in SeqIO.parse(fa_path, 'fasta'):
            sequences[record.id] = str(record.seq)

        ref_leaves = get_leaf_names(ref_tree_str)
        seq_names = sorted(sequences.keys())
        common_leaves = sorted(set(ref_leaves) & set(seq_names))

        if len(common_leaves) < 4:
            print(f'\n  SKIP {family_id}: <4 common leaves ({len(common_leaves)})')
            continue

        # Prune reference tree
        if set(ref_leaves) != set(common_leaves):
            ref_tree_str = prune_tree_to_leaves(ref_tree_str, common_leaves)
            if ref_tree_str is None:
                continue

        seq_list = [sequences[n] for n in common_leaves]

        print(f'\n  {family_id} ({len(common_leaves)} seqs): {description}')
        print(f'    Encoding with PHYLA...')

        try:
            phyla_tree_str = encode_sequences_from_memory(
                seq_list, common_leaves, model, device)

            metric = compute_normrf(phyla_tree_str, ref_tree_str)

            if metric:
                print(f'    PHYLA normRF = {metric["norm_rf"]:.4f}  '
                      f'(RF={metric["rf"]}/{metric["max_rf"]})')
                all_results.append({
                    'family': family_id,
                    'n_seqs': len(common_leaves),
                    'method': 'PHYLA',
                    'normRF': metric['norm_rf'],
                    'rf': metric['rf'],
                    'max_rf': metric['max_rf'],
                })
            else:
                print(f'    PHYLA: normRF computation failed')
        except Exception as e:
            print(f'    PHYLA ERROR: {str(e)[:120]}')

    # Summary
    print(f'\n{"=" * 60}')
    print(f'  PHYLA vs EXPERT TREES — RESULTS')
    print(f'{"=" * 60}')

    phyla_results = [r for r in all_results if r['method'] == 'PHYLA']
    n = len(phyla_results)

    if n == 0:
        print('  No valid results.')
    else:
        avg = sum(r['normRF'] for r in phyla_results) / n
        sorted_nrf = sorted(r['normRF'] for r in phyla_results)
        med = sorted_nrf[n // 2]
        print(f'  Families: {n}')
        print(f'  Avg normRF: {avg:.4f}')
        print(f'  Median:     {med:.4f}')
        print(f'  Per-family:')
        for r in all_results:
            print(f'    {r["family"]:<20} {r["n_seqs"]:>4} seqs  normRF={r["normRF"]:.4f}')

    # Save CSV
    if all_results:
        out = os.path.join(output_dir, 'treebase_phyla_gt.csv')
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows([['family', 'n_seqs', 'method', 'normRF', 'rf', 'max_rf']])
            for r in all_results:
                w.writerow([r['family'], r['n_seqs'], r['method'],
                           f"{r['normRF']:.4f}", r['rf'], r['max_rf']])
        print(f'\n  Saved: {out}')

    print(f'\n{"=" * 60}')
    print(f'  COMPARISON TABLE (all methods vs Expert Trees)')
    print(f'{"=" * 60}')
    print(f'  Hamming avg:    0.394  (from evaluate_treebase_gt.py)')
    print(f'  PHYLA avg:      {avg:.4f}' if n > 0 else '  PHYLA: pending')
    print(f'  Random avg:     1.000  (as expected)')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
