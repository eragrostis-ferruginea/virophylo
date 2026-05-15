#!/usr/bin/env python3
"""
Evaluate literature reference trees with PHYLA, ESM2, Hamming, and Random baselines.
This replaces the 8-tree TreeBase ground-truth benchmark.

Usage:
    # CPU baselines (Hamming, SeqIdentity, Random)
    python evaluate_literature_refs_new.py --baselines hamming seqidentity random
    
    # GPU (PHYLA, ESM2)
    python evaluate_literature_refs_new.py --baselines phyla esm2 --device cuda:0
"""
import os, sys, re, pickle, csv, argparse, random, subprocess, torch
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'phyla'))

from ete3 import Tree
from Bio import Phylo, SeqIO
from io import StringIO

# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def rm_bl(tree_str):
    """Remove branch lengths from Newick tree."""
    t = Phylo.read(StringIO(tree_str), 'newick')
    for n in t.get_nonterminals():
        n.branch_length = None
    for n in t.get_terminals():
        n.branch_length = None
    o = StringIO()
    Phylo.write(t, o, 'newick')
    return re.sub(r':[^,();\n]*', '', o.getvalue()).replace("'", '')

def get_leaves(tree_str):
    """Get terminal leaf names from tree."""
    return sorted(str(x.name) for x in Phylo.read(StringIO(tree_str), 'newick').get_terminals())

def prune_tree(tree_str, keep_names):
    """Prune tree to keep only specified leaves."""
    clean = rm_bl(tree_str)
    t = Tree(clean)
    keep = sorted([n for n in keep_names if n in set(t.get_leaf_names())])
    if len(keep) < 4:
        return None
    t.prune(keep)
    return t.write(format=5).replace(' ', '').replace("'", '')

def norm_rf(pred, ref):
    """Calculate normalized Robinson-Foulds distance."""
    try:
        a = Tree(rm_bl(pred))
        b = Tree(rm_bl(ref))
        x = a.compare(b, unrooted=True)
        if isinstance(x.get('norm_rf'), str):
            return None
        return {
            'rf': int(x['rf']),
            'max_rf': int(x['max_rf']),
            'norm_rf': float(x['norm_rf'])
        }
    except Exception as e:
        return None

def ham_dist(s1, s2):
    """Hamming distance on aligned sequences."""
    if len(s1) != len(s2):
        return 1.0
    v = sum(1 for a, b in zip(s1, s2) if a not in '-.' and b not in '-.')
    if v == 0:
        return 0.5
    return 1.0 - sum(1 for a, b in zip(s1, s2) if a == b and a not in '-.X') / v

def sid_dist(s1, s2):
    """Sequence identity distance on aligned sequences."""
    if len(s1) != len(s2):
        return 1.0
    matches = sum(1 for a, b in zip(s1, s2) if a == b and a not in '-.X')
    aligned = sum(1 for a, b in zip(s1, s2) if a not in '-.X' and b not in '-.X')
    return 1.0 - matches / aligned if aligned else 1.0

def nj_tree(seqs, names, dist_func):
    """Build Neighbor-Joining tree from distance matrix."""
    from skbio import DistanceMatrix
    from skbio.tree import nj
    n = len(names)
    dm = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_func(seqs[i], seqs[j])
            dm[i][j] = dm[j][i] = d
    return nj(DistanceMatrix(dm, names)).__str__().replace(' ', '')

def random_tree(names):
    """Generate random balanced tree."""
    names = names[:]
    random.shuffle(names)
    return '(' + ','.join(names) + ');'

# ─────────────────────────────────────────────────────────────────────────────
# Model inference
# ─────────────────────────────────────────────────────────────────────────────

def encode_phylla(sequences, names, model, device):
    """Encode sequences with PHYLA and reconstruct tree."""
    from phyla.dataset.data import Arbitrary_Sequence_Dataset
    ds = Arbitrary_Sequence_Dataset()
    batch, nms = ds.encode_sequences(sequences, names)
    with torch.no_grad():
        preds = model(
            batch['encoded_sequences'].to(device),
            batch['sequence_mask'].to(device),
            batch['cls_positions'].bool().to(device)
        )
    tree = model.reconstruct_tree(preds, nms)
    return str(tree)

def encode_esm2(sequences, names, model, tokenizer, device):
    """Encode sequences with ESM2 and reconstruct NJ tree."""
    @torch.no_grad()
    def embed(seq_list):
        inp = tokenizer(seq_list, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        inp = {k: v.to(device) for k, v in inp.items()}
        h = model(**inp).last_hidden_state
        m = inp['attention_mask'].unsqueeze(-1).float()
        return ((h * m).sum(1) / m.sum(1)).cpu()
    
    emb = embed(sequences)
    n = len(names)
    dm = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.cdist(emb[i:i+1], emb[j:j+1]).item()
            dm[i][j] = dm[j][i] = d
    
    from skbio import DistanceMatrix
    from skbio.tree import nj
    return nj(DistanceMatrix(dm, names)).__str__().replace(' ', '')

# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate literature reference trees")
    parser.add_argument('--dataset', default='eval_preds/literature_refs_dataset.pickle',
                       help='Input pickle with datasets')
    parser.add_argument('--output', default='eval_preds/literature_refs_results.csv',
                       help='Output CSV file')
    parser.add_argument('--baselines', nargs='+',
                       default=['hamming', 'seqidentity', 'random'],
                       choices=['phyla', 'esm2', 'hamming', 'seqidentity', 'random'],
                       help='Baselines to evaluate')
    parser.add_argument('--device', default='cuda:0', help='GPU device')
    args = parser.parse_args()
    
    # Load datasets
    dataset_path = os.path.join(SCRIPT_DIR, args.dataset)
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found: {dataset_path}")
        print("Please run prepare_literature_dataset.py first")
        sys.exit(1)
    
    datasets = pickle.load(open(dataset_path, 'rb'))
    print(f"Loaded {len(datasets)} datasets")
    
    # Filter usable datasets
    usable = [d for d in datasets if len(d.get('seqs', {})) >= 4]
    print(f"Usable datasets (>=4 seqs): {len(usable)}")
    
    # Initialize models
    phyla_model = None
    esm2_model = None
    esm2_tok = None
    
    if 'phyla' in args.baselines:
        print("Loading PHYLA-beta...")
        from phyla import phyla
        from phyla.utils.eval_configs import Config, Mamba_ModelConfig
        config = Config()
        config.model = Mamba_ModelConfig()
        config.model.d_model = 256
        config.model.n_layer = 16
        config.model.vocab_size = 24
        config.model.num_blocks = 3
        config.model.model_name = "Phyla-beta"
        config.model.bidirectional = True
        config.model.bidirectional_strategy = 'add'
        config.model.bidirectional_weight_tie = True
        phyla_model = phyla(config, device=args.device).load(
            os.path.join(SCRIPT_DIR, 'weights/11564369'))
        phyla_model.eval()
        params = sum(p.numel() for p in phyla_model.parameters()) / 1e6
        print(f"  PHYLA loaded ({params:.0f}M params)")
    
    if 'esm2' in args.baselines:
        print("Loading ESM2-650M...")
        from transformers import AutoModel, AutoTokenizer
        esm2_tok = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
        esm2_model = AutoModel.from_pretrained('facebook/esm2_t33_650M_UR50D')
        esm2_model = esm2_model.to(args.device).eval()
        params = sum(p.numel() for p in esm2_model.parameters()) / 1e6
        print(f"  ESM2 loaded ({params:.0f}M params)")
    
    # Evaluate
    results = []
    for idx, ds in enumerate(usable):
        ds_name = ds['ds_name']
        ref_tree = ds['tree_str']
        seqs = ds['seqs']
        names = sorted(seqs.keys())
        seq_list = [seqs[n] for n in names]
        
        # Prune reference tree to available leaves
        ref_pruned = prune_tree(ref_tree, names)
        if ref_pruned is None:
            print(f"[{idx+1}/{len(usable)}] {ds_name[:60]:<60} SKIP (prune fail)")
            continue
        
        pruned_names = get_leaves(ref_pruned)
        pruned_seqs = [seqs[n] for n in pruned_names]
        
        print(f"[{idx+1}/{len(usable)}] {ds_name[:60]:<60} n={len(pruned_names):>4}", end='')
        
        # Random baseline
        if 'random' in args.baselines:
            try:
                rt = random_tree(pruned_names)
                m = norm_rf(rt, ref_pruned)
                if m:
                    results.append({
                        'ds': ds_name, 'n': len(pruned_names), 'method': 'random', **m
                    })
            except:
                pass
        
        # Hamming baseline (requires aligned sequences)
        if 'hamming' in args.baselines:
            try:
                ht = nj_tree(pruned_seqs, pruned_names, ham_dist)
                m = norm_rf(ht, ref_pruned)
                if m:
                    results.append({
                        'ds': ds_name, 'n': len(pruned_names), 'method': 'hamming', **m
                    })
            except Exception as e:
                print(f" HAM_ERR:{str(e)[:20]}", end='')
        
        # SeqIdentity baseline
        if 'seqidentity' in args.baselines:
            try:
                st = nj_tree(pruned_seqs, pruned_names, sid_dist)
                m = norm_rf(st, ref_pruned)
                if m:
                    results.append({
                        'ds': ds_name, 'n': len(pruned_names), 'method': 'seqidentity', **m
                    })
            except Exception as e:
                print(f" SID_ERR:{str(e)[:20]}", end='')
        
        # PHYLA
        if 'phyla' in args.baselines and phyla_model:
            try:
                pt = encode_phylla(pruned_seqs, pruned_names, phyla_model, args.device)
                m = norm_rf(pt, ref_pruned)
                if m:
                    results.append({
                        'ds': ds_name, 'n': len(pruned_names), 'method': 'phyla', **m
                    })
                    print(f" P={m['norm_rf']:.4f}", end='')
            except Exception as e:
                print(f" P_ERR:{str(e)[:15]}", end='')
        
        # ESM2
        if 'esm2' in args.baselines and esm2_model:
            try:
                et = encode_esm2(pruned_seqs, pruned_names, esm2_model, esm2_tok, args.device)
                m = norm_rf(et, ref_pruned)
                if m:
                    results.append({
                        'ds': ds_name, 'n': len(pruned_names), 'method': 'esm2', **m
                    })
                    print(f" E={m['norm_rf']:.4f}", end='')
            except Exception as e:
                print(f" E_ERR:{str(e)[:15]}", end='')
        
        print()
    
    # Summary
    print(f'\n{"="*60}')
    print(f'LITERATURE REFERENCE TREES — EVALUATION RESULTS')
    print(f'{"="*60}')
    print(f'Total datasets: {len(usable)}')
    print()
    
    methods = ['hamming', 'seqidentity', 'phyla', 'esm2', 'random']
    summary = {}
    for method in methods:
        m_results = [r for r in results if r['method'] == method]
        if not m_results:
            continue
        vals = [r['norm_rf'] for r in m_results]
        avg = sum(vals) / len(vals)
        sv = sorted(vals)
        med = sv[len(vals) // 2]
        perf = sum(1 for v in vals if v == 0) / len(vals) * 100
        worst = sum(1 for v in vals if v >= 0.98) / len(vals) * 100
        
        summary[method] = {
            'n': len(vals),
            'avg': avg,
            'med': med,
            'perf': perf,
            'worst': worst
        }
        print(f"  {method:<15}: n={len(vals):>3}  avg={avg:.4f}  med={med:.4f}  "
              f"perfect={perf:.1f}%  worst={worst:.1f}%")
    
    # Save results
    if results:
        output_path = os.path.join(SCRIPT_DIR, args.output)
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'method', 'n_seqs', 'normRF', 'rf', 'max_rf'])
            for r in results:
                writer.writerow([
                    r['ds'], r['method'], r['n'],
                    f"{r['norm_rf']:.4f}", r['rf'], r['max_rf']
                ])
        print(f'\nSaved: {output_path} ({len(results)} rows)')
        
        # Save summary
        summary_path = output_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"LITERATURE REFERENCE TREES — EVALUATION SUMMARY\n")
            f.write(f"{'='*60}\n")
            f.write(f"Total datasets: {len(usable)}\n\n")
            for method, s in sorted(summary.items()):
                f.write(f"{method:<15}: n={s['n']:>3}  avg={s['avg']:.4f}  "
                        f"med={s['med']:.4f}  perfect={s['perf']:.1f}%  "
                        f"worst={s['worst']:.1f}%\n")
        print(f"Summary: {summary_path}")

if __name__ == "__main__":
    main()
