#!/usr/bin/env python3
"""Run PHYLA + ESM2 on literature reference datasets (GPU)."""
import os, sys, re, pickle, csv, argparse, torch
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR); sys.path.insert(0, os.path.join(SCRIPT_DIR, 'phyla'))
from ete3 import Tree; from Bio import Phylo, SeqIO; from io import StringIO

# ── Tree utilities ──
def rm_bl(s):
    t = Phylo.read(StringIO(s), 'newick')
    for n in t.get_nonterminals(): n.branch_length = None
    for n in t.get_terminals(): n.branch_length = None
    o = StringIO(); Phylo.write(t, o, 'newick')
    return re.sub(r':[^,();\n]*', '', o.getvalue()).replace("'", '')

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

# ── Model inference ──
def encode_sequences(sequences, names, model, device):
    from phyla.dataset.data import Arbitrary_Sequence_Dataset
    ds = Arbitrary_Sequence_Dataset()
    batch, nms = ds.encode_sequences(sequences, names)
    with torch.no_grad():
        preds = model(
            batch['encoded_sequences'].to(device),
            batch['sequence_mask'].to(device),
            batch['cls_positions'].bool().to(device))
    tree = model.reconstruct_tree(preds, nms)
    return str(tree)

def esm2_nj(model, tok, seqs, names, dev):
    @torch.no_grad()
    def embed(sl):
        inp = tok(sl, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        inp = {k: v.to(dev) for k, v in inp.items()}
        h = model(**inp).last_hidden_state
        m = inp['attention_mask'].unsqueeze(-1).float()
        return ((h * m).sum(1) / m.sum(1)).cpu()
    from skbio import DistanceMatrix; from skbio.tree import nj
    emb = embed(seqs)
    n = len(names); dm = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = torch.cdist(emb[i:i+1], emb[j:j+1]).item()
            dm[i][j] = dm[j][i] = d
    return nj(DistanceMatrix(dm, names)).__str__().replace(' ', '')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='eval_preds/literature_eval_dataset_v2.pickle')
    ap.add_argument('--output', default='eval_preds/literature_gpu_results_v2.csv')
    ap.add_argument('--model', choices=['phyla', 'esm2', 'both'], default='both')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    datasets = pickle.load(open(os.path.join(SCRIPT_DIR, args.dataset), 'rb'))
    print(f'Loaded {len(datasets)} datasets')

    # Filter to datasets with sequences
    usable = [d for d in datasets if len(d.get('seqs',{})) >= 4]
    print(f'Usable: {len(usable)}')

    # Load models
    phyla_model = None; esm2_model = None; esm2_tok = None

    if args.model in ('phyla', 'both'):
        print('Loading PHYLA-beta...')
        from phyla import phyla; from phyla.utils.eval_configs import Config, Mamba_ModelConfig
        config = Config(); config.model = Mamba_ModelConfig()
        config.model.d_model = 256; config.model.n_layer = 16; config.model.vocab_size = 24
        config.model.num_blocks = 3; config.model.model_name = 'Phyla-beta'
        config.model.bidirectional = True; config.model.bidirectional_strategy = 'add'
        config.model.bidirectional_weight_tie = True
        phyla_model = phyla(config, device=args.device).load(
            os.path.join(SCRIPT_DIR, 'weights/11564369'))
        phyla_model.eval()
        print(f'  PHYLA loaded ({(sum(p.numel() for p in phyla_model.parameters())/1e6):.0f}M)')

    if args.model in ('esm2', 'both'):
        print('Loading ESM2-650M...')
        from transformers import AutoModel, AutoTokenizer
        esm2_tok = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
        esm2_model = AutoModel.from_pretrained('facebook/esm2_t33_650M_UR50D').to(device).eval()
        print(f'  ESM2 loaded ({sum(p.numel() for p in esm2_model.parameters())/1e6:.0f}M)')

    results = []
    for idx, ds in enumerate(usable):
        ds_name = ds['ds_name']
        ref_tree = ds['tree_str']
        seqs = ds['seqs']
        names = sorted(seqs.keys())
        seq_list = [seqs[n] for n in names]

        ref_pruned = prune(ref_tree, names)
        if ref_pruned is None: continue

        print(f'  [{idx+1}/{len(usable)}] {ds_name[:60]:<60} {len(names):>4} seqs', end='')

        # PHYLA
        if phyla_model:
            try:
                pt = encode_sequences(seq_list, names, phyla_model, device)
                m = normrf(pt, ref_pruned)
                if m: results.append({'ds': ds_name, 'n': len(names), 'method': 'phyla', **m})
                print(f'  P={m["norm_rf"]:.4f}' if m else '  P=FAIL', end='')
            except Exception as e:
                print(f'  P=ERR:{str(e)[:20]}', end='')

        # ESM2
        if esm2_model:
            try:
                et = esm2_nj(esm2_model, esm2_tok, seq_list, names, device)
                m = normrf(et, ref_pruned)
                if m: results.append({'ds': ds_name, 'n': len(names), 'method': 'esm2', **m})
                print(f'  E={m["norm_rf"]:.4f}' if m else '  E=FAIL', end='')
            except Exception as e:
                print(f'  E=ERR:{str(e)[:20]}', end='')

        print()

    # Summary
    print(f'\n{"="*60}')
    print('LITERATURE REFERENCE — GPU RESULTS')
    print(f'{"="*60}')
    for m in ['phyla', 'esm2']:
        m_r = [r for r in results if r['method'] == m]
        if not m_r: continue
        vals = [r['norm_rf'] for r in m_r]
        avg = sum(vals)/len(vals)
        sv = sorted(vals); med = sv[len(vals)//2]
        perf = sum(1 for v in vals if v==0)/len(vals)*100
        worst = sum(1 for v in vals if v>=0.98)/len(vals)*100
        print(f'  {m:<15}: n={len(vals):>3}  avg={avg:.4f}  med={med:.4f}  perf={perf:.1f}%  worst={worst:.1f}%')

    # Combined comparison
    print(f'\n  {"Method":<15} {"n":>6} {"Avg":>8}')
    hamming_vals = []
    with open(os.path.join(SCRIPT_DIR, 'eval_preds/literature_results.csv')) as f:
        import csv
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 5 and row[1] == 'hamming':
                try: hamming_vals.append(float(row[3]))
                except: pass
    if hamming_vals:
        print(f'  {"hamming":<15} {len(hamming_vals):>6} {sum(hamming_vals)/len(hamming_vals):>8.4f}')
    for m in ['phyla', 'esm2']:
        m_r = [r for r in results if r['method'] == m]
        if m_r:
            print(f'  {m:<15} {len(m_r):>6} {sum(r["norm_rf"] for r in m_r)/len(m_r):>8.4f}')
    print(f'  {"random (expected 1.0)":<15} {"—":>6} {"1.0000":>8}')

    if results:
        with open(os.path.join(SCRIPT_DIR, args.output), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows([['dataset', 'method', 'n_seqs', 'normRF', 'rf', 'max_rf']])
            for r in results:
                w.writerow([r['ds'], r['method'], r['n'], f"{r['norm_rf']:.4f}", r['rf'], r['max_rf']])
        print(f'\nSaved: {args.output}')

if __name__ == '__main__':
    main()
