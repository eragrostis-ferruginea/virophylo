"""
ESM2 + NJ baseline for VOGDB virus phylogeny evaluation.
For each VFAM, encodes sequences with ESM2-650M, computes pairwise distances,
builds NJ tree, and compares with FastTree reference (exact paper rf_distance).
"""
import sys
import os
import pickle
import csv
import argparse
import time
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "phyla"))

from ete3 import Tree
from Bio import Phylo
from Bio import SeqIO
from io import StringIO
from transformers import AutoModel, AutoTokenizer
from skbio import DistanceMatrix
from skbio.tree import nj
import re


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


@torch.no_grad()
def encode_sequences(model, tokenizer, sequences, max_len=1024):
    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    embeddings = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return embeddings.cpu()


def compute_normrf(pred_tree_str, ref_tree_str):
    def remove_branch_distances(s):
        t = Phylo.read(StringIO(s), "newick")
        for n in t.get_nonterminals():
            n.branch_length = None
        for n in t.get_terminals():
            n.branch_length = None
        o = StringIO()
        Phylo.write(t, o, "newick")
        s2 = o.getvalue()
        s2 = re.sub(r':[^,();\n]*', '', s2)
        s2 = s2.replace("'", "")
        return s2
    try:
        t1 = Tree(remove_branch_distances(pred_tree_str))
        t2 = Tree(remove_branch_distances(ref_tree_str))
        r = t1.compare(t2, unrooted=True)
        if isinstance(r["norm_rf"], str):
            return None
        return {"rf": int(r["rf"]), "max_rf": int(r["max_rf"]), "norm_rf": r["norm_rf"]}
    except:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-pickle", default="virus_data/vogdb_treefam_v2.pickle")
    parser.add_argument("--faa-dir", default="virus_data/faa")
    parser.add_argument("--output-csv", default="eval_preds/virus_esm2_vs_fasttree.csv")
    parser.add_argument("--checkpoint", default="eval_preds/virus_esm2_checkpoint.pickle")
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-families", type=int, default=None)
    args = parser.parse_args()

    ref_path = os.path.join(SCRIPT_DIR, args.ref_pickle)
    faa_dir = os.path.join(SCRIPT_DIR, args.faa_dir)
    output_dir = os.path.dirname(os.path.join(SCRIPT_DIR, args.output_csv))
    os.makedirs(output_dir, exist_ok=True)

    print("Loading reference trees...")
    ref_data = pickle.load(open(ref_path, "rb"))
    print(f"  {len(ref_data)} families")

    if args.max_families:
        ref_data = dict(list(ref_data.items())[:args.max_families])
        print(f"  Limited to {len(ref_data)} families")

    print(f"Loading ESM2 model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model = model.to(DEVICE)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params/1e6:.0f}M")

    results = {}
    families_done = set()
    if os.path.exists(os.path.join(SCRIPT_DIR, args.checkpoint)):
        results = pickle.load(open(os.path.join(SCRIPT_DIR, args.checkpoint), "rb"))
        families_done = set(results.keys())
        print(f"  Loaded checkpoint with {len(families_done)} families")

    norm_rfs = []
    csv_rows = [["dataset", "model", "rf", "max_rf", "norm_rf"]]
    skipped = 0
    t_start = time.time()

    for i, (vid, ref_entry) in enumerate(sorted(ref_data.items())):
        if vid in families_done:
            metric = results.get(vid)
            if metric:
                norm_rfs.append(metric["norm_rf"])
                csv_rows.append([vid, "ESM2-650M", str(metric["rf"]), str(metric["max_rf"]), f"{metric['norm_rf']:.4f}"])
            else:
                skipped += 1
            continue

        faa_path = os.path.join(faa_dir, f"{vid}.faa")
        if not os.path.exists(faa_path):
            skipped += 1
            continue

        seqs = {}
        for record in SeqIO.parse(faa_path, "fasta"):
            seqs[record.id] = str(record.seq)

        seq_names = sorted(seqs.keys())
        if len(seq_names) < 4:
            skipped += 1
            continue

        seq_list = [seqs[n] for n in seq_names]
        ref_tree_str = ref_entry["tree_newick"]

        try:
            batch_embeds = []
            for b in range(0, len(seq_list), args.batch_size):
                batch = seq_list[b:b+args.batch_size]
                embeds = encode_sequences(model, tokenizer, batch)
                batch_embeds.append(embeds)
            all_embeds = torch.cat(batch_embeds, dim=0)

            n = len(seq_names)
            dm_matrix = [[0.0] * n for _ in range(n)]
            for i2 in range(n):
                for j in range(i2 + 1, n):
                    d = torch.cdist(all_embeds[i2:i2+1], all_embeds[j:j+1]).item()
                    dm_matrix[i2][j] = d
                    dm_matrix[j][i2] = d

            dm = DistanceMatrix(dm_matrix, seq_names)
            pred_tree = nj(dm)
            pred_tree_str = pred_tree.__str__().replace(" ", "")

            metric = compute_normrf(pred_tree_str, ref_tree_str)
        except Exception as e:
            print(f"  WARNING: {vid} failed: {e}")
            metric = None

        if metric is None:
            skipped += 1
            results[vid] = None
        else:
            norm_rfs.append(metric["norm_rf"])
            csv_rows.append([vid, "ESM2-650M", str(metric["rf"]), str(metric["max_rf"]), f"{metric['norm_rf']:.4f}"])
            results[vid] = metric

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed * 3600
            avg = sum(norm_rfs) / len(norm_rfs) if norm_rfs else 0
            print(f"  [{i+1}/{len(ref_data)}] {elapsed:.0f}s elapsed, "
                  f"{rate:.0f} fams/hr, avg_normRF={avg:.6f}, skipped={skipped}")
            pickle.dump(results, open(os.path.join(SCRIPT_DIR, args.checkpoint), "wb"))

    with open(os.path.join(SCRIPT_DIR, args.output_csv), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"\nCSV saved: {args.output_csv} ({len(csv_rows)-1} rows)")

    pickle.dump(results, open(os.path.join(SCRIPT_DIR, args.checkpoint), "wb"))
    print(f"Checkpoint saved: {args.checkpoint}")

    n = len(norm_rfs)
    sorted_rfs = sorted(norm_rfs)
    print(f"\n{'='*50}")
    print(f"  ESM2-650M + NJ on VOGDB")
    print(f"{'='*50}")
    print(f"  Families evaluated: {n}")
    print(f"  Skipped:            {skipped}")
    if n > 0:
        print(f"  Average normRF:     {sum(norm_rfs)/n:.6f}")
        print(f"  Median normRF:      {sorted_rfs[n//2]:.6f}")
        print(f"  Perfect (normRF=0): {sum(1 for x in norm_rfs if x==0)} ({sum(1 for x in norm_rfs if x==0)/n*100:.1f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()