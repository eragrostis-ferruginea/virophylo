"""
PHYLA virus tree prediction: runs independently of FastTree reference tree building.
Saves PHYLA-predicted trees to a separate pickle for later comparison.
"""
import sys
import os
import glob
import pickle
import argparse
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "phyla"))

import torch
from Bio import SeqIO
from phyla import phyla
from phyla.utils.eval_configs import Config, Mamba_ModelConfig


def encode_fasta_from_memory(sequences, sequence_names):
    from phyla.dataset.data import Arbitrary_Sequence_Dataset
    dataset = Arbitrary_Sequence_Dataset()
    batch, names = dataset.encode_sequences(sequences, sequence_names)
    return batch, names


def load_faa_sequences(faa_path):
    seqs = {}
    for record in SeqIO.parse(faa_path, "fasta"):
        seqs[record.id] = str(record.seq)
    return seqs


def process_vfam_batch(vfam_ids, faa_dir, model, device, min_seqs=4):
    results = {}
    for vfam_id in vfam_ids:
        faa_path = os.path.join(faa_dir, f"{vfam_id}.faa")
        if not os.path.exists(faa_path):
            continue

        seqs = load_faa_sequences(faa_path)
        if len(seqs) < min_seqs:
            continue

        seq_names = list(seqs.keys())
        seq_vals = [seqs[n] for n in seq_names]

        try:
            batch, names = encode_fasta_from_memory(seq_vals, seq_names)

            with torch.no_grad():
                preds = model(
                    batch['encoded_sequences'].to(device),
                    batch['sequence_mask'].to(device),
                    batch['cls_positions'].bool().to(device)
                )

            tree = model.reconstruct_tree(preds, names)
            results[vfam_id] = {
                "pred_tree_newick": str(tree),
                "num_seqs": len(seqs),
                "seq_names": names
            }
        except Exception as e:
            print(f"  ERROR: {vfam_id}: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(description="PHYLA virus tree prediction")
    parser.add_argument("--faa-dir", default="virus_data/faa")
    parser.add_argument("--output", default="virus_data/phyla_predictions.pickle")
    parser.add_argument("--checkpoint", default="weights/11564369")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-seqs", type=int, default=4)
    parser.add_argument("--max-families", type=int, default=0)
    args = parser.parse_args()

    faa_dir = os.path.join(SCRIPT_DIR, args.faa_dir)
    os.makedirs(os.path.dirname(os.path.join(SCRIPT_DIR, args.output)), exist_ok=True)

    print("Loading PHYLA-beta model...")
    config = Config()
    config.model = Mamba_ModelConfig()
    config.model.d_model = 256
    config.model.n_layer = 16
    config.model.vocab_size = 24
    config.model.num_blocks = 3
    config.model.model_name = 'Phyla-beta'
    config.model.bidirectional = True
    config.model.bidirectional_strategy = "add"
    config.model.bidirectional_weight_tie = True

    model = phyla(config, device=args.device).load(
        os.path.join(SCRIPT_DIR, args.checkpoint)
    )
    model.eval()
    print("Model loaded on", args.device)

    faa_files = sorted(glob.glob(os.path.join(faa_dir, "VFAM*.faa")))
    vfam_ids = [os.path.basename(f).replace(".faa", "") for f in faa_files]

    # Quick count to filter
    valid_ids = []
    for f, vid in zip(faa_files, vfam_ids):
        n = sum(1 for l in open(f) if l.startswith(">"))
        if n >= args.min_seqs:
            valid_ids.append(vid)

    if args.max_families > 0:
        valid_ids = valid_ids[:args.max_families]

    print(f"Total VFAMs: {len(vfam_ids)}, pass min_seqs={args.min_seqs}: {len(valid_ids)}")

    all_predictions = {}
    batch_size = 500
    total_start = time.time()

    for i in range(0, len(valid_ids), batch_size):
        batch = valid_ids[i:i+batch_size]
        batch_start = time.time()
        results = process_vfam_batch(batch, faa_dir, model, args.device, args.min_seqs)
        all_predictions.update(results)
        elapsed = time.time() - batch_start
        print(f"  Batch {i//batch_size + 1}: {len(batch)} families, "
              f"{len(results)} success, {elapsed:.1f}s "
              f"(avg {elapsed/max(len(results),1):.2f}s/family)")

        if (i + batch_size) % 2000 == 0 or i + batch_size >= len(valid_ids):
            temp_out = args.output.replace(".pickle", f"_checkpoint.pickle")
            with open(os.path.join(SCRIPT_DIR, temp_out), "wb") as f:
                pickle.dump(all_predictions, f)
            print(f"  Checkpoint saved: {len(all_predictions)} families so far")

    total_time = time.time() - total_start
    print(f"\nTotal: {len(all_predictions)} families in {total_time:.1f}s")

    output_path = os.path.join(SCRIPT_DIR, args.output)
    with open(output_path, "wb") as f:
        pickle.dump(all_predictions, f)
    print(f"Saved to {output_path}")
    print(f"File size: {os.path.getsize(output_path)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()