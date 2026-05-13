#!/usr/bin/env python3
"""
VOGDB to TreeFam-format pipeline for PHYLA virus phylogeny evaluation.

Pipeline:
  1. Filter VFAMs with >= min_seqs sequences
  2. Run FastTree on each VFAM's MSA to build reference tree
  3. Package sequences + reference tree into TreeFam-compatible pickle
"""
import os
import sys
import glob
import pickle
import argparse
import subprocess
import multiprocessing as mp
from Bio import SeqIO


def load_faa_sequences(faa_path):
    seqs = {}
    for record in SeqIO.parse(faa_path, "fasta"):
        seqs[record.id] = str(record.seq)
    return seqs


def build_reference_tree(msa_path, fasttree_bin, output_dir, vfam_id):
    nwk_path = os.path.join(output_dir, f"{vfam_id}.nwk")
    if os.path.exists(nwk_path) and os.path.getsize(nwk_path) > 10:
        return nwk_path

    try:
        result = subprocess.run(
            [fasttree_bin, "-quiet", msa_path],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and len(result.stdout.strip()) > 10:
            with open(nwk_path, "w") as f:
                f.write(result.stdout.strip() + "\n")
            return nwk_path
        else:
            return None
    except Exception:
        return None


def process_one_vfam(args):
    vfam_id, faa_path, msa_path, fasttree_bin, ref_tree_dir, min_seqs = args

    faa_file = os.path.join(faa_path, f"{vfam_id}.faa")
    msa_file = os.path.join(msa_path, f"{vfam_id}.msa")

    if not os.path.exists(faa_file) or not os.path.exists(msa_file):
        return None

    seqs = load_faa_sequences(faa_file)
    if len(seqs) < min_seqs:
        return None

    nwk_path = build_reference_tree(msa_file, fasttree_bin, ref_tree_dir, vfam_id)
    if nwk_path is None:
        return None

    with open(nwk_path) as f:
        tree_newick = f.read().strip()

    if not tree_newick or len(tree_newick) < 5:
        return None

    return {vfam_id: {"sequences": seqs, "tree_newick": tree_newick}}


def main():
    parser = argparse.ArgumentParser(description="VOGDB to TreeFam pipeline")
    parser.add_argument("--faa-dir", default="virus_data/faa",
                        help="Path to FAA directory")
    parser.add_argument("--msa-dir", default="virus_data/msa",
                        help="Path to MSA directory")
    parser.add_argument("--ref-tree-dir", default="virus_data/ref_trees",
                        help="Directory for reference trees")
    parser.add_argument("--output", default="virus_data/vogdb_treefam.pickle",
                        help="Output pickle file")
    parser.add_argument("--fasttree", default="fasttree",
                        help="Path to FastTree binary")
    parser.add_argument("--min-seqs", type=int, default=4,
                        help="Minimum sequences per VFAM")
    parser.add_argument("--max-families", type=int, default=0,
                        help="Max families to process (0=all)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process only 10 families")
    args = parser.parse_args()

    os.makedirs(args.ref_tree_dir, exist_ok=True)

    faa_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.faa_dir)
    msa_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.msa_dir)
    ref_tree_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.ref_tree_dir)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    fasttree_bin = args.fasttree
    if not os.path.isabs(fasttree_bin):
        fasttree_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), fasttree_bin)

    faa_files = sorted(glob.glob(os.path.join(faa_dir, "VFAM*.faa")))
    vfam_ids = [os.path.basename(f).replace(".faa", "") for f in faa_files]

    if args.test:
        vfam_ids = vfam_ids[:10]
        args.min_seqs = 4
        args.max_families = 10
    elif args.max_families > 0:
        vfam_ids = vfam_ids[:args.max_families]

    print(f"Processing {len(vfam_ids)} VFAMs (min_seqs={args.min_seqs})")
    print(f"FAA dir: {faa_dir}")
    print(f"MSA dir: {msa_dir}")
    print(f"Ref tree dir: {ref_tree_dir}")
    print(f"FastTree: {fasttree_bin}")

    task_args = [
        (vid, faa_dir, msa_dir, fasttree_bin, ref_tree_dir, args.min_seqs)
        for vid in vfam_ids
    ]

    data_dict = {}
    success = 0
    skipped_too_few = 0
    failed = 0

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(process_one_vfam, task_args)):
                if result is not None:
                    data_dict.update(result)
                    success += 1
                else:
                    failed += 1
                if (i + 1) % 100 == 0:
                    print(f"  Progress: {i+1}/{len(task_args)}, success={success}, failed={failed}")
    else:
        for i, targs in enumerate(task_args):
            result = process_one_vfam(targs)
            if result is not None:
                data_dict.update(result)
                success += 1
            else:
                failed += 1
            if (i + 1) % 500 == 0:
                print(f"  Progress: {i+1}/{len(task_args)}, success={success}, failed={failed}")

    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"\nDone! Output: {output_path}")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"  Total families in pickle: {len(data_dict)}")
    print(f"  File size: {os.path.getsize(output_path)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
