"""
Evaluation wrapper script for PHYLA model.
Run from the Phyla directory:
  python run_eval.py --dataset treefam --device cuda:0
"""
import sys
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "phyla"))

import torch
import pickle

from phyla import phyla
from phyla.utils.eval_configs import Config, Mamba_ModelConfig, EvalConfig, DatasetConfig, TrainerConfig


def create_model(device="cuda:0"):
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

    config.trainer.checkpoint_path = os.path.join(SCRIPT_DIR, "weights", "11564369")
    config.eval.device = device

    model = phyla(config, device=device).load(config.trainer.checkpoint_path)
    model.eval()
    return model


def run_treefam_eval(model, device="cuda:0"):
    from eval.evo_reasoning_eval import tree_reconstruction_benchmark

    treefam_path = os.path.join(SCRIPT_DIR, "treefam.pickle")
    data = pickle.load(open(treefam_path, "rb"))
    num_datasets = [0, len(data)]
    output_file = os.path.join(SCRIPT_DIR, "eval_preds", "treefam_results.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results = tree_reconstruction_benchmark(
        {"Phyla-beta": {"model": model, "alphabet_tokenizer": None}},
        num_datasets,
        output_file,
        "treefam",
        dictionary_data=data,
        device=device
    )
    return results


def run_treebase_eval(model, device="cuda:0"):
    from eval.evo_reasoning_eval import tree_reconstruction_benchmark

    output_file = os.path.join(SCRIPT_DIR, "eval_preds", "treebase_results.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results = tree_reconstruction_benchmark(
        {"Phyla-beta": {"model": model, "alphabet_tokenizer": None}},
        [0, 10],
        output_file,
        "treebase",
        device=device
    )
    return results


def run_single_inference(model, fasta_path, output_path=None):
    encoded_aa, cls_token_mask, sequence_mask, sequence_names = model.encode_fasta(fasta_path)
    print(f"Loaded {len(sequence_names)} sequences from {fasta_path}")

    with torch.no_grad():
        preds = model(encoded_aa, sequence_mask, cls_token_mask, logits=False)

    tree = model.reconstruct_tree(preds, sequence_names)
    tree_str = str(tree)

    if output_path:
        with open(output_path, "w") as f:
            f.write(tree_str + "\n")
        print(f"Tree saved to {output_path}")

    print("Reconstructed tree (Newick format):")
    print(tree_str)
    return tree_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHYLA Evaluation Runner")
    parser.add_argument("--dataset", type=str, default="treefam",
                        choices=["treefam", "treebase", "gtdb", "proteingym", "inference"],
                        help="Dataset to evaluate on")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device")
    parser.add_argument("--fasta", type=str, default=None,
                        help="FASTA file for single inference")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for Newick tree")
    args = parser.parse_args()

    print(f"Loading PHYLA-beta model on {args.device}...")
    model = create_model(device=args.device)
    print("Model loaded successfully!")

    if args.dataset == "inference" or args.fasta is not None:
        fasta = args.fasta or os.path.join(SCRIPT_DIR, "phyla", "data", "40seqs.fasta")
        run_single_inference(model, fasta, args.output)
    elif args.dataset == "treefam":
        run_treefam_eval(model, device=args.device)
    elif args.dataset == "treebase":
        run_treebase_eval(model, device=args.device)
    else:
        print(f"Dataset {args.dataset} evaluation not yet implemented in this wrapper.")
        print("For full evaluation, use: python -m eval.evo_reasoning_eval configs/eval_config.yaml")
