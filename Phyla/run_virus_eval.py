"""
Virus phylogeny evaluation script using PHYLA model.
Reuses the paper's tree_reconstruction_benchmark with VOGDB data.
"""
import sys
import os
import argparse
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "phyla"))

import torch
from phyla import phyla
from phyla.utils.eval_configs import Config, Mamba_ModelConfig
from eval.evo_reasoning_eval import tree_reconstruction_benchmark


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


def run_virus_eval(model, pickle_path, output_csv, device="cuda:0", max_families=0):
    data = pickle.load(open(pickle_path, "rb"))
    families = list(data.keys())
    if max_families > 0:
        families = families[:max_families]
    data = {k: data[k] for k in families}

    print(f"Evaluating {len(families)} virus families")
    print(f"Output: {output_csv}")

    results = tree_reconstruction_benchmark(
        {"Phyla-beta": {"model": model, "alphabet_tokenizer": None}},
        [0, len(families)],
        output_csv,
        "treefam",
        dictionary_data=data,
        device=device
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virus PHYLA Evaluation")
    parser.add_argument("--pickle", type=str,
                        default="virus_data/vogdb_treefam_v2.pickle",
                        help="Path to VOGDB pickle")
    parser.add_argument("--output-csv", type=str,
                        default="eval_preds/virus_results.csv",
                        help="Output CSV file")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-families", type=int, default=0,
                        help="Limit number of families (0=all)")
    args = parser.parse_args()

    print(f"Loading PHYLA-beta model on {args.device}...")
    model = create_model(device=args.device)
    print("Model loaded successfully!")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    run_virus_eval(
        model,
        args.pickle,
        args.output_csv,
        device=args.device,
        max_families=args.max_families
    )
