import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phyla.utils.eval_configs import Config, Mamba_ModelConfig
from phyla import phyla


def run_inference(fasta_file, checkpoint_path, device="cuda:0", output_file=None):
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

    model = phyla(config, device=device).load(checkpoint_path)
    model.eval()

    encoded_aa, cls_token_mask, sequence_mask, sequence_names = model.encode_fasta(fasta_file)
    print(f"Loaded {len(sequence_names)} sequences from {fasta_file}")

    with torch.no_grad():
        preds = model(
            encoded_aa,
            sequence_mask,
            cls_token_mask,
            logits=False
        )

    tree = model.reconstruct_tree(preds, sequence_names)
    tree_str = str(tree)

    if output_file:
        with open(output_file, "w") as f:
            f.write(tree_str + "\n")
        print(f"Tree saved to {output_file}")

    print("Reconstructed tree (Newick format):")
    print(tree_str)
    return tree_str


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_inference.py <fasta_file> <checkpoint_path> [device] [output_file]")
        print("Example: python run_inference.py phyla/data/40seqs.fasta weights/11564369 cuda:0 output.nwk")
        sys.exit(1)

    fasta_file = sys.argv[1]
    checkpoint_path = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
    output_file = sys.argv[4] if len(sys.argv) > 4 else None

    run_inference(fasta_file, checkpoint_path, device, output_file)
