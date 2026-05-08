import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from Bio import SeqIO

from src.models.route_a.viral_phylogpn import ViralPhyloGPN
from src.models.route_a.gtr_module import GTRModel
from src.models.route_a.felsenstein import FelsensteinPruning
from src.models.tree.nj_builder import nj_from_distance_matrix
from src.models.tree.tree_metrics import TreeMetrics
from src.models.distance.k2p_baseline import K2PDistance, compute_k2p_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="ViroPhylo Route A Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/route_a")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class PhyloGPNTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split="train", window_size=241, max_gap_frac=0.5):
        self.data_dir = data_dir
        self.window_size = window_size
        self.max_gap_frac = max_gap_frac
        self.samples = []
        self._load(split)

    def _load(self, split):
        aln_dir = os.path.join(self.data_dir, "alignments", split)
        tree_dir = os.path.join(self.data_dir, "trees", split)

        if not os.path.exists(aln_dir):
            return

        for f in sorted(os.listdir(aln_dir)):
            if not f.endswith(('.fasta', '.fa', '.fna')):
                continue
            aln_path = os.path.join(aln_dir, f)
            base = f.rsplit('.', 1)[0]
            tree_path = os.path.join(tree_dir, base + '.nwk') if os.path.exists(tree_dir) else None

            sequences, names = [], []
            for record in SeqIO.parse(aln_path, "fasta"):
                sequences.append(str(record.seq).upper())
                names.append(record.id)

            if len(sequences) < 4:
                continue

            ref_tree = None
            if tree_path and os.path.exists(tree_path):
                with open(tree_path) as tf:
                    ref_tree = tf.read().strip()

            self.samples.append({
                "sequences": sequences,
                "names": names,
                "ref_tree": ref_tree,
                "source": f,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        seqs = sample["sequences"]
        n = len(seqs)
        L = len(seqs[0])

        encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, '-': 4, 'N': 4}
        encoded = torch.zeros(n, L, dtype=torch.long)
        for i, seq in enumerate(seqs):
            for j, c in enumerate(seq):
                encoded[i, j] = encoding.get(c, 4)

        gap_frac = (encoded == 4).float().mean(dim=0)
        site_weights = torch.clamp(1.0 - gap_frac / self.max_gap_frac, min=0.05, max=1.0)

        return {
            "encoded_alignment": encoded,
            "site_weights": site_weights,
            "names": sample["names"],
            "ref_tree": sample["ref_tree"],
            "source": sample["source"],
        }


def compute_phylogpn_loss(model, encoded_alignment, site_weights, gtr_model, felsenstein,
                          tree_structure=None):
    n_seqs, L = encoded_alignment.shape
    n_bases = 5
    onehot = F.one_hot(encoded_alignment, num_classes=n_bases).float()

    raw_rates, raw_freq, raw_alpha, site_embeddings = model(onehot)

    rates = F.softplus(raw_rates)
    rates = rates * 6.0 / rates.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    frequencies = F.softmax(raw_freq, dim=-1)
    alpha = F.softplus(raw_alpha).clamp(min=0.1, max=10.0)

    avg_rates = rates.mean(dim=(0, 1))
    avg_freq = frequencies.mean(dim=(0, 1))
    avg_alpha = alpha.mean()

    Q = gtr_model.compute_Q_matrix(avg_rates, avg_freq)

    if tree_structure is not None:
        log_likelihood = felsenstein(
            encoded_alignment, tree_structure, Q, avg_freq, avg_alpha
        )
        loss = -log_likelihood
    else:
        loss = torch.tensor(0.0, device=encoded_alignment.device)

    entropy = -(frequencies * torch.log(frequencies + 1e-10)).sum(dim=-1)
    site_weights_expanded = site_weights.unsqueeze(0).expand_as(entropy)
    weighted_entropy = (entropy * site_weights_expanded).mean()

    loss = loss + 0.01 * weighted_entropy

    return loss


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    model = ViralPhyloGPN(
        window_size=config.get("window_size", 241),
        d_model=config.get("d_model", 960),
        d_inner=config.get("d_inner", 960),
        n_blocks=config.get("n_blocks", 40),
        kernel_size=config.get("kernel_size", 9),
    ).to(device)

    gtr_model = GTRModel().to(device)
    felsenstein = FelsensteinPruning(n_bases=4, n_gamma_categories=4).to(device)

    if args.resume:
        state_dict = torch.load(args.resume, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"ViralPhyloGPN parameters: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=config.get("learning_rate", 5e-4),
                      weight_decay=config.get("weight_decay", 0.01))
    n_epochs = config.get("epochs", 30)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    train_dataset = PhyloGPNTrainingDataset(args.data_dir, split="train",
                                             window_size=config.get("window_size", 241))
    val_dataset = PhyloGPNTrainingDataset(args.data_dir, split="val",
                                           window_size=config.get("window_size", 241))

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    best_loss = float('inf')
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        indices = np.random.permutation(len(train_dataset))
        for idx in tqdm(indices, desc=f"Epoch {epoch}", leave=False):
            sample = train_dataset[idx]
            encoded = sample["encoded_alignment"].to(device)
            site_w = sample["site_weights"].to(device)

            with torch.amp.autocast("cuda", enabled=args.bf16 and device.type == "cuda"):
                loss = compute_phylogpn_loss(
                    model, encoded, site_w, gtr_model, felsenstein,
                )

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> Saved best model (loss={avg_loss:.4f})")

        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt"))

    print(f"Training complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
