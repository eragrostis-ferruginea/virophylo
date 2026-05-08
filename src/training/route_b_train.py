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

from src.models.route_b.phyla_viral import PHYLAViralModel
from src.training.losses import QuartetLoss, PhyloLikelihoodLoss
from src.data.viral_dataset import CompositionFeatureExtractor
from src.models.distance.k2p_baseline import K2PDistance
from src.models.tree.nj_builder import nj_from_distance_matrix
from src.models.tree.tree_metrics import TreeMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="ViroPhylo Route B Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/route_b")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        return rank, world_size
    return 0, 1


class RouteBDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split="train", max_seqs=64, max_seq_length=2048,
                 n_quartets=100):
        self.data_dir = data_dir
        self.split = split
        self.max_seqs = max_seqs
        self.max_seq_length = max_seq_length
        self.n_quartets = n_quartets
        self.k2p = K2PDistance()
        self.samples = []
        self._load()

    def _load(self):
        aln_dir = os.path.join(self.data_dir, "alignments", self.split)
        tree_dir = os.path.join(self.data_dir, "trees", self.split)

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
                "sequences": sequences[:self.max_seqs],
                "names": names[:self.max_seqs],
                "ref_tree": ref_tree,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        seqs = sample["sequences"]
        names = sample["names"]
        n = len(seqs)

        quartet_indices = []
        if n >= 4:
            rng = np.random.RandomState(idx)
            for _ in range(self.n_quartets):
                q = tuple(sorted(rng.choice(n, 4, replace=False).tolist()))
                quartet_indices.append(q)

        return {
            "sequences": seqs,
            "names": names,
            "quartet_indices": quartet_indices,
            "ref_tree": sample["ref_tree"],
        }


def train_one_epoch(model, dataset, optimizer, epoch, device, comp_extractor,
                    n_epochs, bf16=False):
    model.train()
    total_loss = 0.0
    n_batches = 0

    quartet_loss_fn = QuartetLoss()
    likelihood_loss_fn = PhyloLikelihoodLoss()

    phase = epoch / n_epochs
    alpha = max(0.3, 1.0 - phase)
    beta = min(0.7, phase)

    indices = np.random.permutation(len(dataset))

    for idx in tqdm(indices, desc=f"Epoch {epoch}", leave=False):
        sample = dataset[idx]
        sequences = sample["sequences"]
        names = sample["names"]
        quartet_indices = sample["quartet_indices"]
        n_seqs = len(sequences)

        if n_seqs < 4:
            continue

        comp_features = comp_extractor.extract_batch(sequences)
        comp_tensor = torch.from_numpy(comp_features).to(device)

        with torch.amp.autocast("cuda", enabled=bf16 and device.type == "cuda"):
            dist_matrix, embeddings, gtr_ll = model(
                sequences, composition_features=comp_tensor,
            )

            loss_q = quartet_loss_fn(dist_matrix, quartet_indices=quartet_indices)

            loss_l = torch.tensor(0.0, device=device)
            if gtr_ll is not None:
                loss_l = likelihood_loss_fn(gtr_ll)

            loss = alpha * loss_q + beta * loss_l

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, dataset, device, comp_extractor, metrics_obj):
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            sequences = sample["sequences"]
            names = sample["names"]
            ref_tree = sample.get("ref_tree")

            if ref_tree is None or len(sequences) < 4:
                continue

            comp_features = comp_extractor.extract_batch(sequences)
            comp_tensor = torch.from_numpy(comp_features).to(device)

            try:
                pred_tree, _ = model.predict_tree(sequences, names, composition_features=comp_tensor)
                m = metrics_obj.evaluate(pred_tree, ref_tree, dataset_name=f"sample_{idx}")
                all_metrics.append(m)
            except Exception as e:
                print(f"Eval error: {e}")

    if not all_metrics:
        return {"nrf": None, "qa": None}

    avg = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if m[key] is not None]
        avg[key] = np.mean(vals) if vals else None
    return avg


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    model = PHYLAViralModel(
        d_model=config.get("d_model", 768),
        n_mamba_layers=config.get("n_mamba_layers", 12),
        d_state=config.get("d_state", 16),
        d_conv=config.get("d_conv", 4),
        expand=config.get("expand", 2),
        n_tree_heads=config.get("n_tree_heads", 8),
        use_calibration=config.get("use_calibration", True),
        use_gtr_head=config.get("use_gtr_head", True),
    ).to(device)

    if args.resume:
        state_dict = torch.load(args.resume, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PHYLAViral: {trainable:,} trainable / {total_params:,} total")

    optimizer = AdamW(model.parameters(), lr=config.get("learning_rate", 3e-4),
                      weight_decay=config.get("weight_decay", 0.01))
    n_epochs = config.get("epochs", 30)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    comp_extractor = CompositionFeatureExtractor(k=4)
    metrics_obj = TreeMetrics()

    train_dataset = RouteBDataset(args.data_dir, split="train", max_seqs=config.get("max_seqs", 64))
    val_dataset = RouteBDataset(args.data_dir, split="val", max_seqs=config.get("max_seqs", 64))

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    best_nrf = float('inf')
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(
            model, train_dataset, optimizer, epoch, device,
            comp_extractor, n_epochs, bf16=args.bf16,
        )
        scheduler.step()

        val_metrics = evaluate(model, val_dataset, device, comp_extractor, metrics_obj)
        nrf = val_metrics.get("nrf", float('inf'))
        qa = val_metrics.get("qa", 0.0)
        print(f"Epoch {epoch}: loss={train_loss:.4f}, nRF={nrf:.4f}, QA={qa:.4f}")

        if nrf < best_nrf:
            best_nrf = nrf
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> Best model saved (nRF={nrf:.4f})")

        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt"))

    print(f"Training complete. Best nRF: {best_nrf:.4f}")


if __name__ == "__main__":
    main()
