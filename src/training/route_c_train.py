import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

from src.models.route_c.route_c_model import RouteCModel
from src.models.backbone.dnabert2_wrapper import DNABERT2Wrapper, NTWrapper
from src.training.losses import TripleLoss, LossWeightScheduler
from src.data.viral_dataset import CompositionFeatureExtractor
from src.data.phylo_dataset import PhyloTrainingDataset
from src.models.distance.k2p_baseline import K2PDistance
from src.models.tree.nj_builder import nj_from_distance_matrix
from src.models.tree.tree_metrics import TreeMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="ViroPhylo Route C Training")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--output_dir", type=str, default="outputs/route_c", help="Output directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        return rank, world_size
    return 0, 1


def build_model(config, device):
    backbone_name = config.get("backbone", "dnabert2")
    lora_rank = config.get("lora_rank", 16)
    lora_alpha = config.get("lora_alpha", 32)
    lora_dropout = config.get("lora_dropout", 0.1)

    if backbone_name == "dnabert2":
        backbone = DNABERT2Wrapper(
            lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        )
    elif backbone_name == "nt500m":
        backbone = NTWrapper(
            lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    model = RouteCModel(
        backbone=backbone,
        embed_dim=backbone.embed_dim,
        n_composition_features=config.get("n_composition_features", 20),
        use_calibration=config.get("use_calibration", True),
        use_hybrid_distance=config.get("use_hybrid_distance", True),
        distance_head_type=config.get("distance_head_type", "mlp"),
        distance_hidden_dim=config.get("distance_hidden_dim", 256),
    )

    return model.to(device)


def train_one_epoch(model, dataset, optimizer, loss_fn, scheduler, epoch, device,
                    comp_extractor, k2p, config, bf16=False):
    model.train()
    total_loss = 0.0
    n_batches = 0

    alpha, beta, gamma = scheduler.get_weights(epoch)
    loss_fn.set_weights(alpha, beta, gamma)

    indices = np.random.permutation(len(dataset))
    n_quartets = config.get("n_quartets_per_sample", 100)

    for idx in tqdm(indices, desc=f"Epoch {epoch}", leave=False):
        sample = dataset[idx]
        sequences = sample["sequences"]
        names = sample["names"]
        n_seqs = len(sequences)

        if n_seqs < 4:
            continue

        comp_features = comp_extractor.extract_batch(sequences)
        comp_tensor = torch.from_numpy(comp_features).to(device)

        encoded_seqs = sample.get("encoded_seqs")
        if encoded_seqs is not None:
            encoded_seqs = encoded_seqs.to(device)

        target_dist = sample.get("target_distance")
        if target_dist is not None:
            target_dist = target_dist.to(device)

        quartet_indices = sample.get("quartet_indices", [])
        if not quartet_indices:
            rng = np.random.RandomState(epoch * 10000 + idx)
            quartet_indices = []
            for _ in range(n_quartets):
                q = tuple(sorted(rng.choice(n_seqs, 4, replace=False).tolist()))
                quartet_indices.append(q)

        with torch.amp.autocast("cuda", enabled=bf16 and device.type == "cuda"):
            dist_matrix, embeddings = model(
                sequences, composition_features=comp_tensor,
                encoded_seqs=encoded_seqs, return_embeddings=True,
            )

            k2p_dist = sample.get("k2p_distance")
            if k2p_dist is not None:
                k2p_dist = k2p_dist.to(device)

            dist_mask = None
            if target_dist is not None:
                diag_mask = ~torch.eye(n_seqs, dtype=torch.bool, device=device)
                triu_mask = torch.triu(diag_mask, diagonal=1)
                dist_mask = triu_mask.float()

            loss = loss_fn(
                dist_matrix=dist_matrix,
                target_dist=target_dist,
                quartet_indices=quartet_indices,
                dist_mask=dist_mask,
            )

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, dataset, device, comp_extractor, k2p, metrics_obj):
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
            encoded_seqs = sample.get("encoded_seqs")
            if encoded_seqs is not None:
                encoded_seqs = encoded_seqs.to(device)

            try:
                pred_tree, _ = model.predict_tree(
                    sequences, names,
                    composition_features=comp_tensor,
                    encoded_seqs=encoded_seqs,
                )
                m = metrics_obj.evaluate(pred_tree, ref_tree, dataset_name=f"sample_{idx}")
                all_metrics.append(m)
            except Exception as e:
                print(f"Eval error sample {idx}: {e}")

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

    model = build_model(config, device)
    if args.resume:
        state_dict = torch.load(args.resume, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.1f}%)")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 0.01),
    )

    n_epochs = config.get("epochs", 20)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    loss_fn = TripleLoss()
    loss_scheduler = LossWeightScheduler(n_epochs, schedule_type=config.get("loss_schedule", "phased"))

    comp_extractor = CompositionFeatureExtractor(k=4)
    k2p = K2PDistance()
    metrics_obj = TreeMetrics()

    train_dataset = PhyloTrainingDataset(
        args.data_dir, split="train",
        n_quartets_per_sample=config.get("n_quartets_per_sample", 100),
        max_seqs=config.get("max_seqs_per_batch", 64),
    )
    val_dataset = PhyloTrainingDataset(
        args.data_dir, split="val",
        n_quartets_per_sample=50,
        max_seqs=config.get("max_seqs_per_batch", 64),
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    best_nrf = float('inf')
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(
            model, train_dataset, optimizer, loss_fn, loss_scheduler,
            epoch, device, comp_extractor, k2p, config, bf16=args.bf16,
        )
        scheduler.step()

        val_metrics = evaluate(model, val_dataset, device, comp_extractor, k2p, metrics_obj)

        nrf = val_metrics.get("nrf", float('inf'))
        qa = val_metrics.get("qa", 0.0)
        print(f"Epoch {epoch}: loss={train_loss:.4f}, val_nRF={nrf:.4f}, val_QA={qa:.4f}")

        if nrf < best_nrf:
            best_nrf = nrf
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved best model (nRF={nrf:.4f})")

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
            }, ckpt_path)

    print(f"Training complete. Best nRF: {best_nrf:.4f}")


if __name__ == "__main__":
    main()
