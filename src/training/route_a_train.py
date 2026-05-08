import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.route_a.viral_phylogpn import ViralPhyloGPN
from src.models.route_a.gtr_module import GTRModel
from src.models.route_a.felsenstein import FelsensteinPruning, newick_to_tree_structure
from src.models.distance.k2p_baseline import K2PDistance
from src.models.distance.distance_head import DistanceHead
from src.models.calibration.zca_whitening import EmbeddingCalibration
from src.training.losses import QuartetLoss, DistanceRegressionLoss, PhyloLikelihoodLoss, TripleLoss, LossWeightScheduler
from src.data.phylo_dataset import PhyloTrainingDataset


def compute_phylogpn_loss(model, onehot, alignment, tree_structure, Q_ref, freqs_ref,
                          alpha_ref, felsenstein, dist_head, calibration,
                          target_dist, quartet_indices, quartet_topologies,
                          comp_features, loss_fn, device):
    rates, freqs, alpha, site_emb = model(onehot)

    rates_norm = model.normalize_rates(rates.mean(dim=1))
    freqs_norm = model.normalize_frequencies(freqs.mean(dim=1))
    alpha_norm = F.softplus(alpha.mean(dim=1)).clamp(min=0.1, max=10.0)

    gtr = GTRModel()
    Q = gtr.compute_Q_matrix(rates_norm[0], freqs_norm[0])

    ll = felsenstein(alignment.unsqueeze(0), tree_structure, Q, freqs_norm[0], alpha_norm[0])

    if site_emb.dim() == 3:
        pooled = site_emb.mean(dim=1)
    else:
        pooled = site_emb

    if comp_features is not None:
        calibrated_emb, _ = calibration(pooled, comp_features.to(device))
    else:
        calibrated_emb = pooled

    pred_dist = dist_head.pairwise_distances(calibrated_emb)

    loss = loss_fn(
        pred_dist,
        target_dist=target_dist,
        log_likelihood=ll,
        quartet_indices=quartet_indices,
        quartet_topologies=quartet_topologies,
    )

    entropy = -(freqs_norm * freqs_norm.log().clamp(min=-10)).sum(dim=-1).mean()
    loss = loss + 0.01 * entropy

    return loss, ll, pred_dist


def train_route_a(config):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = ViralPhyloGPN(
        window_size=config.get("window_size", 241),
        d_model=config.get("d_model", 256),
        d_inner=config.get("d_inner", 512),
        n_blocks=config.get("n_blocks", 8),
        kernel_size=config.get("kernel_size", 3),
        n_bases=5,
    ).to(device)

    dist_head = DistanceHead(
        embed_dim=config.get("d_model", 256),
        hidden_dim=config.get("d_model", 256),
    ).to(device)

    calibration = EmbeddingCalibration(
        embed_dim=config.get("d_model", 256),
        n_composition_features=config.get("n_composition_features", 20),
        use_zca=True,
        use_debias=True,
        use_site_weight=True,
    ).to(device)

    felsenstein = FelsensteinPruning(n_bases=4, n_gamma_categories=4)

    dataset = PhyloTrainingDataset(
        data_dir=config["data_dir"],
        split="train",
        n_quartets_per_sample=config.get("n_quartets_per_sample", 100),
        max_seqs=config.get("max_seqs", 64),
        max_seq_length=config.get("max_seq_length", 2048),
    )

    val_dataset = PhyloTrainingDataset(
        data_dir=config["data_dir"],
        split="val",
        n_quartets_per_sample=config.get("n_quartets_per_sample", 50),
        max_seqs=config.get("max_seqs", 64),
        max_seq_length=config.get("max_seq_length", 2048),
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.get("num_workers", 4))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.get("num_workers", 2))

    all_params = list(model.parameters()) + list(dist_head.parameters()) + list(calibration.parameters())
    optimizer = AdamW(all_params, lr=config.get("lr", 1e-4), weight_decay=config.get("weight_decay", 0.01))
    scheduler = CosineAnnealingLR(optimizer, T_max=config.get("epochs", 100))

    loss_scheduler = LossWeightScheduler(
        total_epochs=config.get("epochs", 100),
        schedule_type=config.get("loss_schedule", "phased"),
    )

    k2p = K2PDistance()

    best_val_loss = float('inf')
    output_dir = config.get("output_dir", "outputs/route_a")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(config.get("epochs", 100)):
        alpha_w, beta_w, gamma_w = loss_scheduler.get_weights(epoch)

        model.train()
        dist_head.train()
        calibration.train()

        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            sequences = batch["sequences"]
            names = batch["names"]
            comp_features = batch["composition_features"]
            encoded_seqs = batch["encoded_seqs"].to(device)
            k2p_dist = batch["k2p_distance"].to(device)
            target_dist = batch["target_distance"]
            quartet_indices_raw = batch["quartet_indices"]
            ref_tree = batch.get("ref_tree")

            n_seqs = len(sequences)
            L = encoded_seqs.shape[1]

            onehot = F.one_hot(encoded_seqs, num_classes=5).float()

            tree_structure = None
            if ref_tree is not None and isinstance(ref_tree, list) and ref_tree[0] is not None:
                try:
                    tree_structure = newick_to_tree_structure(ref_tree[0], names)
                except Exception:
                    tree_structure = None

            quartet_indices = []
            quartet_topologies = []
            if isinstance(quartet_indices_raw, list) and len(quartet_indices_raw) > 0:
                for q in quartet_indices_raw:
                    if isinstance(q, (list, tuple)) and len(q) == 4:
                        quartet_indices.append(tuple(q))

            quartet_topologies_raw = batch.get("quartet_topologies", [])
            if isinstance(quartet_topologies_raw, list) and len(quartet_topologies_raw) > 0:
                for t in quartet_topologies_raw:
                    quartet_topologies.append(int(t) if isinstance(t, (int, float)) else 0)
            else:
                quartet_topologies = [0] * len(quartet_indices)

            if target_dist is not None and target_dist.numel() > 0:
                target_dist = target_dist.to(device)
            else:
                target_dist = k2p_dist

            loss_fn = TripleLoss(alpha=alpha_w, beta=beta_w, gamma=gamma_w)

            optimizer.zero_grad()
            loss, ll, pred_dist = compute_phylogpn_loss(
                model, onehot, encoded_seqs, tree_structure, None, None,
                None, felsenstein, dist_head, calibration,
                target_dist, quartet_indices, quartet_topologies,
                comp_features, loss_fn, device,
            )

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}: loss={loss.item():.4f} ll={ll.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

        if (epoch + 1) % config.get("eval_every", 5) == 0:
            model.eval()
            dist_head.eval()
            calibration.eval()

            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in val_dataloader:
                    sequences = batch["sequences"]
                    names = batch["names"]
                    comp_features = batch["composition_features"]
                    encoded_seqs = batch["encoded_seqs"].to(device)
                    k2p_dist = batch["k2p_distance"].to(device)
                    target_dist = batch["target_distance"]
                    quartet_indices_raw = batch["quartet_indices"]
                    ref_tree = batch.get("ref_tree")

                    onehot = F.one_hot(encoded_seqs, num_classes=5).float()

                    tree_structure = None
                    if ref_tree is not None and isinstance(ref_tree, list) and ref_tree[0] is not None:
                        try:
                            tree_structure = newick_to_tree_structure(ref_tree[0], names)
                        except Exception:
                            tree_structure = None

                    quartet_indices = []
                    quartet_topologies = []
                    if isinstance(quartet_indices_raw, list) and len(quartet_indices_raw) > 0:
                        for q in quartet_indices_raw:
                            if isinstance(q, (list, tuple)) and len(q) == 4:
                                quartet_indices.append(tuple(q))

                    quartet_topologies_raw = batch.get("quartet_topologies", [])
                    if isinstance(quartet_topologies_raw, list) and len(quartet_topologies_raw) > 0:
                        for t in quartet_topologies_raw:
                            quartet_topologies.append(int(t) if isinstance(t, (int, float)) else 0)
                    else:
                        quartet_topologies = [0] * len(quartet_indices)

                    if target_dist is not None and target_dist.numel() > 0:
                        target_dist = target_dist.to(device)
                    else:
                        target_dist = k2p_dist

                    loss_fn = TripleLoss(alpha=alpha_w, beta=beta_w, gamma=gamma_w)
                    loss, _, _ = compute_phylogpn_loss(
                        model, onehot, encoded_seqs, tree_structure, None, None,
                        None, felsenstein, dist_head, calibration,
                        target_dist, quartet_indices, quartet_topologies,
                        comp_features, loss_fn, device,
                    )
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            print(f"Validation loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    "model": model.state_dict(),
                    "dist_head": dist_head.state_dict(),
                    "calibration": calibration.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": avg_val_loss,
                }, os.path.join(output_dir, "best_model.pt"))
                print(f"Saved best model (val_loss={avg_val_loss:.4f})")

        if (epoch + 1) % config.get("save_every", 10) == 0:
            torch.save({
                "model": model.state_dict(),
                "dist_head": dist_head.state_dict(),
                "calibration": calibration.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }, os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt"))

    torch.save({
        "model": model.state_dict(),
        "dist_head": dist_head.state_dict(),
        "calibration": calibration.state_dict(),
    }, os.path.join(output_dir, "final_model.pt"))
    print("Training complete.")


if __name__ == "__main__":
    import yaml
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/train/route_a_pretrain.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    train_route_a(config)
