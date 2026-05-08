import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.route_b.phyla_viral import PHYLAViralModel
from src.models.route_a.felsenstein import newick_to_tree_structure
from src.training.losses import QuartetLoss, DistanceRegressionLoss, PhyloLikelihoodLoss, TripleLoss, LossWeightScheduler
from src.data.phylo_dataset import PhyloTrainingDataset


def train_route_b(config):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = PHYLAViralModel(
        d_model=config.get("d_model", 256),
        n_mamba_layers=config.get("n_mamba_layers", 6),
        d_state=config.get("d_state", 16),
        d_conv=config.get("d_conv", 3),
        expand=config.get("expand", 2),
        n_tree_heads=config.get("n_tree_heads", 8),
        n_composition_features=config.get("n_composition_features", 20),
        use_calibration=config.get("use_calibration", True),
        use_gtr_head=config.get("use_gtr_head", True),
        max_seq_length=config.get("max_seq_length", 2048),
    ).to(device)

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

    optimizer = AdamW(model.parameters(), lr=config.get("lr", 5e-5), weight_decay=config.get("weight_decay", 0.01))
    scheduler = CosineAnnealingLR(optimizer, T_max=config.get("epochs", 50))

    loss_scheduler = LossWeightScheduler(
        total_epochs=config.get("epochs", 50),
        schedule_type=config.get("loss_schedule", "phased"),
    )

    best_val_loss = float('inf')
    output_dir = config.get("output_dir", "outputs/route_b")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(config.get("epochs", 50)):
        alpha_w, beta_w, gamma_w = loss_scheduler.get_weights(epoch)

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            sequences = batch["sequences"]
            names = batch["names"]
            comp_features = batch["composition_features"]
            target_dist = batch["target_distance"]
            k2p_dist = batch["k2p_distance"].to(device)
            quartet_indices_raw = batch["quartet_indices"]
            ref_tree = batch.get("ref_tree")

            ref_tree_newick = None
            if ref_tree is not None and isinstance(ref_tree, list) and ref_tree[0] is not None:
                ref_tree_newick = ref_tree[0]

            if target_dist is not None and target_dist.numel() > 0:
                target_dist = target_dist.to(device)
            else:
                target_dist = k2p_dist

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

            optimizer.zero_grad()

            dist_matrix, seq_emb, log_likelihood = model(
                sequences,
                composition_features=comp_features,
                ref_tree_newick=ref_tree_newick,
            )

            loss_fn = TripleLoss(alpha=alpha_w, beta=beta_w, gamma=gamma_w)
            loss = loss_fn(
                dist_matrix,
                target_dist=target_dist,
                log_likelihood=log_likelihood,
                quartet_indices=quartet_indices,
                quartet_topologies=quartet_topologies,
            )

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if batch_idx % 50 == 0:
                ll_str = f"{log_likelihood.item():.4f}" if log_likelihood is not None else "N/A"
                print(f"Epoch {epoch} Batch {batch_idx}: loss={loss.item():.4f} ll={ll_str}")

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

        if (epoch + 1) % config.get("eval_every", 5) == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in val_dataloader:
                    sequences = batch["sequences"]
                    comp_features = batch["composition_features"]
                    target_dist = batch["target_distance"]
                    k2p_dist = batch["k2p_distance"].to(device)
                    quartet_indices_raw = batch["quartet_indices"]
                    ref_tree = batch.get("ref_tree")

                    ref_tree_newick = None
                    if ref_tree is not None and isinstance(ref_tree, list) and ref_tree[0] is not None:
                        ref_tree_newick = ref_tree[0]

                    if target_dist is not None and target_dist.numel() > 0:
                        target_dist = target_dist.to(device)
                    else:
                        target_dist = k2p_dist

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

                    dist_matrix, seq_emb, log_likelihood = model(
                        sequences,
                        composition_features=comp_features,
                        ref_tree_newick=ref_tree_newick,
                    )

                    loss_fn = TripleLoss(alpha=alpha_w, beta=beta_w, gamma=gamma_w)
                    loss = loss_fn(
                        dist_matrix,
                        target_dist=target_dist,
                        log_likelihood=log_likelihood,
                        quartet_indices=quartet_indices,
                        quartet_topologies=quartet_topologies,
                    )
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            print(f"Validation loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": avg_val_loss,
                }, os.path.join(output_dir, "best_model.pt"))

        if (epoch + 1) % config.get("save_every", 10) == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }, os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt"))

    torch.save({"model": model.state_dict()}, os.path.join(output_dir, "final_model.pt"))
    print("Training complete.")


if __name__ == "__main__":
    import yaml
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/train/route_b_dual.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    train_route_b(config)
