import torch
import numpy as np
import os
import yaml
from Bio import SeqIO
from src.models.route_c.route_c_model import RouteCModel
from src.models.backbone.dnabert2_wrapper import DNABERT2Wrapper, NTWrapper
from src.models.route_a.viral_phylogpn import ViralPhyloGPN
from src.models.route_b.phyla_viral import PHYLAViralModel
from src.models.distance.k2p_baseline import K2PDistance, compute_k2p_matrix
from src.models.tree.nj_builder import nj_from_distance_matrix
from src.data.viral_dataset import CompositionFeatureExtractor


class ViroPhyloPipeline:
    def __init__(self, config_path, device="cuda"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = device
        self.route = self.config.get("route", "C")
        self.comp_extractor = CompositionFeatureExtractor(k=4)
        self.k2p = K2PDistance()
        self.model = self._load_model()

    def _load_model(self):
        route = self.route.upper()
        if route == "C":
            return self._load_route_c()
        elif route == "A":
            return self._load_route_a()
        elif route == "B":
            return self._load_route_b()
        else:
            raise ValueError(f"Unknown route: {route}")

    def _load_route_c(self):
        backbone_name = self.config.get("backbone", "dnabert2")
        lora_rank = self.config.get("lora_rank", 16)
        lora_alpha = self.config.get("lora_alpha", 32)

        if backbone_name == "dnabert2":
            backbone = DNABERT2Wrapper(
                lora_rank=lora_rank, lora_alpha=lora_alpha,
            ).to(self.device)
        else:
            backbone = NTWrapper(
                lora_rank=lora_rank, lora_alpha=lora_alpha,
            ).to(self.device)

        model = RouteCModel(
            backbone=backbone,
            embed_dim=backbone.embed_dim,
            use_calibration=self.config.get("use_calibration", True),
            use_hybrid_distance=self.config.get("use_hybrid_distance", True),
        ).to(self.device)

        checkpoint = self.config.get("checkpoint")
        if checkpoint and os.path.exists(checkpoint):
            state_dict = torch.load(checkpoint, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)

        return model

    def _load_route_a(self):
        model = ViralPhyloGPN(
            window_size=self.config.get("window_size", 241),
            d_model=self.config.get("d_model", 960),
            n_blocks=self.config.get("n_blocks", 40),
        ).to(self.device)

        checkpoint = self.config.get("checkpoint")
        if checkpoint and os.path.exists(checkpoint):
            state_dict = torch.load(checkpoint, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)

        return model

    def _load_route_b(self):
        model = PHYLAViralModel(
            d_model=self.config.get("d_model", 768),
            n_mamba_layers=self.config.get("n_mamba_layers", 12),
        ).to(self.device)

        checkpoint = self.config.get("checkpoint")
        if checkpoint and os.path.exists(checkpoint):
            state_dict = torch.load(checkpoint, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)

        return model

    def predict_tree(self, fasta_path, output_path=None):
        sequences, names = [], []
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append(str(record.seq).upper())
            names.append(record.id)

        if self.route.upper() == "C":
            comp_features = self.comp_extractor.extract_batch(sequences)
            comp_tensor = torch.from_numpy(comp_features).to(self.device)
            newick, dist_matrix = self.model.predict_tree(
                sequences, names, composition_features=comp_tensor
            )
        elif self.route.upper() == "B":
            newick, dist_matrix = self.model.predict_tree(sequences, names)
        else:
            newick, dist_matrix = self._predict_route_a(sequences, names)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(newick)
            np.save(output_path.replace('.nwk', '.npy'), dist_matrix)

        return newick, dist_matrix

    def _predict_route_a(self, sequences, names):
        from src.models.route_a.gtr_module import GTRModel
        encoded = self.k2p.encode_sequences(sequences).to(self.device)
        n = len(sequences)
        L = encoded.shape[1]
        onehot = torch.nn.functional.one_hot(encoded, num_classes=5).float()
        raw_rates, raw_freq, raw_alpha, _ = self.model(onehot)

        rates = torch.nn.functional.softplus(raw_rates)
        rates = rates * 6.0 / rates.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        frequencies = torch.nn.functional.softmax(raw_freq, dim=-1)
        alpha = torch.nn.functional.softplus(raw_alpha).clamp(min=0.1, max=10.0)

        avg_rates = rates.mean(dim=(0, 1))
        avg_freq = frequencies.mean(dim=(0, 1))

        gtr = GTRModel()
        Q = gtr.compute_Q_matrix(avg_rates, avg_freq)

        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                P = gtr.compute_P_matrix(Q, t=1.0)
                seq_i = onehot[i]
                seq_j = onehot[j]
                valid = (encoded[i] <= 3) & (encoded[j] <= 3)
                if valid.sum() == 0:
                    d = 0.0
                else:
                    ll = 0.0
                    for pos in range(L):
                        if valid[pos]:
                            obs_j = encoded[j, pos].item()
                            if obs_j <= 3:
                                ll += torch.log(P[encoded[i, pos].item(), obs_j] + 1e-30).item()
                    d = -ll / max(valid.sum().item(), 1)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        newick = nj_from_distance_matrix(dist_matrix, names)
        return newick, dist_matrix

    def predict_distance_matrix(self, fasta_path):
        sequences, names = [], []
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append(str(record.seq).upper())
            names.append(record.id)

        if self.route.upper() == "C":
            comp_features = self.comp_extractor.extract_batch(sequences)
            comp_tensor = torch.from_numpy(comp_features).to(self.device)
            _, dist_matrix = self.model.predict_tree(
                sequences, names, composition_features=comp_tensor
            )
        else:
            _, dist_matrix = self.predict_tree(fasta_path)

        return dist_matrix, names
