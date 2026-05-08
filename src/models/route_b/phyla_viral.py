import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.route_b.bimamba_block import BiMambaStack, TreeHead, NucleotideTokenizer
from src.models.route_a.gtr_module import GTRModel, GTRParameterHead
from src.models.route_a.felsenstein import FelsensteinPruning
from src.models.distance.distance_head import DistanceHead
from src.models.calibration.zca_whitening import EmbeddingCalibration
from src.models.tree.nj_builder import nj_from_distance_matrix


class PHYLAViralModel(nn.Module):
    def __init__(self, d_model=768, n_mamba_layers=12, d_state=16, d_conv=4,
                 expand=2, n_tree_heads=8, n_composition_features=20,
                 use_calibration=True, use_gtr_head=True):
        super().__init__()
        self.d_model = d_model
        self.use_calibration = use_calibration
        self.use_gtr_head = use_gtr_head

        self.tokenizer = NucleotideTokenizer(k=6, d_model=d_model)
        self.bimamba = BiMambaStack(d_model, n_mamba_layers, d_state, d_conv, expand)
        self.tree_head = TreeHead(d_model, n_tree_heads)

        if use_calibration:
            self.calibration = EmbeddingCalibration(
                embed_dim=d_model,
                n_composition_features=n_composition_features,
            )

        self.distance_head = DistanceHead(d_model, hidden_dim=256)

        if use_gtr_head:
            self.gtr_head = GTRParameterHead(d_model, hidden_dim=128)
            self.gtr_model = GTRModel()
            self.felsenstein = FelsensteinPruning(n_bases=4, n_gamma_categories=4)

    def forward(self, sequences, composition_features=None, tree_structures=None,
                alignment=None):
        token_embeddings = self.tokenizer(sequences)

        n_seqs = len(sequences)
        seq_len = token_embeddings.shape[1]

        all_tokens = token_embeddings.reshape(1, n_seqs * seq_len, -1)
        encoded = self.bimamba(all_tokens)
        encoded = encoded.reshape(n_seqs, seq_len, -1)

        cls_tokens = encoded[:, 0, :]

        if self.use_calibration and composition_features is not None:
            cls_tokens, _ = self.calibration(cls_tokens, composition_features)

        n = cls_tokens.shape[0]
        dist_matrix = self.distance_head.pairwise_distances(cls_tokens)

        gtr_log_likelihood = None
        if self.use_gtr_head and tree_structures is not None and alignment is not None:
            raw_rates, raw_freq, raw_alpha = self.gtr_head(cls_tokens)
            avg_rates = raw_rates.mean(dim=0)
            avg_freq = raw_freq.mean(dim=0)
            avg_alpha = raw_alpha.mean()

            P, rates, frequencies, alpha, Q = self.gtr_model(
                avg_rates, avg_freq, avg_alpha, t=0.1
            )
            gtr_log_likelihood = self.felsenstein(
                alignment, tree_structures, Q, frequencies, alpha
            )

        return dist_matrix, cls_tokens, gtr_log_likelihood

    def predict_tree(self, sequences, names=None, composition_features=None):
        self.eval()
        with torch.no_grad():
            dist_matrix, _, _ = self.forward(sequences, composition_features)
            dist_np = dist_matrix.cpu().numpy()
            newick = nj_from_distance_matrix(dist_np, names)
        return newick, dist_np
