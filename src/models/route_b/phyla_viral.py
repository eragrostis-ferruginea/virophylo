import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.route_b.bimamba_block import BiMambaStack, TreeHead, NucleotideTokenizer
from src.models.distance.distance_head import EuclideanDistanceHead
from src.models.distance.hybrid_distance import HybridDistance
from src.models.distance.k2p_baseline import K2PDistance
from src.models.calibration.zca_whitening import EmbeddingCalibration
from src.models.route_a.gtr_module import GTRModel, GTRParameterHead
from src.models.route_a.felsenstein import FelsensteinPruning, newick_to_tree_structure


class PHYLAViralModel(nn.Module):
    def __init__(self, d_model=256, n_mamba_layers=6, d_state=16, d_conv=3, expand=2,
                 n_tree_heads=8, n_composition_features=20,
                 use_calibration=True, use_gtr_head=True,
                 max_seq_length=2048):
        super().__init__()
        self.d_model = d_model
        self.use_calibration = use_calibration
        self.use_gtr_head = use_gtr_head
        self.max_seq_length = max_seq_length

        self.tokenizer = NucleotideTokenizer(k=6, d_model=d_model, vocab_size=4096)
        self.mamba_stack = BiMambaStack(d_model=d_model, n_layers=n_mamba_layers,
                                         d_state=d_state, d_conv=d_conv, expand=expand)
        self.tree_head = TreeHead(d_model=d_model, n_heads=n_tree_heads)
        self.dist_head = EuclideanDistanceHead()
        self.hybrid_distance = HybridDistance(learnable_alpha=True, init_alpha=0.8)

        if use_calibration:
            self.calibration = EmbeddingCalibration(
                embed_dim=d_model,
                n_composition_features=n_composition_features,
                use_zca=True,
                use_debias=True,
                use_site_weight=True,
            )

        if use_gtr_head:
            self.gtr_head = GTRParameterHead(embed_dim=d_model)
            self.felsenstein = FelsensteinPruning(n_bases=4, n_gamma_categories=4)

        self.k2p = K2PDistance()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_length + 1, d_model) * 0.02)

    def _encode_sequences(self, sequences):
        batch_embeddings = []
        batch_lengths = []

        for seq in sequences:
            seq = seq.upper().replace('U', 'T')
            k = self.tokenizer.k
            tokens = []
            for i in range(0, len(seq) - k + 1, k):
                kmer = seq[i:i + k]
                if all(c in 'ACGT' for c in kmer):
                    idx = 0
                    for j, c in enumerate(kmer):
                        idx = idx * 4 + {'A': 0, 'C': 1, 'G': 2, 'T': 3}[c]
                    tokens.append(idx)

            if len(tokens) == 0:
                tokens = [0]

            token_tensor = torch.tensor(tokens, dtype=torch.long, device=self.cls_token.device)
            token_emb = self.tokenizer.embedding(token_tensor)

            n_tokens = min(token_emb.shape[0], self.max_seq_length - 1)
            token_emb = token_emb[:n_tokens]
            batch_lengths.append(n_tokens)
            batch_embeddings.append(token_emb)

        max_len = max(batch_lengths)
        padded = torch.zeros(len(sequences), max_len, self.d_model, device=self.cls_token.device)
        for i, emb in enumerate(batch_embeddings):
            n = batch_lengths[i]
            padded[i, :n] = emb

        return padded, batch_lengths

    def forward(self, sequences, composition_features=None, ref_tree_newick=None):
        device = self.cls_token.device
        n_seqs = len(sequences)

        token_emb, seq_lengths = self._encode_sequences(sequences)

        cls = self.cls_token.expand(n_seqs, -1, -1)
        max_len = token_emb.shape[1]

        combined = torch.cat([cls, token_emb], dim=1)
        pos = self.pos_embed[:, :combined.shape[1], :]
        combined = combined + pos

        mamba_out = self.mamba_stack(combined)

        cls_out = mamba_out[:, 0:1, :]

        seq_pooled = []
        for i in range(n_seqs):
            n = seq_lengths[i]
            if n > 0:
                pooled = mamba_out[i, 1:n + 1, :].mean(dim=0)
            else:
                pooled = mamba_out[i, 1, :]
            seq_pooled.append(pooled)
        seq_emb = torch.stack(seq_pooled)

        if self.use_calibration and composition_features is not None:
            seq_emb, _ = self.calibration(seq_emb, composition_features.to(device))

        ll_dist = self.dist_head.pairwise_distances(seq_emb)
        k2p_dist = self.k2p.compute(sequences).to(device)
        dist_matrix = self.hybrid_distance(ll_dist, k2p_dist)

        log_likelihood = None
        if self.use_gtr_head and ref_tree_newick is not None:
            try:
                raw_rates, raw_freq, raw_alpha = self.gtr_head(seq_emb)
                rates = self.gtr_head.normalize_rates_fn(raw_rates)
                freqs = self.gtr_head.normalize_frequencies_fn(raw_freq)
                alpha = F.softplus(raw_alpha).clamp(min=0.1, max=10.0)

                gtr = GTRModel()
                Q = gtr.compute_Q_matrix(rates.mean(dim=0), freqs.mean(dim=0))

                encoded = self.k2p.encode_sequences(sequences)
                tree_struct = newick_to_tree_structure(ref_tree_newick)
                ll = self.felsenstein(encoded, tree_struct, Q, freqs.mean(dim=0), alpha.mean())
                log_likelihood = ll
            except Exception:
                log_likelihood = torch.tensor(0.0, device=device)

        return dist_matrix, seq_emb, log_likelihood
