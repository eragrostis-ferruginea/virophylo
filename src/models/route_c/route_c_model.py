import torch
import torch.nn as nn
from src.models.calibration.zca_whitening import EmbeddingCalibration
from src.models.distance.distance_head import DistanceHead
from src.models.distance.hybrid_distance import HybridDistance
from src.models.distance.k2p_baseline import K2PDistance
from src.models.tree.nj_builder import nj_from_distance_matrix


class RouteCModel(nn.Module):
    def __init__(self, backbone, embed_dim, n_composition_features=20,
                 use_calibration=True, use_hybrid_distance=True,
                 distance_head_type="mlp", distance_hidden_dim=256):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.use_calibration = use_calibration
        self.use_hybrid_distance = use_hybrid_distance

        if use_calibration:
            self.calibration = EmbeddingCalibration(
                embed_dim=embed_dim,
                n_composition_features=n_composition_features,
                use_zca=True,
                use_debias=True,
                use_site_weight=True,
            )

        if distance_head_type == "mlp":
            self.distance_head = DistanceHead(embed_dim, distance_hidden_dim)
        else:
            self.distance_head = None

        if use_hybrid_distance:
            self.hybrid_distance = HybridDistance(
                learnable_alpha=True,
                init_alpha=0.8,
                use_cross_branch_correction=True,
            )
        self.k2p = K2PDistance()

    def forward(self, sequences, composition_features=None, encoded_seqs=None,
                return_embeddings=False):
        cls_embeddings, pos_embeddings = self.backbone(sequences)

        site_weights = None
        if self.use_calibration:
            cls_embeddings, site_weights = self.calibration(
                cls_embeddings, composition_features
            )

        n = cls_embeddings.shape[0]

        if self.distance_head is not None:
            d_llm = self.distance_head.pairwise_distances(cls_embeddings)
        else:
            d_llm = torch.cdist(cls_embeddings.unsqueeze(0), cls_embeddings.unsqueeze(0), p=2).squeeze(0)

        d_k2p = None
        if self.use_hybrid_distance:
            if encoded_seqs is not None:
                n = encoded_seqs.shape[0]
                d_k2p = torch.zeros(n, n, device=cls_embeddings.device)
                for i in range(n):
                    for j in range(i + 1, n):
                        valid = (encoded_seqs[i] <= 3) & (encoded_seqs[j] <= 3)
                        n_valid = valid.sum().item()
                        if n_valid == 0:
                            d_k2p[i, j] = 0.0
                            d_k2p[j, i] = 0.0
                            continue
                        same = (encoded_seqs[i] == encoded_seqs[j]) & valid
                        diff = valid & ~same
                        p = diff.sum().float() / n_valid
                        if p >= 0.75:
                            d_k2p[i, j] = 10.0
                        else:
                            d_k2p[i, j] = -0.75 * torch.log(1.0 - 4.0 * p / 3.0)
                        d_k2p[j, i] = d_k2p[i, j]
            else:
                d_k2p = self.k2p.compute(sequences).to(cls_embeddings.device)

        if self.use_hybrid_distance and d_k2p is not None:
            dist_matrix = self.hybrid_distance(d_llm, d_k2p)
        else:
            dist_matrix = d_llm

        if return_embeddings:
            return dist_matrix, cls_embeddings

        return dist_matrix

    def predict_tree(self, sequences, names=None, composition_features=None, encoded_seqs=None):
        self.eval()
        with torch.no_grad():
            dist_matrix = self.forward(sequences, composition_features, encoded_seqs)
            dist_np = dist_matrix.cpu().numpy()
            newick = nj_from_distance_matrix(dist_np, names)
        return newick, dist_np
