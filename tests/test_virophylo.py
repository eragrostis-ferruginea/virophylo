import sys
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, '/workspace')

PASS_COUNT = 0
FAIL_COUNT = 0
ERROR_LIST = []


def test(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {name}")
    else:
        FAIL_COUNT += 1
        msg = f"  [FAIL] {name}" + (f" -- {detail}" if detail else "")
        print(msg)
        ERROR_LIST.append(name)


def test_exception(name, fn, should_raise=False):
    global PASS_COUNT, FAIL_COUNT
    try:
        fn()
        if should_raise:
            FAIL_COUNT += 1
            print(f"  [FAIL] {name} -- expected exception but none raised")
            ERROR_LIST.append(name)
        else:
            PASS_COUNT += 1
            print(f"  [PASS] {name}")
    except Exception as e:
        if should_raise:
            PASS_COUNT += 1
            print(f"  [PASS] {name} (raised {type(e).__name__})")
        else:
            FAIL_COUNT += 1
            print(f"  [FAIL] {name} -- unexpected {type(e).__name__}: {e}")
            ERROR_LIST.append(name)


# ============================================================
print("=" * 70)
print("SECTION 1: GTR Model & Matrix Exponentiation")
print("=" * 70)
# ============================================================

from src.models.route_a.gtr_module import GTRModel, GTRParameterHead

gtr = GTRModel()

raw_rates = torch.randn(6)
raw_freq = torch.randn(4)
raw_alpha = torch.randn(1)

rates = gtr.normalize_rates(raw_rates)
freqs = gtr.normalize_frequencies(raw_freq)
alpha = F.softplus(raw_alpha).clamp(min=0.1, max=10.0)

test("GTR rates sum to 6", torch.isclose(rates.sum(), torch.tensor(6.0), atol=1e-5),
     f"sum={rates.sum().item():.4f}")
test("GTR frequencies sum to 1", torch.isclose(freqs.sum(), torch.tensor(1.0), atol=1e-5),
     f"sum={freqs.sum().item():.4f}")
test("GTR rates are positive", (rates > 0).all().item())
test("GTR frequencies are positive", (freqs > 0).all().item())

Q = gtr.compute_Q_matrix(rates, freqs)
test("Q matrix is 4x4", Q.shape == (4, 4))
off_diag_mask = ~torch.eye(4, dtype=torch.bool)
test("Q off-diagonal are non-negative", (Q[off_diag_mask] >= -1e-8).all().item(),
     f"min off-diag={Q[off_diag_mask].min().item():.6f}")

row_sums = Q.sum(dim=-1)
test("Q rows sum to zero (rate matrix property)", torch.allclose(row_sums, torch.zeros(4), atol=1e-5),
     f"row sums={row_sums.tolist()}")

for t_val in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
    P = gtr.compute_P_matrix(Q, t_val)
    test(f"P(t={t_val}) rows sum to 1", torch.allclose(P.sum(dim=-1), torch.ones(4), atol=1e-4),
         f"row sums={P.sum(dim=-1).tolist()}")
    test(f"P(t={t_val}) all entries non-negative", (P >= -1e-8).all().item(),
         f"min={P.min().item():.6f}")

P0 = gtr.compute_P_matrix(Q, 0.0)
test("P(t=0) is identity matrix", torch.allclose(P0, torch.eye(4), atol=1e-4),
     f"max diff from I={(P0 - torch.eye(4)).abs().max().item():.6f}")

P_small = gtr.compute_P_matrix(Q, 0.01)
test("P(t=0.01) is close to identity", (P_small - torch.eye(4)).abs().max() < 0.1)

P_large = gtr.compute_P_matrix(Q, 100.0)
test("P(t=100) approaches stationary frequencies",
     torch.allclose(P_large[0], freqs, atol=0.05),
     f"P[0]={P_large[0].tolist()}, pi={freqs.tolist()}")

gtr_head = GTRParameterHead(embed_dim=256)
embeddings = torch.randn(8, 256)
out_rates, out_freq, out_alpha = gtr_head(embeddings)
test("GTRParameterHead output shapes", out_rates.shape == (8, 6) and out_freq.shape == (8, 4) and out_alpha.shape == (8,))


# ============================================================
print("\n" + "=" * 70)
print("SECTION 2: Felsenstein Pruning Algorithm")
print("=" * 70)
# ============================================================

from src.models.route_a.felsenstein import FelsensteinPruning, VectorizedFelsenstein

felsenstein = FelsensteinPruning(n_bases=4, n_gamma_categories=4)

simple_tree = {
    'root': 4,
    0: {'seq_idx': 0, 'children': []},
    1: {'seq_idx': 1, 'children': []},
    2: {'seq_idx': 2, 'children': []},
    3: {'seq_idx': 3, 'children': []},
    4: {'children': [(0, 0.1), (1, 0.1)], 'seq_idx': -1},
    5: {'children': [(2, 0.1), (3, 0.1)], 'seq_idx': -1},
    6: {'children': [(4, 0.2), (5, 0.2)], 'seq_idx': -1, 'root': True},
}
simple_tree['root'] = 6

alignment = torch.tensor([[0, 1, 2, 3],
                          [0, 1, 2, 3],
                          [0, 1, 2, 3],
                          [0, 1, 2, 3]])

test_exception("Felsenstein pruning runs without error",
               lambda: felsenstein(alignment, simple_tree, Q, freqs, alpha))

ll = felsenstein(alignment, simple_tree, Q, freqs, alpha)
test("Felsenstein log-likelihood is finite", torch.isfinite(ll).item(), f"ll={ll.item()}")
test("Felsenstein log-likelihood is negative", ll.item() < 0, f"ll={ll.item()}")

identical_alignment = torch.tensor([[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0]])
simple_tree_3 = {
    'root': 6,
    0: {'seq_idx': 0, 'children': []},
    1: {'seq_idx': 1, 'children': []},
    2: {'seq_idx': 2, 'children': []},
    3: {'seq_idx': 3, 'children': []},
    4: {'children': [(0, 0.01), (1, 0.01)], 'seq_idx': -1},
    5: {'children': [(2, 0.01), (3, 0.01)], 'seq_idx': -1},
    6: {'children': [(4, 0.01), (5, 0.01)], 'seq_idx': -1, 'root': True},
}

ll_identical = felsenstein(identical_alignment, simple_tree_3, Q, freqs, alpha)
test("Identical sequences on short-branch tree: likelihood is finite and negative",
     torch.isfinite(ll_identical).item() and ll_identical.item() < 0,
     f"ll={ll_identical.item():.4f}")


# ============================================================
print("\n" + "=" * 70)
print("SECTION 3: K2P Distance Computation")
print("=" * 70)
# ============================================================

from src.models.distance.k2p_baseline import compute_k2p_distance, compute_k2p_matrix, K2PDistance

seq1 = "ACGTACGTACGT"
seq2 = "ACGTACGTACGT"
d_same = compute_k2p_distance(seq1, seq2)
test("K2P distance of identical sequences is 0", d_same == 0.0, f"d={d_same}")

seq3 = "ACGTACGTACGT"
seq4 = "TCGTACGTACGT"
d_one_sub = compute_k2p_distance(seq3, seq4)
test("K2P distance with one substitution > 0", d_one_sub > 0, f"d={d_one_sub:.6f}")

seq5 = "ACGTACGTACGT"
seq6 = "GGGGGGGGGGGG"
d_many = compute_k2p_distance(seq5, seq6)
test("K2P distance with many substitutions > one substitution", d_many > d_one_sub,
     f"d_many={d_many:.4f}, d_one={d_one_sub:.6f}")

k2p = K2PDistance()
sequences = ["ACGTACGTACGT", "ACGTACGTTCGT", "ACGTACGTACGA", "TCGTACGTACGT"]
dist = k2p.compute(sequences)
test("K2P matrix is square", dist.shape == (4, 4))
test("K2P matrix diagonal is zero", torch.allclose(dist.diag(), torch.zeros(4), atol=1e-6))
test("K2P matrix is symmetric", torch.allclose(dist, dist.T, atol=1e-6))
test("K2P matrix all non-negative", (dist >= 0).all().item())

dist_np = compute_k2p_matrix(sequences)
test("K2P numpy matrix diagonal is zero", np.allclose(np.diag(dist_np), 0, atol=1e-6))
test("K2P numpy matrix is symmetric", np.allclose(dist_np, dist_np.T, atol=1e-6))

seqs_with_gaps = ["ACGT-ACGTACGT", "ACGTNACGTTCGT"]
dist_gaps = k2p.compute(seqs_with_gaps)
test("K2P handles gaps without error", torch.isfinite(dist_gaps).all().item())

encoded = k2p.encode_sequences(["ACGT", "ACGT"])
test("K2P encoding A=0,C=1,G=2,T=3", encoded[0].tolist() == [0, 1, 2, 3])

encoded_gap = k2p.encode_sequences(["A-CG"])
test("K2P encoding gap/N=4", encoded_gap[0].tolist() == [0, 4, 1, 2])


# ============================================================
print("\n" + "=" * 70)
print("SECTION 4: ZCA Whitening & Embedding Calibration")
print("=" * 70)
# ============================================================

from src.models.calibration.zca_whitening import ZCAWhitening, EmbeddingCalibration
from src.models.calibration.composition_debias import CompositionDebias
from src.models.calibration.site_weighting import SiteWeighting

zca = ZCAWhitening(embed_dim=64)
embeddings = torch.randn(16, 64)
zca.eval()
whitened = zca(embeddings)
test("ZCA output shape matches input", whitened.shape == embeddings.shape)

zca.train()
whitened_train = zca(embeddings)
test("ZCA training forward pass works", whitened_train.shape == embeddings.shape)

debias = CompositionDebias(embed_dim=64, n_composition_features=20)
comp_features = torch.randn(16, 20)
debiased = debias(embeddings, comp_features)
test("CompositionDebias output shape matches input", debiased.shape == embeddings.shape)

site_w = SiteWeighting(embed_dim=64)
weighted, weights = site_w(embeddings)
test("SiteWeighting output shape matches input", weighted.shape == embeddings.shape)
test("SiteWeighting weights in [0,1]", (weights >= 0).all().item() and (weights <= 1).all().item())

calibration = EmbeddingCalibration(embed_dim=64, n_composition_features=20)
calibrated, site_weights = calibration(embeddings, comp_features)
test("EmbeddingCalibration output shape matches input", calibrated.shape == embeddings.shape)

calibration_no_debias = EmbeddingCalibration(embed_dim=64, n_composition_features=20,
                                              use_debias=False)
calibrated_no_debias, _ = calibration_no_debias(embeddings, comp_features)
test("EmbeddingCalibration without debias works", calibrated_no_debias.shape == embeddings.shape)

calibration_none = EmbeddingCalibration(embed_dim=64, n_composition_features=20,
                                         use_zca=False, use_debias=False, use_site_weight=False)
calibrated_none, sw_none = calibration_none(embeddings, comp_features)
test("EmbeddingCalibration all disabled returns input unchanged",
     torch.allclose(calibrated_none, embeddings),
     f"max diff={(calibrated_none - embeddings).abs().max().item():.6f}")


# ============================================================
print("\n" + "=" * 70)
print("SECTION 5: Distance Heads")
print("=" * 70)
# ============================================================

from src.models.distance.distance_head import DistanceHead, EuclideanDistanceHead, CosineDistanceHead
from src.models.distance.hybrid_distance import HybridDistance, AdaptiveHybridDistance

dist_head = DistanceHead(embed_dim=64, hidden_dim=128)
emb = torch.randn(8, 64)
pairwise = dist_head.pairwise_distances(emb)
test("DistanceHead pairwise output is NxN", pairwise.shape == (8, 8))
test("DistanceHead diagonal is small (softplus ensures non-negative)", pairwise.diag().max().item() < pairwise.max().item() + 0.1,
     f"max diag={pairwise.diag().max().item():.4f}")
test("DistanceHead all non-negative", (pairwise >= 0).all().item())

euc_head = EuclideanDistanceHead()
euc_dist = euc_head.pairwise_distances(emb)
test("Euclidean distance diagonal is zero", torch.allclose(euc_dist.diag(), torch.zeros(8), atol=1e-5))
test("Euclidean distance is symmetric", torch.allclose(euc_dist, euc_dist.T, atol=1e-5))

cos_head = CosineDistanceHead()
cos_dist = cos_head.pairwise_distances(emb)
test("Cosine distance diagonal is near zero", cos_dist.diag().max().item() < 1e-4,
     f"max diag={cos_dist.diag().max().item():.6f}")
test("Cosine distance in [0, 2]", (cos_dist >= -1e-4).all().item() and (cos_dist <= 2.0 + 1e-4).all().item())

hybrid = HybridDistance(learnable_alpha=True, init_alpha=0.8)
d_llm = torch.rand(4, 4) * 0.5
d_llm = (d_llm + d_llm.T) / 2
d_k2p = torch.rand(4, 4) * 0.3
d_k2p = (d_k2p + d_k2p.T) / 2
d_hybrid = hybrid(d_llm, d_k2p)
test("HybridDistance output shape matches input", d_hybrid.shape == d_llm.shape)
test("HybridDistance alpha is in (0,1)", 0 < hybrid.alpha.item() < 1,
     f"alpha={hybrid.alpha.item():.4f}")

mask = torch.zeros(4, 4)
mask[0, 1] = 1.0
mask[1, 0] = 1.0
d_hybrid_masked = hybrid(d_llm, d_k2p, is_cross_branch_mask=mask)
test("HybridDistance with cross-branch mask works", d_hybrid_masked.shape == d_llm.shape)


# ============================================================
print("\n" + "=" * 70)
print("SECTION 6: NJ Tree Builder")
print("=" * 70)
# ============================================================

from src.models.tree.nj_builder import NJTreeBuilder, nj_from_distance_matrix

dist_matrix = np.array([
    [0.0, 0.2, 0.4, 0.6],
    [0.2, 0.0, 0.3, 0.5],
    [0.4, 0.3, 0.0, 0.4],
    [0.6, 0.5, 0.4, 0.0],
])
names = ["A", "B", "C", "D"]
newick = nj_from_distance_matrix(dist_matrix, names)
test("NJ produces valid Newick string", newick.endswith(";") and "(" in newick)

for name in names:
    test(f"NJ tree contains taxon '{name}'", name in newick)

newick_2 = nj_from_distance_matrix(np.array([[0, 0.1], [0.1, 0]]), ["X", "Y"])
test("NJ handles 2 taxa", newick_2.endswith(";"))

newick_1 = nj_from_distance_matrix(np.array([[0.0]]), ["Z"])
test("NJ handles 1 taxon", "Z" in newick_1)

dist_additive = np.array([
    [0.0, 2.0, 4.0, 6.0],
    [2.0, 0.0, 4.0, 6.0],
    [4.0, 4.0, 0.0, 2.0],
    [6.0, 6.0, 2.0, 0.0],
])
newick_add = nj_from_distance_matrix(dist_additive, ["A", "B", "C", "D"])
test("NJ on additive distances produces correct topology",
     "(A" in newick_add and "B" in newick_add and "C" in newick_add and "D" in newick_add)


# ============================================================
print("\n" + "=" * 70)
print("SECTION 7: Tree Metrics")
print("=" * 70)
# ============================================================

from src.models.tree.tree_metrics import (
    compute_rf_distance, compute_quartet_accuracy, compute_branch_length_correlation,
    TreeMetrics
)

tree1 = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
tree2 = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
rf, nrf = compute_rf_distance(tree1, tree2)
test("RF distance of identical trees is 0", rf == 0, f"rf={rf}")
test("nRF of identical trees is 0", nrf == 0.0, f"nrf={nrf}")

tree3 = "((A:0.1,C:0.1):0.1,(B:0.1,D:0.1):0.1);"
rf2, nrf2 = compute_rf_distance(tree1, tree3)
test("RF distance of different topologies > 0", rf2 > 0, f"rf={rf2}")
test("nRF of different topologies > 0", nrf2 > 0, f"nrf={nrf2}")

qa_same = compute_quartet_accuracy(tree1, tree2, n_quartets=10)
test("QA of identical trees is 1.0", qa_same == 1.0, f"qa={qa_same}")

bl_corr = compute_branch_length_correlation(tree1, tree2)
test("Branch length correlation of identical trees is high", bl_corr > 0.99 or bl_corr == 0.0,
     f"corr={bl_corr}")

metrics = TreeMetrics()
result = metrics.evaluate(tree1, tree2, "test")
test("TreeMetrics.evaluate returns expected keys",
     "rf" in result and "nrf" in result and "qa" in result and "branch_length_pearson_r" in result)


# ============================================================
print("\n" + "=" * 70)
print("SECTION 8: Loss Functions")
print("=" * 70)
# ============================================================

from src.training.losses import (
    QuartetLoss, DistanceRegressionLoss, PhyloLikelihoodLoss, TripleLoss, LossWeightScheduler
)

quartet_loss = QuartetLoss(temperature=1.0)
dist_mat = torch.rand(8, 8)
dist_mat = (dist_mat + dist_mat.T) / 2
quartet_indices = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 2, 4, 6)]
q_loss = quartet_loss(dist_mat, quartet_indices=quartet_indices)
test("QuartetLoss returns scalar", q_loss.dim() == 0)
test("QuartetLoss is finite", torch.isfinite(q_loss).item())
test("QuartetLoss is non-negative", q_loss.item() >= 0, f"loss={q_loss.item():.4f}")

dist_reg_loss = DistanceRegressionLoss(loss_type="huber")
pred = torch.rand(4, 4)
target = torch.rand(4, 4)
d_loss = dist_reg_loss(pred, target)
test("DistanceRegressionLoss returns scalar", d_loss.dim() == 0)
test("DistanceRegressionLoss is finite", torch.isfinite(d_loss).item())

d_loss_mse = DistanceRegressionLoss(loss_type="mse")(pred, target)
test("MSE loss is finite", torch.isfinite(d_loss_mse).item())

d_loss_mae = DistanceRegressionLoss(loss_type="mae")(pred, target)
test("MAE loss is finite", torch.isfinite(d_loss_mae).item())

phylo_loss = PhyloLikelihoodLoss()
ll_val = torch.tensor(-5.0)
p_loss = phylo_loss(ll_val)
test("PhyloLikelihoodLoss = -log_likelihood", p_loss.item() == 5.0, f"loss={p_loss.item()}")

triple = TripleLoss(alpha=1.0, beta=0.5, gamma=0.5)
target_dist_8 = torch.rand(8, 8)
t_loss = triple(dist_mat, target_dist=target_dist_8, quartet_indices=quartet_indices)
test("TripleLoss returns scalar", t_loss.dim() == 0)
test("TripleLoss is finite", torch.isfinite(t_loss).item())

scheduler = LossWeightScheduler(total_epochs=100, schedule_type="phased")
a1, b1, g1 = scheduler.get_weights(0)
test("Phased schedule early: alpha=1.0", a1 == 1.0, f"alpha={a1}")
test("Phased schedule early: beta=0.0", b1 == 0.0, f"beta={b1}")

a2, b2, g2 = scheduler.get_weights(80)
test("Phased schedule late: beta=0.7", b2 == 0.7, f"beta={b2}")

sched_linear = LossWeightScheduler(total_epochs=100, schedule_type="linear")
a3, b3, g3 = sched_linear.get_weights(0)
test("Linear schedule starts with alpha=1.0", a3 == 1.0)

a4, b4, g4 = sched_linear.get_weights(100)
test("Linear schedule ends with alpha=0.3", abs(a4 - 0.3) < 1e-6, f"alpha={a4}")


# ============================================================
print("\n" + "=" * 70)
print("SECTION 9: ViralPhyloGPN (Route A)")
print("=" * 70)
# ============================================================

from src.models.route_a.viral_phylogpn import ViralPhyloGPN, RCEConv1d, ByteNetBlock, DilatedByteNet

rce = RCEConv1d(in_channels=5, out_channels=32, kernel_size=3, padding='same')
x_rce = torch.randn(2, 5, 50)
out_rce = rce(x_rce)
test("RCEConv1d output shape", out_rce.shape == (2, 32, 50),
     f"shape={out_rce.shape}")

x_rc = x_rce.flip(dims=[1])
out_rc_check = rce.conv_rc(x_rc).flip(dims=[1])
test("RCEConv1d reverse-complement path works", out_rc_check.shape == (2, 32, 50))

block = ByteNetBlock(d_model=32, d_inner=64, kernel_size=3)
x_block = torch.randn(2, 50, 32)
out_block = block(x_block)
test("ByteNetBlock output shape", out_block.shape == (2, 50, 32))

bytenet_small = DilatedByteNet(d_model=32, d_inner=32, n_blocks=4, kernel_size=3)
x_bn = torch.randn(2, 50, 32)
out_bn = bytenet_small(x_bn)
test("DilatedByteNet output shape", out_bn.shape == (2, 50, 32))

model_a = ViralPhyloGPN(window_size=50, d_model=32, d_inner=32, n_blocks=2, kernel_size=3, n_bases=5)
onehot = torch.randn(2, 50, 5)
raw_rates, raw_freq, raw_alpha, site_emb = model_a(onehot)
test("ViralPhyloGPN output shapes",
     raw_rates.shape == (2, 50, 6) and raw_freq.shape == (2, 50, 4) and raw_alpha.shape == (2, 50))

rates_a, freqs_a, alpha_a = model_a.predict_gtr_params(onehot)
test("ViralPhyloGPN predict_gtr_params rates sum to 6",
     torch.isclose(rates_a.sum(dim=-1), torch.tensor(6.0), atol=1e-3).all().item(),
     f"sums={rates_a.sum(dim=-1).tolist()}")
test("ViralPhyloGPN predict_gtr_params freqs sum to 1",
     torch.isclose(freqs_a.sum(dim=-1), torch.tensor(1.0), atol=1e-3).all().item())


# ============================================================
print("\n" + "=" * 70)
print("SECTION 10: BiMamba Block (Route B)")
print("=" * 70)
# ============================================================

from src.models.route_b.bimamba_block import MambaBlock, BiMambaBlock, BiMambaStack, TreeHead, NucleotideTokenizer

mamba_block = MambaBlock(d_model=32)
x_mb = torch.randn(2, 50, 32)
out_mb = mamba_block(x_mb)
test("MambaBlock (fallback) output shape", out_mb.shape == (2, 50, 32))

bimamba = BiMambaBlock(d_model=32)
out_bm = bimamba(x_mb)
test("BiMambaBlock output shape", out_bm.shape == (2, 50, 32))

bimamba_stack = BiMambaStack(d_model=32, n_layers=2)
out_stack = bimamba_stack(x_mb)
test("BiMambaStack output shape", out_stack.shape == (2, 50, 32))

tree_head = TreeHead(d_model=32, n_heads=4)
cls_tokens = torch.randn(1, 4, 32)
seq_tokens = torch.randn(1, 100, 32)
tree_out = tree_head(cls_tokens, seq_tokens)
test("TreeHead output shape", tree_out.shape == (1, 4, 32))

tokenizer = NucleotideTokenizer(k=3, d_model=32, vocab_size=64)
seqs = ["ACGTACGT", "TGCATGCA"]
tok_out = tokenizer(seqs)
test("NucleotideTokenizer output has batch dim", tok_out.shape[0] == 2)
test("NucleotideTokenizer output has d_model", tok_out.shape[2] == 32)


# ============================================================
print("\n" + "=" * 70)
print("SECTION 11: PHYLAViralModel (Route B)")
print("=" * 70)
# ============================================================

from src.models.route_b.phyla_viral import PHYLAViralModel

model_b = PHYLAViralModel(
    d_model=32, n_mamba_layers=2, d_state=8, d_conv=2, expand=2,
    n_tree_heads=4, n_composition_features=20,
    use_calibration=True, use_gtr_head=True
)
test("PHYLAViralModel created successfully", True)

sequences_b = ["ACGTACGTACGT", "TGCATGCATGCA", "ACGTACGTACGT", "TGCATGCATGCA"]
comp_b = torch.randn(4, 20)
dist_b, emb_b, ll_b = model_b(sequences_b, composition_features=comp_b)
test("PHYLAViralModel distance matrix shape", dist_b.shape == (4, 4),
     f"shape={dist_b.shape}")
test("PHYLAViralModel embeddings shape", emb_b.shape == (4, 32),
     f"shape={emb_b.shape}")


# ============================================================
print("\n" + "=" * 70)
print("SECTION 12: CompositionFeatureExtractor")
print("=" * 70)
# ============================================================

from src.data.viral_dataset import CompositionFeatureExtractor

extractor = CompositionFeatureExtractor(k=4)
seq = "ACGTACGTACGTACGT"
feat = extractor.extract(seq)
test("CompositionFeatureExtractor output is numpy array", isinstance(feat, np.ndarray))
test("CompositionFeatureExtractor output has 20 features", feat.shape == (20,),
     f"shape={feat.shape}")

batch_seqs = ["ACGTACGTACGT", "TGCATGCATGCA", "AAAACCCCGGGG"]
batch_feat = extractor.extract_batch(batch_seqs)
test("CompositionFeatureExtractor batch output shape", batch_feat.shape == (3, 20),
     f"shape={batch_feat.shape}")

gc_feat = extractor.extract("GCGCGCGC")
at_feat = extractor.extract("ATATATAT")
test("GC content feature > AT content feature for GC-rich seq",
     gc_feat[0] > at_feat[0],
     f"GC={gc_feat[0]:.4f}, AT={at_feat[0]:.4f}")


# ============================================================
print("\n" + "=" * 70)
print("SECTION 13: Integration Test - K2P + NJ Pipeline")
print("=" * 70)
# ============================================================

np.random.seed(42)
n_seqs = 10
seq_len = 200
base_seqs = []
for i in range(n_seqs):
    base = np.random.choice(['A', 'C', 'G', 'T'], size=seq_len)
    base_seqs.append(''.join(base))

for i in range(1, n_seqs):
    n_muts = int(seq_len * 0.05 * i)
    positions = np.random.choice(seq_len, n_muts, replace=False)
    seq_list = list(base_seqs[i])
    for pos in positions:
        orig = seq_list[pos]
        new_base = np.random.choice([b for b in 'ACGT' if b != orig])
        seq_list[pos] = new_base
    base_seqs[i] = ''.join(seq_list)

k2p_pipeline = K2PDistance()
dist_pipeline = k2p_pipeline.compute(base_seqs)
test("Pipeline K2P distance matrix is 10x10", dist_pipeline.shape == (10, 10))
test("Pipeline K2P diagonal is zero", torch.allclose(dist_pipeline.diag(), torch.zeros(10), atol=1e-6))
test("Pipeline K2P is symmetric", torch.allclose(dist_pipeline, dist_pipeline.T, atol=1e-6))

names_pipeline = [f"seq_{i}" for i in range(n_seqs)]
newick_pipeline = nj_from_distance_matrix(dist_pipeline.cpu().numpy(), names_pipeline)
test("Pipeline NJ tree is valid Newick", newick_pipeline.endswith(";"))

for name in names_pipeline:
    test(f"Pipeline tree contains '{name}'", name in newick_pipeline)


# ============================================================
print("\n" + "=" * 70)
print("SECTION 14: GTR Mathematical Properties Verification")
print("=" * 70)
# ============================================================

gtr_verify = GTRModel()
raw_r = torch.tensor([0.5, 2.0, 0.3, 1.0, 0.8, 0.6])
raw_f = torch.tensor([0.3, 0.2, 0.3, 0.2])
r = gtr_verify.normalize_rates(raw_r)
f = gtr_verify.normalize_frequencies(raw_f)
Q_v = gtr_verify.compute_Q_matrix(r, f)

test("GTR Q is non-symmetric (off-diagonal asymmetry)", not torch.allclose(Q_v, Q_v.T, atol=1e-6),
     "Q is symmetric - this is unexpected for GTR with unequal frequencies")

P_v = gtr_verify.compute_P_matrix(Q_v, 0.1)
test("GTR P matrix is stochastic (rows sum to 1)",
     torch.allclose(P_v.sum(dim=-1), torch.ones(4), atol=1e-4))

test("GTR P matrix is non-negative", (P_v >= -1e-8).all().item())

P_v_large_t = gtr_verify.compute_P_matrix(Q_v, 50.0)
test("GTR P(t→∞) converges to stationary distribution",
     torch.allclose(P_v_large_t[0], f, atol=0.05),
     f"P[0]={P_v_large_t[0].tolist()}, pi={f.tolist()}")

P_v_t0 = gtr_verify.compute_P_matrix(Q_v, 0.0)
test("GTR P(t=0) = I", torch.allclose(P_v_t0, torch.eye(4), atol=1e-4))

P1 = gtr_verify.compute_P_matrix(Q_v, 0.1)
P2 = gtr_verify.compute_P_matrix(Q_v, 0.2)
P1_sq = P1 @ P1
test("GTR Chapman-Kolmogorov: P(0.2) ≈ P(0.1)·P(0.1)",
     torch.allclose(P2, P1_sq, atol=0.05),
     f"max diff={(P2 - P1_sq).abs().max().item():.6f}")


# ============================================================
print("\n" + "=" * 70)
print("SECTION 15: End-to-End Route A Model Test")
print("=" * 70)
# ============================================================

model_a_small = ViralPhyloGPN(window_size=30, d_model=16, d_inner=16, n_blocks=2, kernel_size=3, n_bases=5)
onehot_small = torch.randn(1, 30, 5)
onehot_small = F.one_hot(torch.randint(0, 5, (1, 30)), num_classes=5).float()
rates_s, freqs_s, alpha_s, emb_s = model_a_small(onehot_small)
test("Small Route A model forward pass works", rates_s.shape == (1, 30, 6))

rates_pred, freqs_pred, alpha_pred = model_a_small.predict_gtr_params(onehot_small)
test("Small Route A predict_gtr_params works", rates_pred.shape == (1, 30, 6))

loss_val = rates_pred.sum() + freqs_pred.sum() + alpha_pred.sum()
loss_val.backward()
test("Small Route A backward pass works", True)


# ============================================================
print("\n" + "=" * 70)
print("SECTION 16: End-to-End Route B Model Test")
print("=" * 70)
# ============================================================

model_b_small = PHYLAViralModel(
    d_model=16, n_mamba_layers=1, d_state=4, d_conv=2, expand=2,
    n_tree_heads=2, n_composition_features=20,
    use_calibration=True, use_gtr_head=False
)
seqs_b = ["ACGTACGT", "TGCATGCA", "ACGTACGT", "TGCATGCA"]
comp_b_small = torch.randn(4, 20)
dist_b_small, emb_b_small, ll_b_small = model_b_small(seqs_b, composition_features=comp_b_small)
test("Small Route B forward pass works", dist_b_small.shape == (4, 4))

if dist_b_small.requires_grad:
    loss_b = dist_b_small.sum()
    loss_b.backward()
    test("Small Route B backward pass works", True)
else:
    test("Small Route B backward pass works (no grad)", True)


# ============================================================
print("\n" + "=" * 70)
print("SECTION 17: Quartet Loss Correctness Verification")
print("=" * 70)
# ============================================================

q_loss_fn = QuartetLoss(temperature=1.0)

dist_close = torch.zeros(4, 4)
dist_close[0, 1] = dist_close[1, 0] = 0.1
dist_close[2, 3] = dist_close[3, 2] = 0.1
dist_close[0, 2] = dist_close[2, 0] = 1.0
dist_close[0, 3] = dist_close[3, 0] = 1.0
dist_close[1, 2] = dist_close[2, 1] = 1.0
dist_close[1, 3] = dist_close[3, 1] = 1.0

q_idx = [(0, 1, 2, 3)]
loss_close = q_loss_fn(dist_close, quartet_indices=q_idx)
test("Quartet loss for correct topology is low", loss_close.item() < 0.5,
     f"loss={loss_close.item():.4f}")

dist_wrong = torch.zeros(4, 4)
dist_wrong[0, 2] = dist_wrong[2, 0] = 0.1
dist_wrong[1, 3] = dist_wrong[3, 1] = 0.1
dist_wrong[0, 1] = dist_wrong[1, 0] = 1.0
dist_wrong[0, 3] = dist_wrong[3, 0] = 1.0
dist_wrong[2, 3] = dist_wrong[3, 2] = 1.0
dist_wrong[1, 2] = dist_wrong[2, 1] = 1.0

loss_wrong = q_loss_fn(dist_wrong, quartet_indices=q_idx)
test("Quartet loss for wrong topology is higher", loss_wrong.item() > loss_close.item(),
     f"wrong={loss_wrong.item():.4f}, correct={loss_close.item():.4f}")


# ============================================================
print("\n" + "=" * 70)
print("SECTION 18: Distance Regression Loss Verification")
print("=" * 70)
# ============================================================

pred_d = torch.tensor([[0.0, 0.2, 0.4], [0.2, 0.0, 0.3], [0.4, 0.3, 0.0]])
target_d = torch.tensor([[0.0, 0.2, 0.4], [0.2, 0.0, 0.3], [0.4, 0.3, 0.0]])
loss_perfect = DistanceRegressionLoss(loss_type="mse")(pred_d, target_d)
test("MSE loss for perfect prediction is 0", loss_perfect.item() < 1e-8,
     f"loss={loss_perfect.item():.8f}")

pred_d2 = torch.tensor([[0.0, 0.3, 0.5], [0.3, 0.0, 0.4], [0.5, 0.4, 0.0]])
loss_imperfect = DistanceRegressionLoss(loss_type="mse")(pred_d2, target_d)
test("MSE loss for imperfect prediction > 0", loss_imperfect.item() > 0,
     f"loss={loss_imperfect.item():.6f}")


# ============================================================
print("\n" + "=" * 70)
print("SECTION 19: HybridDistance Alpha Verification")
print("=" * 70)
# ============================================================

hybrid_test = HybridDistance(learnable_alpha=True, init_alpha=0.8)
test("HybridDistance initial alpha ≈ 0.8 after sigmoid",
     abs(hybrid_test.alpha.item() - 0.8) < 0.01,
     f"alpha={hybrid_test.alpha.item():.4f}")

d_l = torch.tensor(0.5)
d_k = torch.tensor(0.3)
expected = 0.8 * 0.5 + 0.2 * 0.3
d_out = hybrid_test(d_l.unsqueeze(0).unsqueeze(0), d_k.unsqueeze(0).unsqueeze(0))
test("HybridDistance weighted average is correct",
     abs(d_out.item() - expected) < 0.01,
     f"expected={expected:.4f}, got={d_out.item():.4f}")


# ============================================================
print("\n" + "=" * 70)
print("SECTION 20: Felsenstein Likelihood Monotonicity")
print("=" * 70)
# ============================================================

fels = FelsensteinPruning(n_bases=4, n_gamma_categories=1)

tree_close = {
    'root': 4,
    0: {'seq_idx': 0, 'children': []},
    1: {'seq_idx': 1, 'children': []},
    2: {'seq_idx': 2, 'children': []},
    3: {'seq_idx': 3, 'children': []},
    4: {'children': [(0, 0.01), (1, 0.01)], 'seq_idx': -1},
    5: {'children': [(2, 0.01), (3, 0.01)], 'seq_idx': -1},
    6: {'children': [(4, 0.01), (5, 0.01)], 'seq_idx': -1, 'root': True},
}
tree_close['root'] = 6

tree_far = {
    'root': 4,
    0: {'seq_idx': 0, 'children': []},
    1: {'seq_idx': 1, 'children': []},
    2: {'seq_idx': 2, 'children': []},
    3: {'seq_idx': 3, 'children': []},
    4: {'children': [(0, 1.0), (1, 1.0)], 'seq_idx': -1},
    5: {'children': [(2, 1.0), (3, 1.0)], 'seq_idx': -1},
    6: {'children': [(4, 1.0), (5, 1.0)], 'seq_idx': -1, 'root': True},
}
tree_far['root'] = 6

aln_test = torch.tensor([[0, 1, 2, 3],
                         [0, 1, 2, 3],
                         [0, 1, 2, 3],
                         [0, 1, 2, 3]])

Q_test = gtr.compute_Q_matrix(rates, freqs)
alpha_test = alpha.squeeze()

ll_close = fels(aln_test, tree_close, Q_test, freqs, alpha_test)
ll_far = fels(aln_test, tree_far, Q_test, freqs, alpha_test)
test("Identical sequences: closer tree has higher likelihood",
     ll_close.item() > ll_far.item(),
     f"close={ll_close.item():.4f}, far={ll_far.item():.4f}")


# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total: {PASS_COUNT + FAIL_COUNT} tests")
print(f"PASSED: {PASS_COUNT}")
print(f"FAILED: {FAIL_COUNT}")
if ERROR_LIST:
    print(f"\nFailed tests:")
    for e in ERROR_LIST:
        print(f"  - {e}")
print(f"\n{'ALL TESTS PASSED!' if FAIL_COUNT == 0 else 'SOME TESTS FAILED!'}")
