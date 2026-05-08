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

P_large = gtr.compute_P_matrix(Q, 100.0)
test("P(t=100) approaches stationary frequencies",
     torch.allclose(P_large[0], freqs, atol=0.05),
     f"P[0]={P_large[0].tolist()}, pi={freqs.tolist()}")

gtr_head = GTRParameterHead(embed_dim=256)
embeddings = torch.randn(8, 256)
out_rates, out_freq, out_alpha = gtr_head(embeddings)
test("GTRParameterHead output shapes", out_rates.shape == (8, 6) and out_freq.shape == (8, 4) and out_alpha.shape == (8,))

norm_r = gtr_head.normalize_rates_fn(out_rates)
test("GTRParameterHead normalize_rates_fn produces positive rates",
     (norm_r > 0).all().item(),
     f"min={norm_r.min().item():.4f}")

norm_f = gtr_head.normalize_frequencies_fn(out_freq)
test("GTRParameterHead normalize_frequencies_fn sums to 1",
     torch.isclose(norm_f.sum(dim=-1), torch.tensor(1.0), atol=1e-4).all().item())


# ============================================================
print("\n" + "=" * 70)
print("SECTION 2: Newick Parser & Felsenstein Pruning")
print("=" * 70)
# ============================================================

from src.models.route_a.felsenstein import FelsensteinPruning, newick_to_tree_structure, topological_order

newick_str = "((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);"
tree_struct = newick_to_tree_structure(newick_str, leaf_names=["A", "B", "C", "D"])
test("Newick parser produces tree_structure dict", isinstance(tree_struct, dict))
test("Newick parser has 'root' key", 'root' in tree_struct)

n_leaves = sum(1 for k, v in tree_struct.items() if isinstance(v, dict) and len(v.get('children', [])) == 0)
test("Newick parser finds 4 leaves", n_leaves == 4, f"found {n_leaves}")

traversal = topological_order(tree_struct)
test("Topological order is valid list", isinstance(traversal, list) and len(traversal) > 0)
test("Topological order ends with root", traversal[-1] == tree_struct['root'])

felsenstein = FelsensteinPruning(n_bases=4, n_gamma_categories=1)

alignment = torch.tensor([[0, 1, 2, 3],
                          [0, 1, 2, 3],
                          [0, 1, 2, 3],
                          [0, 1, 2, 3]])

ll = felsenstein(alignment, tree_struct, Q, freqs, alpha.squeeze())
test("Felsenstein log-likelihood is finite", torch.isfinite(ll).item(), f"ll={ll.item()}")
test("Felsenstein log-likelihood is negative", ll.item() < 0, f"ll={ll.item()}")

newick_close = "((A:0.01,B:0.01):0.01,(C:0.01,D:0.01):0.01);"
tree_close = newick_to_tree_structure(newick_close, leaf_names=["A", "B", "C", "D"])
ll_close = felsenstein(alignment, tree_close, Q, freqs, alpha.squeeze())

newick_far = "((A:2.0,B:2.0):2.0,(C:2.0,D:2.0):2.0);"
tree_far = newick_to_tree_structure(newick_far, leaf_names=["A", "B", "C", "D"])
ll_far = felsenstein(alignment, tree_far, Q, freqs, alpha.squeeze())

test("Closer tree has higher likelihood for identical sequences",
     ll_close.item() > ll_far.item(),
     f"close={ll_close.item():.4f}, far={ll_far.item():.4f}")


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

k2p = K2PDistance()
sequences = ["ACGTACGTACGT", "ACGTACGTTCGT", "ACGTACGTACGA", "TCGTACGTACGT"]
dist = k2p.compute(sequences)
test("K2P matrix is square", dist.shape == (4, 4))
test("K2P matrix diagonal is zero", torch.allclose(dist.diag(), torch.zeros(4), atol=1e-6))
test("K2P matrix is symmetric", torch.allclose(dist, dist.T, atol=1e-6))
test("K2P matrix all non-negative", (dist >= 0).all().item())

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
embeddings_t = torch.randn(16, 64)
zca.eval()
whitened = zca(embeddings_t)
test("ZCA output shape matches input", whitened.shape == embeddings_t.shape)

debias = CompositionDebias(embed_dim=64, n_composition_features=20)
comp_features = torch.randn(16, 20)
debiased = debias(embeddings_t, comp_features)
test("CompositionDebias output shape matches input", debiased.shape == embeddings_t.shape)

site_w = SiteWeighting(embed_dim=64)
weighted, weights = site_w(embeddings_t)
test("SiteWeighting output shape matches input", weighted.shape == embeddings_t.shape)
test("SiteWeighting weights in [0,1]", (weights >= 0).all().item() and (weights <= 1).all().item())

calibration = EmbeddingCalibration(embed_dim=64, n_composition_features=20)
calibrated, site_weights = calibration(embeddings_t, comp_features)
test("EmbeddingCalibration output shape matches input", calibrated.shape == embeddings_t.shape)

calibration_none = EmbeddingCalibration(embed_dim=64, n_composition_features=20,
                                         use_zca=False, use_debias=False, use_site_weight=False)
calibrated_none, sw_none = calibration_none(embeddings_t, comp_features)
test("EmbeddingCalibration all disabled returns input unchanged",
     torch.allclose(calibrated_none, embeddings_t),
     f"max diff={(calibrated_none - embeddings_t).abs().max().item():.6f}")


# ============================================================
print("\n" + "=" * 70)
print("SECTION 5: Distance Heads & HybridDistance")
print("=" * 70)
# ============================================================

from src.models.distance.distance_head import DistanceHead, EuclideanDistanceHead, CosineDistanceHead
from src.models.distance.hybrid_distance import HybridDistance, AdaptiveHybridDistance

dist_head = DistanceHead(embed_dim=64, hidden_dim=128)
emb = torch.randn(8, 64)
pairwise = dist_head.pairwise_distances(emb)
test("DistanceHead pairwise output is NxN", pairwise.shape == (8, 8))
test("DistanceHead all non-negative", (pairwise >= 0).all().item())

euc_head = EuclideanDistanceHead()
euc_dist = euc_head.pairwise_distances(emb)
test("Euclidean distance diagonal is zero", torch.allclose(euc_dist.diag(), torch.zeros(8), atol=1e-5))
test("Euclidean distance is symmetric", torch.allclose(euc_dist, euc_dist.T, atol=1e-5))

hybrid = HybridDistance(learnable_alpha=True, init_alpha=0.8)
test("HybridDistance initial alpha ≈ 0.8",
     abs(hybrid.alpha.item() - 0.8) < 0.01,
     f"alpha={hybrid.alpha.item():.4f}")

d_llm = torch.rand(4, 4) * 0.5
d_llm = (d_llm + d_llm.T) / 2
d_k2p = torch.rand(4, 4) * 0.3
d_k2p = (d_k2p + d_k2p.T) / 2
d_hybrid = hybrid(d_llm, d_k2p)
test("HybridDistance output shape matches input", d_hybrid.shape == d_llm.shape)


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


# ============================================================
print("\n" + "=" * 70)
print("SECTION 7: Tree Metrics")
print("=" * 70)
# ============================================================

from src.models.tree.tree_metrics import compute_rf_distance, compute_quartet_accuracy, compute_branch_length_correlation, TreeMetrics

tree1 = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
tree2 = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
rf, nrf = compute_rf_distance(tree1, tree2)
test("RF distance of identical trees is 0", rf == 0, f"rf={rf}")

tree3 = "((A:0.1,C:0.1):0.1,(B:0.1,D:0.1):0.1);"
rf2, nrf2 = compute_rf_distance(tree1, tree3)
test("RF distance of different topologies > 0", rf2 > 0, f"rf={rf2}")

qa_same = compute_quartet_accuracy(tree1, tree2, n_quartets=10)
test("QA of identical trees is 1.0", qa_same == 1.0, f"qa={qa_same}")

metrics = TreeMetrics()
result = metrics.evaluate(tree1, tree2, "test")
test("TreeMetrics.evaluate returns expected keys",
     "rf" in result and "nrf" in result and "qa" in result)


# ============================================================
print("\n" + "=" * 70)
print("SECTION 8: Loss Functions (with quartet_topologies)")
print("=" * 70)
# ============================================================

from src.training.losses import QuartetLoss, DistanceRegressionLoss, PhyloLikelihoodLoss, TripleLoss, LossWeightScheduler, get_quartet_topology_from_tree

quartet_loss = QuartetLoss(temperature=1.0)
dist_mat = torch.rand(8, 8)
dist_mat = (dist_mat + dist_mat.T) / 2
quartet_indices = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 2, 4, 6)]
quartet_topos = [0, 0, 0]
q_loss = quartet_loss(dist_mat, quartet_indices=quartet_indices, quartet_topologies=quartet_topos)
test("QuartetLoss with topologies returns scalar", q_loss.dim() == 0)
test("QuartetLoss with topologies is finite", torch.isfinite(q_loss).item())
test("QuartetLoss with topologies is non-negative", q_loss.item() >= 0, f"loss={q_loss.item():.4f}")

dist_close = torch.zeros(4, 4)
dist_close[0, 1] = dist_close[1, 0] = 0.1
dist_close[2, 3] = dist_close[3, 2] = 0.1
dist_close[0, 2] = dist_close[2, 0] = 1.0
dist_close[0, 3] = dist_close[3, 0] = 1.0
dist_close[1, 2] = dist_close[2, 1] = 1.0
dist_close[1, 3] = dist_close[3, 1] = 1.0
loss_close = quartet_loss(dist_close, quartet_indices=[(0, 1, 2, 3)], quartet_topologies=[0])
test("Quartet loss for correct topology is low", loss_close.item() < 0.5,
     f"loss={loss_close.item():.4f}")

dist_wrong = torch.zeros(4, 4)
dist_wrong[0, 2] = dist_wrong[2, 0] = 0.1
dist_wrong[1, 3] = dist_wrong[3, 1] = 0.1
dist_wrong[0, 1] = dist_wrong[1, 0] = 1.0
dist_wrong[0, 3] = dist_wrong[3, 0] = 1.0
dist_wrong[2, 3] = dist_wrong[3, 2] = 1.0
dist_wrong[1, 2] = dist_wrong[2, 1] = 1.0
loss_wrong = quartet_loss(dist_wrong, quartet_indices=[(0, 1, 2, 3)], quartet_topologies=[0])
test("Quartet loss for wrong topology is higher", loss_wrong.item() > loss_close.item(),
     f"wrong={loss_wrong.item():.4f}, correct={loss_close.item():.4f}")

topo_0 = get_quartet_topology_from_tree("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);", ["A", "B", "C", "D"])
test("Quartet topology from tree: (A,B|C,D) -> 0", topo_0 == 0, f"topo={topo_0}")

dist_reg_loss = DistanceRegressionLoss(loss_type="huber")
pred = torch.rand(4, 4)
target = torch.rand(4, 4)
d_loss = dist_reg_loss(pred, target)
test("DistanceRegressionLoss returns scalar", d_loss.dim() == 0)

phylo_loss = PhyloLikelihoodLoss()
p_loss = phylo_loss(torch.tensor(-5.0))
test("PhyloLikelihoodLoss = -log_likelihood", p_loss.item() == 5.0)

triple = TripleLoss(alpha=1.0, beta=0.5, gamma=0.5)
target_dist_8 = torch.rand(8, 8)
t_loss = triple(dist_mat, target_dist=target_dist_8, quartet_indices=quartet_indices,
                quartet_topologies=quartet_topos)
test("TripleLoss with topologies returns scalar", t_loss.dim() == 0)
test("TripleLoss with topologies is finite", torch.isfinite(t_loss).item())

scheduler = LossWeightScheduler(total_epochs=100, schedule_type="phased")
a1, b1, g1 = scheduler.get_weights(0)
test("Phased schedule early: alpha=1.0", a1 == 1.0)
test("Phased schedule early: beta=0.0", b1 == 0.0)

a2, b2, g2 = scheduler.get_weights(80)
test("Phased schedule late: beta=0.7", b2 == 0.7)


# ============================================================
print("\n" + "=" * 70)
print("SECTION 9: ViralPhyloGPN (Route A)")
print("=" * 70)
# ============================================================

from src.models.route_a.viral_phylogpn import ViralPhyloGPN, RCEConv1d, ByteNetBlock, DilatedByteNet

rce = RCEConv1d(in_channels=5, out_channels=32, kernel_size=3, padding='same')
x_rce = torch.randn(2, 5, 50)
out_rce = rce(x_rce)
test("RCEConv1d output shape", out_rce.shape == (2, 32, 50), f"shape={out_rce.shape}")

block = ByteNetBlock(d_model=32, d_inner=64, kernel_size=3)
x_block = torch.randn(2, 50, 32)
out_block = block(x_block)
test("ByteNetBlock output shape", out_block.shape == (2, 50, 32))

model_a = ViralPhyloGPN(window_size=50, d_model=32, d_inner=32, n_blocks=2, kernel_size=3, n_bases=5)
onehot = F.one_hot(torch.randint(0, 5, (2, 50)), num_classes=5).float()
raw_rates, raw_freq, raw_alpha, site_emb = model_a(onehot)
test("ViralPhyloGPN output shapes",
     raw_rates.shape == (2, 50, 6) and raw_freq.shape == (2, 50, 4) and raw_alpha.shape == (2, 50))

rates_a, freqs_a, alpha_a = model_a.predict_gtr_params(onehot)
test("ViralPhyloGPN predict_gtr_params rates sum to 6",
     torch.isclose(rates_a.sum(dim=-1), torch.tensor(6.0), atol=1e-3).all().item())
test("ViralPhyloGPN predict_gtr_params freqs sum to 1",
     torch.isclose(freqs_a.sum(dim=-1), torch.tensor(1.0), atol=1e-3).all().item())

loss_val = rates_a.sum() + freqs_a.sum() + alpha_a.sum()
loss_val.backward()
test("ViralPhyloGPN backward pass works", True)


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
    use_calibration=True, use_gtr_head=False
)
sequences_b = ["ACGTACGTACGT", "TGCATGCATGCA", "ACGTACGTACGT", "TGCATGCATGCA"]
comp_b = torch.randn(4, 20)
dist_b, emb_b, ll_b = model_b(sequences_b, composition_features=comp_b)
test("PHYLAViralModel distance matrix shape", dist_b.shape == (4, 4), f"shape={dist_b.shape}")
test("PHYLAViralModel embeddings shape", emb_b.shape == (4, 32), f"shape={emb_b.shape}")
test("PHYLAViralModel without ref_tree returns None ll", ll_b is None)

newick_test = "((seq0:0.1,seq1:0.1):0.1,(seq2:0.1,seq3:0.1):0.1);"
dist_b2, emb_b2, ll_b2 = model_b(sequences_b, composition_features=comp_b, ref_tree_newick=newick_test)
test("PHYLAViralModel with ref_tree returns ll", ll_b2 is not None or not model_b.use_gtr_head)


# ============================================================
print("\n" + "=" * 70)
print("SECTION 12: CompositionFeatureExtractor")
print("=" * 70)
# ============================================================

from src.data.viral_dataset import CompositionFeatureExtractor

extractor = CompositionFeatureExtractor(k=4)
seq = "ACGTACGTACGTACGT"
feat = extractor.extract(seq)
test("CompositionFeatureExtractor output has 20 features", feat.shape == (20,), f"shape={feat.shape}")

batch_seqs = ["ACGTACGTACGT", "TGCATGCATGCA", "AAAACCCCGGGG"]
batch_feat = extractor.extract_batch(batch_seqs)
test("CompositionFeatureExtractor batch output shape", batch_feat.shape == (3, 20))

gc_feat = extractor.extract("GCGCGCGC")
at_feat = extractor.extract("ATATATAT")
test("GC content feature > AT content feature for GC-rich seq",
     gc_feat[0] > at_feat[0],
     f"GC={gc_feat[0]:.4f}, AT={at_feat[0]:.4f}")


# ============================================================
print("\n" + "=" * 70)
print("SECTION 13: Integration - K2P + NJ Pipeline")
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

names_pipeline = [f"seq_{i}" for i in range(n_seqs)]
newick_pipeline = nj_from_distance_matrix(dist_pipeline.cpu().numpy(), names_pipeline)
test("Pipeline NJ tree is valid Newick", newick_pipeline.endswith(";"))


# ============================================================
print("\n" + "=" * 70)
print("SECTION 14: GTR Chapman-Kolmogorov Verification")
print("=" * 70)
# ============================================================

gtr_verify = GTRModel()
raw_r = torch.tensor([0.5, 2.0, 0.3, 1.0, 0.8, 0.6])
raw_f = torch.tensor([0.3, 0.2, 0.3, 0.2])
r = gtr_verify.normalize_rates(raw_r)
f = gtr_verify.normalize_frequencies(raw_f)
Q_v = gtr_verify.compute_Q_matrix(r, f)

P1 = gtr_verify.compute_P_matrix(Q_v, 0.1)
P2 = gtr_verify.compute_P_matrix(Q_v, 0.2)
P1_sq = P1 @ P1
test("GTR Chapman-Kolmogorov: P(0.2) ≈ P(0.1)·P(0.1)",
     torch.allclose(P2, P1_sq, atol=0.05),
     f"max diff={(P2 - P1_sq).abs().max().item():.6f}")

P_v_large_t = gtr_verify.compute_P_matrix(Q_v, 50.0)
test("GTR P(t→∞) converges to stationary distribution",
     torch.allclose(P_v_large_t[0], f, atol=0.05),
     f"P[0]={P_v_large_t[0].tolist()}, pi={f.tolist()}")


# ============================================================
print("\n" + "=" * 70)
print("SECTION 15: End-to-End Route A with Felsenstein")
print("=" * 70)
# ============================================================

model_a_small = ViralPhyloGPN(window_size=30, d_model=16, d_inner=16, n_blocks=2, kernel_size=3, n_bases=5)
onehot_small = F.one_hot(torch.randint(0, 5, (1, 30)), num_classes=5).float()
rates_s, freqs_s, alpha_s, emb_s = model_a_small(onehot_small)

gtr_small = GTRModel()
rates_pred = gtr_small.normalize_rates(rates_s.mean(dim=1))
freqs_pred = gtr_small.normalize_frequencies(freqs_s.mean(dim=1))
alpha_pred = F.softplus(alpha_s.mean(dim=1)).clamp(min=0.1, max=10.0)

gtr_small = GTRModel()
Q_small = gtr_small.compute_Q_matrix(rates_pred[0], freqs_pred[0])

newick_a_test = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
tree_a_test = newick_to_tree_structure(newick_a_test, leaf_names=["A", "B", "C", "D"])

alignment_a = torch.tensor([[0, 1, 2, 3],
                             [0, 1, 2, 3],
                             [0, 1, 2, 3],
                             [0, 1, 2, 3]])

fels_small = FelsensteinPruning(n_bases=4, n_gamma_categories=1)
ll_a = fels_small(alignment_a, tree_a_test, Q_small, freqs_pred[0], alpha_pred[0])
test("Route A Felsenstein end-to-end: ll is finite", torch.isfinite(ll_a).item(), f"ll={ll_a.item()}")
test("Route A Felsenstein end-to-end: ll is negative", ll_a.item() < 0, f"ll={ll_a.item()}")


# ============================================================
print("\n" + "=" * 70)
print("SECTION 16: End-to-End Route B with Felsenstein")
print("=" * 70)
# ============================================================

model_b_small = PHYLAViralModel(
    d_model=16, n_mamba_layers=1, d_state=4, d_conv=2, expand=2,
    n_tree_heads=2, n_composition_features=20,
    use_calibration=True, use_gtr_head=True
)
seqs_b = ["ACGTACGTACGT", "TGCATGCATGCA", "ACGTACGTACGT", "TGCATGCATGCA"]
comp_b_small = torch.randn(4, 20)
newick_b_test = "((seq0:0.1,seq1:0.1):0.1,(seq2:0.1,seq3:0.1):0.1);"
dist_b_small, emb_b_small, ll_b_small = model_b_small(
    seqs_b, composition_features=comp_b_small, ref_tree_newick=newick_b_test)
test("Route B with GTR head + ref_tree: distance matrix shape", dist_b_small.shape == (4, 4))
test("Route B with GTR head + ref_tree: ll is not None", ll_b_small is not None,
     "ll is None - GTR head may have failed silently")


# ============================================================
print("\n" + "=" * 70)
print("SECTION 17: Quartet Topology from Reference Tree")
print("=" * 70)
# ============================================================

tree_ab_cd = "((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);"
topo_ab_cd = get_quartet_topology_from_tree(tree_ab_cd, ["A", "B", "C", "D"])
test("Topology (A,B|C,D) -> 0", topo_ab_cd == 0, f"topo={topo_ab_cd}")

tree_ac_bd = "((A:0.1,C:0.1):0.2,(B:0.1,D:0.1):0.2);"
topo_ac_bd = get_quartet_topology_from_tree(tree_ac_bd, ["A", "B", "C", "D"])
test("Topology (A,C|B,D) -> 1", topo_ac_bd == 1, f"topo={topo_ac_bd}")

tree_ad_bc = "((A:0.1,D:0.1):0.2,(B:0.1,C:0.1):0.2);"
topo_ad_bc = get_quartet_topology_from_tree(tree_ad_bc, ["A", "B", "C", "D"])
test("Topology (A,D|B,C) -> 2", topo_ad_bc == 2, f"topo={topo_ad_bc}")


# ============================================================
print("\n" + "=" * 70)
print("SECTION 18: Route C Model (without backbone download)")
print("=" * 70)
# ============================================================

from src.models.route_c.route_c_model import RouteCModel

class MockBackbone(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, sequences, max_length=2048):
        n = len(sequences)
        cls_emb = torch.randn(n, self.embed_dim)
        pos_emb = torch.randn(n, 10, self.embed_dim)
        return cls_emb, pos_emb

mock_backbone = MockBackbone(embed_dim=128)
route_c = RouteCModel(
    backbone=mock_backbone,
    embed_dim=128,
    n_composition_features=20,
    use_calibration=True,
    use_hybrid_distance=True,
)
test("RouteCModel created with mock backbone", True)

seqs_c = ["ACGTACGTACGT", "TGCATGCATGCA", "ACGTACGTACGT", "TGCATGCATGCA"]
comp_c = torch.randn(4, 20)
dist_c = route_c(seqs_c, composition_features=comp_c)
test("RouteCModel forward produces distance matrix", dist_c.shape == (4, 4), f"shape={dist_c.shape}")
test("RouteCModel distance matrix has non-negative entries", (dist_c >= -1e-4).all().item())


# ============================================================
print("\n" + "=" * 70)
print("SECTION 19: Data Pipeline Integration")
print("=" * 70)
# ============================================================

import tempfile
import os
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

tmpdir = tempfile.mkdtemp()
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(tmpdir, "alignments", split), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, "trees", split), exist_ok=True)

records = []
for i in range(8):
    seq = ''.join(np.random.choice(['A', 'C', 'G', 'T'], size=100))
    records.append(SeqRecord(Seq(seq), id=f"seq_{i}", description=""))

from Bio import SeqIO
aln_path = os.path.join(tmpdir, "alignments", "train", "test_sample.fasta")
SeqIO.write(records, aln_path, "fasta")

tree_str = "((seq_0:0.1,seq_1:0.1):0.2,(seq_2:0.1,seq_3:0.1):0.2,((seq_4:0.1,seq_5:0.1):0.1,(seq_6:0.1,seq_7:0.1):0.1):0.1);"
tree_path = os.path.join(tmpdir, "trees", "train", "test_sample.nwk")
with open(tree_path, 'w') as f:
    f.write(tree_str)

from src.data.phylo_dataset import PhyloTrainingDataset
dataset = PhyloTrainingDataset(data_dir=tmpdir, split="train", n_quartets_per_sample=10)
test("PhyloTrainingDataset loads data", len(dataset) > 0, f"len={len(dataset)}")

if len(dataset) > 0:
    sample = dataset[0]
    test("Dataset returns sequences", "sequences" in sample)
    test("Dataset returns names", "names" in sample)
    test("Dataset returns composition_features", "composition_features" in sample)
    test("Dataset returns encoded_seqs", "encoded_seqs" in sample)
    test("Dataset returns k2p_distance", "k2p_distance" in sample)
    test("Dataset returns quartet_indices", "quartet_indices" in sample)
    test("Dataset returns quartet_topologies", "quartet_topologies" in sample)
    test("Dataset returns ref_tree", "ref_tree" in sample)
    test("Dataset quartet_topologies is list", isinstance(sample["quartet_topologies"], list))
    test("Dataset quartet_topologies length matches quartet_indices",
         len(sample["quartet_topologies"]) == len(sample["quartet_indices"]),
         f"topos={len(sample['quartet_topologies'])}, indices={len(sample['quartet_indices'])}")

import shutil
shutil.rmtree(tmpdir)


# ============================================================
print("\n" + "=" * 70)
print("SECTION 20: Config Files Validation")
print("=" * 70)
# ============================================================

import yaml

for config_path in [
    "/workspace/configs/train/route_a_pretrain.yaml",
    "/workspace/configs/train/route_b_dual.yaml",
    "/workspace/configs/train/route_c_lora.yaml",
]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    test(f"{os.path.basename(config_path)} is valid YAML", cfg is not None)
    test(f"{os.path.basename(config_path)} has data_dir", "data_dir" in cfg,
         f"keys={list(cfg.keys())}")

for model_config in [
    "/workspace/configs/model/route_a_phylogpn.yaml",
    "/workspace/configs/model/route_b_phyla_viral.yaml",
    "/workspace/configs/model/route_c_dnabert2.yaml",
    "/workspace/configs/model/route_c_nt500m.yaml",
]:
    with open(model_config) as f:
        cfg = yaml.safe_load(f)
    test(f"{os.path.basename(model_config)} is valid YAML", cfg is not None)


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
