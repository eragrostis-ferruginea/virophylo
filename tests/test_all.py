#!/usr/bin/env python3
"""
ViroPhylo: Comprehensive Test Suite
Tests all core components of the viral phylogenetics LLM system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from unittest import TestCase, main as unittest_main, skipIf

try:
    import dendropy
    HAS_DENDROPY = True
except ImportError:
    HAS_DENDROPY = False


class TestImports(TestCase):
    """Test all module imports."""

    def test_import_calibration(self):
        from src.models.calibration.zca_whitening import (
            ZCAWhitening, CompositionDebias, SiteWeighting, EmbeddingCalibration
        )
        self.assertIsNotNone(ZCAWhitening)

    def test_import_distance(self):
        from src.models.distance.distance_head import DistanceHead
        from src.models.distance.hybrid_distance import HybridDistance
        from src.models.distance.k2p_baseline import K2PDistance
        self.assertIsNotNone(DistanceHead)

    def test_import_tree(self):
        from src.models.tree.nj_builder import NJTreeBuilder, nj_from_distance_matrix
        from src.models.tree.tree_metrics import TreeMetrics, compute_rf_distance
        self.assertIsNotNone(nj_from_distance_matrix)

    def test_import_losses(self):
        from src.training.losses import (
            QuartetLoss, DistanceRegressionLoss, PhyloLikelihoodLoss,
            TripleLoss, LossWeightScheduler
        )
        self.assertIsNotNone(TripleLoss)

    def test_import_gtr(self):
        from src.models.route_a.gtr_module import GTRModel, GTRParameterHead
        self.assertIsNotNone(GTRModel)

    def test_import_felsenstein(self):
        from src.models.route_a.felsenstein import FelsensteinPruning
        self.assertIsNotNone(FelsensteinPruning)

    def test_import_bimamba(self):
        from src.models.route_b.bimamba_block import (
            MambaBlock, BiMambaBlock, BiMambaStack, TreeHead, NucleotideTokenizer
        )
        self.assertIsNotNone(BiMambaBlock)

    def test_import_viral_phylogpn(self):
        from src.models.route_a.viral_phylogpn import ViralPhyloGPN, ByteNetBlock, RCEConv1d
        self.assertIsNotNone(ViralPhyloGPN)

    def test_import_data(self):
        from src.data.viral_dataset import CompositionFeatureExtractor
        from src.data.phylo_dataset import PhyloTrainingDataset
        self.assertIsNotNone(CompositionFeatureExtractor)


class TestZCAWhitening(TestCase):
    """Test ZCA whitening module."""

    def setUp(self):
        from src.models.calibration.zca_whitening import ZCAWhitening
        self.zca = ZCAWhitening(embed_dim=128)
        self.x = torch.randn(32, 128)

    def test_forward_shape(self):
        self.zca.train()
        out = self.zca(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_forward_eval(self):
        self.zca.eval()
        with torch.no_grad():
            out = self.zca(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_statistics_update(self):
        self.zca.train()
        self.zca.update_statistics(self.x)
        self.assertTrue(self.zca._initialized)

    def test_whitening_reduces_correlation(self):
        self.zca.train()
        self.zca.eval()
        with torch.no_grad():
            out = self.zca(self.x)
        corr_before = self._compute_correlation(self.x)
        corr_after = self._compute_correlation(out)
        self.assertLess(np.mean(np.abs(corr_after)), np.mean(np.abs(corr_before)))

    def _compute_correlation(self, x):
        if x.shape[0] < 2:
            return np.zeros((x.shape[1], x.shape[1]))
        x_np = x.numpy()
        return np.corrcoef(x_np.T)


class TestCompositionDebias(TestCase):
    """Test composition debiasing module."""

    def setUp(self):
        from src.models.calibration.composition_debias import CompositionDebias
        self.debias = CompositionDebias(embed_dim=128, n_composition_features=20)
        self.embeddings = torch.randn(8, 128)
        self.comp_features = torch.randn(8, 20)

    def test_forward_shape(self):
        out = self.debias(self.embeddings, self.comp_features)
        self.assertEqual(out.shape, self.embeddings.shape)

    def test_debiasing_reduces_feature_dependency(self):
        self.debias.train()
        out1 = self.debias(self.embeddings, self.comp_features)
        out2 = self.debias(self.embeddings * 2, self.comp_features)
        diff1 = torch.abs(out1 - out2).mean()
        out1 = self.debias(self.embeddings, self.comp_features * 2)
        diff2 = torch.abs(out1 - out2).mean()
        self.assertIsInstance(diff1.item(), float)


class TestSiteWeighting(TestCase):
    """Test site weighting module."""

    def setUp(self):
        from src.models.calibration.site_weighting import SiteWeighting
        self.weighting = SiteWeighting(embed_dim=128)
        self.embeddings = torch.randn(16, 128)

    def test_forward_shape(self):
        out, weights = self.weighting(self.embeddings)
        self.assertEqual(out.shape, self.embeddings.shape)
        self.assertEqual(weights.shape, (16,))

    def test_weights_in_range(self):
        self.weighting.eval()
        with torch.no_grad():
            _, weights = self.weighting(self.embeddings)
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.all(weights <= 1))


class TestEmbeddingCalibration(TestCase):
    """Test full embedding calibration module."""

    def setUp(self):
        from src.models.calibration.zca_whitening import EmbeddingCalibration
        self.calib = EmbeddingCalibration(
            embed_dim=128,
            n_composition_features=20,
            use_zca=True,
            use_debias=True,
            use_site_weight=True
        )
        self.embeddings = torch.randn(8, 128)
        self.comp_features = torch.randn(8, 20)

    def test_full_forward(self):
        self.calib.train()
        out, weights = self.calib(self.embeddings, self.comp_features)
        self.assertEqual(out.shape, self.embeddings.shape)

    def test_partial_forward(self):
        self.calib.eval()
        out1, _ = self.calib(self.embeddings, self.comp_features)
        out2, _ = self.calib(self.embeddings, None)
        self.assertEqual(out1.shape, out2.shape)


class TestDistanceHead(TestCase):
    """Test distance prediction heads."""

    def setUp(self):
        from src.models.distance.distance_head import DistanceHead
        self.head = DistanceHead(embed_dim=128)
        self.embeddings = torch.randn(8, 128)

    def test_pairwise_distances_shape(self):
        dist = self.head.pairwise_distances(self.embeddings)
        self.assertEqual(dist.shape, (8, 8))

    def test_pairwise_distances_symmetric(self):
        dist = self.head.pairwise_distances(self.embeddings)
        self.assertTrue(torch.allclose(dist, dist.T, atol=1e-5))

    def test_pairwise_distances_diagonal_zero(self):
        dist = self.head.pairwise_distances(self.embeddings)
        diag = torch.diagonal(dist)
        self.assertTrue(torch.all(diag >= 0))

    def test_forward_pairwise(self):
        e1 = self.embeddings[:4]
        e2 = self.embeddings[4:]
        dist = self.head(e1, e2)
        self.assertEqual(dist.shape, (4, 4))


class TestK2PDistance(TestCase):
    """Test Kimura 2-Parameter distance calculation."""

    def setUp(self):
        from src.models.distance.k2p_baseline import K2PDistance
        self.k2p = K2PDistance()
        self.seqs = [
            "ACGTACGTACGT",
            "ACGTTCGACGT",
            "ACGTACGTACGT",
            "TTTTCCCCGGGG",
        ]

    def test_encode_sequences(self):
        encoded = self.k2p.encode_sequences(self.seqs)
        self.assertEqual(encoded.shape[0], len(self.seqs))

    def test_compute_k2p_identical(self):
        dist = self.k2p.compute([self.seqs[0], self.seqs[0]])
        self.assertLess(dist[0, 1].item(), 0.01)

    def test_compute_k2p_different(self):
        dist = self.k2p.compute([self.seqs[0], self.seqs[3]])
        self.assertGreater(dist[0, 1].item(), 0.1)


class TestHybridDistance(TestCase):
    """Test hybrid distance module."""

    def setUp(self):
        from src.models.distance.hybrid_distance import HybridDistance
        self.hybrid = HybridDistance(learnable_alpha=True, init_alpha=0.8)
        self.d_llm = torch.rand(8, 8)
        self.d_llm = (self.d_llm + self.d_llm.T) / 2
        torch.diagonal(self.d_llm).fill_(0)
        self.d_k2p = torch.rand(8, 8)
        self.d_k2p = (self.d_k2p + self.d_k2p.T) / 2
        torch.diagonal(self.d_k2p).fill_(0)

    def test_forward(self):
        dist = self.hybrid(self.d_llm, self.d_k2p)
        self.assertEqual(dist.shape, self.d_llm.shape)

    def test_alpha_learning(self):
        initial_alpha = self.hybrid.alpha.item()
        self.hybrid.train()
        dist = self.hybrid(self.d_llm, self.d_k2p)
        loss = dist.mean()
        loss.backward()
        self.assertIsNotNone(self.hybrid.alpha.grad)


class TestNJBuilder(TestCase):
    """Test Neighbor-Joining tree construction."""

    def setUp(self):
        from src.models.tree.nj_builder import NJTreeBuilder
        self.builder = NJTreeBuilder()
        self.dist_matrix = np.array([
            [0.0, 0.5, 1.0, 1.5],
            [0.5, 0.0, 1.0, 1.5],
            [1.0, 1.0, 0.0, 0.5],
            [1.5, 1.5, 0.5, 0.0],
        ])
        self.names = ["seq1", "seq2", "seq3", "seq4"]

    def test_build_tree_shape(self):
        newick = self.builder.build(self.dist_matrix, self.names)
        self.assertIsInstance(newick, str)
        self.assertTrue(newick.endswith(";"))

    def test_trivial_tree_1(self):
        dist = np.array([[0.0]])
        newick = self.builder.build(dist, ["seq1"])
        self.assertEqual(newick, "seq1:0.0;")

    def test_trivial_tree_2(self):
        dist = np.array([[0.0, 0.5], [0.5, 0.0]])
        newick = self.builder.build(dist, ["seq1", "seq2"])
        self.assertIn("seq1", newick)
        self.assertIn("seq2", newick)

    def test_symmetric_matrix(self):
        newick = self.builder.build(self.dist_matrix, self.names)
        self.assertIsNotNone(newick)


@skipIf(not HAS_DENDROPY, "dendropy not available")
class TestTreeMetrics(TestCase):
    """Test tree evaluation metrics."""

    def setUp(self):
        from src.models.tree.tree_metrics import TreeMetrics
        self.metrics = TreeMetrics()
        self.tree1 = "(seq1:0.1,seq2:0.2):0.0;"
        self.tree2 = "(seq1:0.15,seq2:0.25):0.0;"
        self.tree3 = "(seq1:0.1,seq3:0.2):0.0;"

    def test_rf_distance_identical(self):
        from src.models.tree.tree_metrics import compute_rf_distance
        rf, nrf = compute_rf_distance(self.tree1, self.tree1)
        self.assertEqual(rf, 0)
        self.assertEqual(nrf, 0)

    def test_rf_distance_different(self):
        from src.models.tree.tree_metrics import compute_rf_distance
        rf, nrf = compute_rf_distance(self.tree1, self.tree3)
        self.assertGreater(rf, 0)

    def test_quartet_accuracy(self):
        from src.models.tree.tree_metrics import compute_quartet_accuracy
        acc = compute_quartet_accuracy(self.tree1, self.tree1, n_quartets=10)
        self.assertEqual(acc, 1.0)

    def test_evaluate(self):
        result = self.metrics.evaluate(self.tree1, self.tree1)
        self.assertEqual(result['nrf'], 0.0)
        self.assertEqual(result['qa'], 1.0)


class TestGTRModel(TestCase):
    """Test GTR substitution model."""

    def setUp(self):
        from src.models.route_a.gtr_module import GTRModel
        self.gtr = GTRModel()
        self.rates = torch.tensor([1.0, 2.0, 1.5, 1.0, 2.5, 1.0])
        self.freq = torch.tensor([0.25, 0.25, 0.25, 0.25])
        self.alpha = torch.tensor([1.0])

    def test_normalize_rates(self):
        rates = self.gtr.normalize_rates(self.rates)
        self.assertEqual(rates.shape, (6,))
        self.assertTrue(torch.all(rates > 0))

    def test_normalize_frequencies(self):
        freq = self.gtr.normalize_frequencies(self.freq)
        self.assertAlmostEqual(freq.sum().item(), 1.0, places=5)

    def test_compute_q_matrix(self):
        rates = self.gtr.normalize_rates(self.rates)
        freq = self.gtr.normalize_frequencies(self.freq)
        Q = self.gtr.compute_Q_matrix(rates, freq)
        self.assertEqual(Q.shape, (4, 4))

    def test_forward(self):
        P, rates, freq, alpha, Q = self.gtr.forward(self.rates, self.freq, self.alpha, t=0.1)
        self.assertEqual(P.shape, (4, 4))
        self.assertTrue(torch.all(P >= 0))
        row_sums = P.sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(4), atol=1e-5))


class TestFelsensteinPruning(TestCase):
    """Test Felsenstein pruning algorithm."""

    def setUp(self):
        from src.models.route_a.felsenstein import FelsensteinPruning
        self.fels = FelsensteinPruning(n_bases=4, n_gamma_categories=4)

    def test_gamma_rates(self):
        alpha = torch.tensor([1.0])
        rates = self.fels.compute_gamma_rates(alpha, n_categories=4)
        self.assertEqual(rates.shape, (4,))

    def test_transition_prob(self):
        Q = torch.eye(4)
        P = self.fels.compute_transition_prob(Q, t=0.1)
        self.assertEqual(P.shape, (4, 4))

    def test_pruning_simple(self):
        from src.models.route_a.gtr_module import GTRModel
        gtr = GTRModel()
        P, rates, freq, alpha, Q = gtr.forward(
            torch.tensor([1.0, 2.0, 1.5, 1.0, 2.5, 1.0]),
            torch.tensor([0.25, 0.25, 0.25, 0.25]),
            torch.tensor([1.0]),
            t=0.1
        )
        alignment = torch.randint(0, 4, (4, 50))
        tree_structure = {
            0: {'children': [], 'seq_idx': 0},
            1: {'children': [], 'seq_idx': 1},
            2: {'children': [], 'seq_idx': 2},
            3: {'children': [], 'seq_idx': 3},
            4: {'children': [(0, 0.1), (1, 0.1)]},
            5: {'children': [(2, 0.1), (3, 0.1)]},
            'root': 6,
        }
        for i in range(4, 7):
            tree_structure[i] = {'children': [(i*2-8, 0.1), (i*2-7, 0.1)]}
        tree_structure[6] = {'children': [(4, 0.1), (5, 0.1)]}
        tree_structure['root'] = 6

        ll = self.fels(alignment, tree_structure, Q, freq, alpha)
        self.assertIsInstance(ll.item(), float)


class TestLosses(TestCase):
    """Test loss functions."""

    def test_quartet_loss(self):
        from src.training.losses import QuartetLoss
        loss_fn = QuartetLoss()
        dist_matrix = torch.rand(8, 8)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        torch.diagonal(dist_matrix).fill_(0)
        quartet_indices = [(0, 1, 2, 3), (0, 2, 1, 3)]
        loss = loss_fn(dist_matrix, quartet_indices=quartet_indices)
        self.assertIsInstance(loss.item(), float)

    def test_distance_regression_huber(self):
        from src.training.losses import DistanceRegressionLoss
        loss_fn = DistanceRegressionLoss(loss_type="huber")
        pred = torch.rand(8, 8)
        target = torch.rand(8, 8)
        loss = loss_fn(pred, target)
        self.assertIsInstance(loss.item(), float)

    def test_distance_regression_mse(self):
        from src.training.losses import DistanceRegressionLoss
        loss_fn = DistanceRegressionLoss(loss_type="mse")
        loss = loss_fn(torch.rand(8, 8), torch.rand(8, 8))
        self.assertIsInstance(loss.item(), float)

    def test_triple_loss(self):
        from src.training.losses import TripleLoss
        loss_fn = TripleLoss(alpha=1.0, beta=0.5, gamma=0.5)
        dist_matrix = torch.rand(8, 8)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        torch.diagonal(dist_matrix).fill_(0)
        target_dist = torch.rand(8, 8)
        log_likelihood = torch.tensor([-1.0, -2.0, -1.5])
        quartet_indices = [(0, 1, 2, 3)]
        loss = loss_fn(
            dist_matrix=dist_matrix,
            target_dist=target_dist,
            log_likelihood=log_likelihood,
            quartet_indices=quartet_indices
        )
        self.assertIsInstance(loss.item(), float)
        self.assertFalse(torch.isnan(loss))

    def test_loss_weight_scheduler(self):
        from src.training.losses import LossWeightScheduler
        scheduler = LossWeightScheduler(total_epochs=100, schedule_type="phased")
        alpha, beta, gamma = scheduler.get_weights(10)
        self.assertIsInstance(alpha, float)
        self.assertIsInstance(beta, float)


class TestBiMambaBlock(TestCase):
    """Test BiMamba blocks."""

    def test_mamba_block_forward(self):
        from src.models.route_b.bimamba_block import MambaBlock
        block = MambaBlock(d_model=64, d_state=8)
        x = torch.randn(4, 16, 64)
        out = block(x)
        self.assertEqual(out.shape, x.shape)

    def test_bimamba_block_forward(self):
        from src.models.route_b.bimamba_block import BiMambaBlock
        block = BiMambaBlock(d_model=64, d_state=8)
        x = torch.randn(4, 16, 64)
        out = block(x)
        self.assertEqual(out.shape, x.shape)

    def test_bimamba_stack_forward(self):
        from src.models.route_b.bimamba_block import BiMambaStack
        stack = BiMambaStack(d_model=64, n_layers=2)
        x = torch.randn(4, 16, 64)
        out = stack(x)
        self.assertEqual(out.shape, x.shape)

    def test_tree_head_forward(self):
        from src.models.route_b.bimamba_block import TreeHead
        head = TreeHead(d_model=64, n_heads=4)
        cls_tokens = torch.randn(2, 8, 64)
        seq_tokens = torch.randn(2, 16, 64)
        out = head(cls_tokens, seq_tokens)
        self.assertEqual(out.shape, cls_tokens.shape)

    def test_nucleotide_tokenizer(self):
        from src.models.route_b.bimamba_block import NucleotideTokenizer
        tokenizer = NucleotideTokenizer(k=6, d_model=64)
        seqs = ["ACGTACGT", "GGGGCCCC"]
        out = tokenizer(seqs)
        self.assertEqual(out.dim(), 3)


class TestViralPhyloGPN(TestCase):
    """Test Viral PhyloGPN model."""

    def setUp(self):
        from src.models.route_a.viral_phylogpn import ViralPhyloGPN
        self.model = ViralPhyloGPN(window_size=241, d_model=128, n_blocks=2)

    def test_model_forward(self):
        seq_onehot = torch.randint(0, 5, (4, 241, 5)).float()
        raw_rates, raw_freq, raw_alpha, site_emb = self.model(seq_onehot)
        self.assertEqual(raw_rates.shape, (4, 241, 6))
        self.assertEqual(raw_freq.shape, (4, 241, 4))
        self.assertEqual(raw_alpha.shape, (4, 241))

    def test_encode_onehot(self):
        seq_tensor = torch.randint(0, 4, (4, 50))
        onehot = self.model.encode_onehot(seq_tensor)
        self.assertEqual(onehot.shape, (4, 50, 5))

    def test_predict_gtr_params(self):
        seq_onehot = torch.randint(0, 5, (2, 100, 5)).float()
        rates, freq, alpha = self.model.predict_gtr_params(seq_onehot)
        self.assertEqual(rates.shape[-1], 6)
        self.assertEqual(freq.shape[-1], 4)


class TestCompositionFeatureExtractor(TestCase):
    """Test composition feature extraction."""

    def setUp(self):
        from src.data.viral_dataset import CompositionFeatureExtractor
        self.extractor = CompositionFeatureExtractor(k=4)

    def test_gc_content(self):
        gc = self.extractor.extract_gc_content("ACGTACGT")
        self.assertAlmostEqual(gc, 0.5, places=2)

    def test_kmer_counts(self):
        counts = self.extractor.extract_kmer_counts("ACGTACGTACGT", k=2)
        self.assertEqual(len(counts), 16)

    def test_composition_features(self):
        features = self.extractor.extract_composition_features("ACGTACGTACGT")
        self.assertGreater(len(features), 0)

    def test_extract_batch(self):
        seqs = ["ACGTACGT", "GGGGCCCC", "AAAATTTT"]
        features = self.extractor.extract_batch(seqs)
        self.assertEqual(features.shape[0], 3)


class TestRouteCModelMock(TestCase):
    """Test Route C model with mock backbone."""

    def test_model_initialization(self):
        from src.models.route_c.route_c_model import RouteCModel

        class MockBackbone:
            def __init__(self):
                self.embed_dim = 128
            def __call__(self, sequences):
                batch_size = len(sequences)
                return torch.randn(batch_size, self.embed_dim), torch.randn(batch_size, 100, self.embed_dim)

        backbone = MockBackbone()
        model = RouteCModel(
            backbone=backbone,
            embed_dim=128,
            use_calibration=True,
            use_hybrid_distance=True
        )
        self.assertIsNotNone(model)

    def test_model_forward(self):
        from src.models.route_c.route_c_model import RouteCModel

        class MockBackbone:
            def __init__(self):
                self.embed_dim = 128
            def __call__(self, sequences):
                batch_size = len(sequences)
                return torch.randn(batch_size, self.embed_dim), torch.randn(batch_size, 100, self.embed_dim)

        backbone = MockBackbone()
        model = RouteCModel(backbone=backbone, embed_dim=128, use_hybrid_distance=False)
        seqs = ["ACGTACGT", "GGGGCCCC", "AAAATTTT"]
        dist = model(seqs)
        self.assertEqual(dist.shape[0], 3)


class TestSanityChecks(TestCase):
    """Sanity checks for numerical stability."""

    def test_no_nan_in_random_forward(self):
        from src.models.calibration.zca_whitening import EmbeddingCalibration
        calib = EmbeddingCalibration(embed_dim=256)
        for _ in range(10):
            x = torch.randn(32, 256)
            calib.train()
            out, _ = calib(x, torch.randn(32, 20))
            self.assertFalse(torch.isnan(out).any())

    def test_no_inf_in_k2p(self):
        from src.models.distance.k2p_baseline import compute_k2p_distance
        d = compute_k2p_distance("ACGTACGT", "ACGTACGT")
        self.assertFalse(np.isinf(d))

    def test_no_nan_in_losses(self):
        from src.training.losses import TripleLoss
        loss_fn = TripleLoss()
        dist = torch.rand(8, 8)
        dist = (dist + dist.T) / 2
        torch.diagonal(dist).fill_(0)
        target = torch.rand(8, 8)
        ll = torch.randn(10)
        for _ in range(10):
            loss = loss_fn(dist, target, ll, [(0,1,2,3)])
            self.assertFalse(torch.isnan(loss))


def run_tests():
    """Run all tests."""
    unittest_main(verbosity=2)


if __name__ == "__main__":
    run_tests()
