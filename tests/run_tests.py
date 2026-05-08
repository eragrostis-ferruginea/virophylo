#!/usr/bin/env python3
"""Standalone test runner for ViroPhylo."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

print("=" * 60)
print("ViroPhylo: Testing Core Components")
print("=" * 60)

def test_imports():
    print("\n[1/10] Testing module imports...")
    try:
        from src.models.calibration.zca_whitening import EmbeddingCalibration
        from src.models.distance.distance_head import DistanceHead
        from src.models.distance.hybrid_distance import HybridDistance
        from src.models.distance.k2p_baseline import K2PDistance
        from src.models.tree.nj_builder import NJTreeBuilder
        from src.models.tree.tree_metrics import TreeMetrics
        from src.training.losses import TripleLoss, QuartetLoss
        from src.models.route_a.gtr_module import GTRModel
        from src.models.route_a.felsenstein import FelsensteinPruning
        from src.models.route_b.bimamba_block import BiMambaStack
        from src.models.route_a.viral_phylogpn import ViralPhyloGPN
        from src.data.viral_dataset import CompositionFeatureExtractor
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_calibration():
    print("\n[2/10] Testing ZCA Whitening...")
    try:
        from src.models.calibration.zca_whitening import ZCAWhitening, EmbeddingCalibration
        zca = ZCAWhitening(embed_dim=128)
        x = torch.randn(32, 128)
        zca.train()
        zca.eval()
        with torch.no_grad():
            out = zca(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
        print("  ✓ ZCA Whitening works correctly")
        return True
    except Exception as e:
        print(f"  ✗ ZCA Whitening failed: {e}")
        return False

def test_composition_debias():
    print("\n[3/10] Testing Composition Debiasing...")
    try:
        from src.models.calibration.composition_debias import CompositionDebias
        debias = CompositionDebias(embed_dim=128, n_composition_features=20)
        emb = torch.randn(8, 128)
        comp = torch.randn(8, 20)
        out = debias(emb, comp)
        assert out.shape == emb.shape
        print("  ✓ Composition Debiasing works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Composition Debiasing failed: {e}")
        return False

def test_distance_head():
    print("\n[4/10] Testing Distance Head...")
    try:
        from src.models.distance.distance_head import DistanceHead
        head = DistanceHead(embed_dim=128)
        emb = torch.randn(8, 128)
        dist = head.pairwise_distances(emb)
        assert dist.shape == (8, 8)
        assert torch.allclose(dist, dist.T, atol=1e-5)
        print("  ✓ Distance Head works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Distance Head failed: {e}")
        return False

def test_k2p():
    print("\n[5/10] Testing K2P Distance...")
    try:
        from src.models.distance.k2p_baseline import K2PDistance
        k2p = K2PDistance()
        seqs = ["ACGTACGT", "ACGTTTTT", "AAAAAAAA"]
        dist = k2p.compute(seqs)
        assert dist.shape == (3, 3)
        assert torch.allclose(dist, dist.T, atol=1e-5)
        print("  ✓ K2P Distance works correctly")
        return True
    except Exception as e:
        print(f"  ✗ K2P Distance failed: {e}")
        return False

def test_nj_tree():
    print("\n[6/10] Testing NJ Tree Building...")
    try:
        from src.models.tree.nj_builder import NJTreeBuilder
        builder = NJTreeBuilder()
        dist = np.array([
            [0.0, 0.5, 1.0, 1.5],
            [0.5, 0.0, 1.0, 1.5],
            [1.0, 1.0, 0.0, 0.5],
            [1.5, 1.5, 0.5, 0.0],
        ])
        names = ["s1", "s2", "s3", "s4"]
        newick = builder.build(dist, names)
        assert isinstance(newick, str)
        assert newick.endswith(";")
        print(f"  ✓ NJ Tree Builder works correctly: {newick[:50]}...")
        return True
    except Exception as e:
        print(f"  ✗ NJ Tree Builder failed: {e}")
        return False

def test_tree_metrics():
    print("\n[7/10] Testing Tree Metrics...")
    try:
        from src.models.tree.tree_metrics import compute_rf_distance, TreeMetrics
        tree1 = "(seq1:0.1,seq2:0.2):0.0;"
        rf, nrf = compute_rf_distance(tree1, tree1)
        assert rf == 0
        assert nrf == 0
        metrics = TreeMetrics()
        result = metrics.evaluate(tree1, tree1)
        assert result['nrf'] == 0.0
        print("  ✓ Tree Metrics work correctly")
        return True
    except ImportError:
        print("  ⊘ Skipped (dendropy not available)")
        return True
    except Exception as e:
        print(f"  ✗ Tree Metrics failed: {e}")
        return False

def test_gtr_model():
    print("\n[8/10] Testing GTR Model...")
    try:
        from src.models.route_a.gtr_module import GTRModel
        gtr = GTRModel()
        rates = torch.tensor([1.0, 2.0, 1.5, 1.0, 2.5, 1.0])
        freq = torch.tensor([0.25, 0.25, 0.25, 0.25])
        alpha = torch.tensor([1.0])
        P, rates_n, freq_n, alpha_n, Q = gtr.forward(rates, freq, alpha, t=0.1)
        assert P.shape == (4, 4)
        assert torch.all(P >= 0)
        print("  ✓ GTR Model works correctly")
        return True
    except Exception as e:
        print(f"  ✗ GTR Model failed: {e}")
        return False

def test_bimamba():
    print("\n[9/10] Testing BiMamba Blocks...")
    try:
        from src.models.route_b.bimamba_block import BiMambaStack, BiMambaBlock
        block = BiMambaBlock(d_model=64, d_state=8)
        x = torch.randn(4, 16, 64)
        out = block(x)
        assert out.shape == x.shape
        stack = BiMambaStack(d_model=64, n_layers=2)
        out = stack(x)
        assert out.shape == x.shape
        print("  ✓ BiMamba Blocks work correctly")
        return True
    except Exception as e:
        print(f"  ✗ BiMamba failed: {e}")
        return False

def test_viral_phylogpn():
    print("\n[10/10] Testing Viral PhyloGPN...")
    try:
        from src.models.route_a.viral_phylogpn import ViralPhyloGPN
        model = ViralPhyloGPN(window_size=100, d_model=128, n_blocks=2)
        seq = torch.randint(0, 5, (4, 100, 5)).float()
        rates, freq, alpha, emb = model(seq)
        assert rates.shape == (4, 100, 6)
        assert freq.shape == (4, 100, 4)
        print("  ✓ Viral PhyloGPN works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Viral PhyloGPN failed: {e}")
        return False

def test_sanity_checks():
    print("\n[Bonus] Running sanity checks...")
    try:
        from src.training.losses import TripleLoss
        loss_fn = TripleLoss()
        dist = torch.rand(8, 8)
        dist = (dist + dist.T) / 2
        torch.diagonal(dist).fill_(0)
        target = torch.rand(8, 8)
        ll = torch.randn(10)
        loss = loss_fn(dist, target, ll, [(0,1,2,3)])
        assert not torch.isnan(loss)
        print("  ✓ Loss computation is numerically stable")
        return True
    except Exception as e:
        print(f"  ✗ Sanity check failed: {e}")
        return False

def main():
    results = []
    results.append(test_imports())
    results.append(test_calibration())
    results.append(test_composition_debias())
    results.append(test_distance_head())
    results.append(test_k2p())
    results.append(test_nj_tree())
    results.append(test_tree_metrics())
    results.append(test_gtr_model())
    results.append(test_bimamba())
    results.append(test_viral_phylogpn())
    results.append(test_sanity_checks())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    if passed == total:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
