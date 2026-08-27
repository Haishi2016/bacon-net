"""Unit tests for :class:`VectorHybridTreeHead` (binary spine + full sub-tree)."""

import math

import pytest
import torch

from bacon.aggregators.lsp.full_weight import lsp_power_mean
from bacon.vectorizedHybridTree import VectorHybridTreeHead


def test_split_sizes_quarter():
    h = VectorHybridTreeHead(112, 200, bin_frac=0.25)
    assert h.n_bin == 28
    assert h.n_full == 84
    assert h.n_bin + h.n_full == 112


def test_split_clamped_both_sides_nonempty():
    # tiny input: still one feature each side
    h = VectorHybridTreeHead(2, 4, bin_frac=0.25)
    assert h.n_bin == 1 and h.n_full == 1
    h2 = VectorHybridTreeHead(2, 4, bin_frac=0.99)
    assert h2.n_bin == 1 and h2.n_full == 1


def test_forward_shape_2d():
    h = VectorHybridTreeHead(20, 7, bin_frac=0.25)
    out = h(torch.rand(5, 20))
    assert out.shape == (5, 7)


def test_forward_shape_3d_perhead():
    h = VectorHybridTreeHead(20, 7, bin_frac=0.25)
    out = h(torch.rand(5, 7, 20))
    assert out.shape == (5, 7)


def test_output_bounded():
    h = VectorHybridTreeHead(16, 3, bin_frac=0.5)
    out = h(torch.rand(32, 16))
    assert torch.all(out > 0) and torch.all(out < 1)


def test_wrong_feature_dim_raises():
    h = VectorHybridTreeHead(16, 3)
    with pytest.raises(ValueError):
        h(torch.rand(4, 15))


def test_input_size_too_small_raises():
    with pytest.raises(ValueError):
        VectorHybridTreeHead(1, 3)


def test_single_bin_feature_matches_pmean_of_full_and_feat():
    # n_bin=1: output == lsp_power_mean([full_agg, feat]) with the spine's params.
    torch.manual_seed(0)
    h = VectorHybridTreeHead(5, 2, bin_frac=0.2)      # n_bin=1, n_full=4
    assert h.n_bin == 1
    x = torch.rand(8, 5)
    full_agg = h.full(x[:, h.n_bin:])                  # (8,2)
    feat = x[:, 0].unsqueeze(1).expand(8, 2)           # (8,2)
    a = torch.sigmoid(h.bin_andness[:, 0]) * 3 - 1      # (2,)
    wf = torch.sigmoid(h.bin_weight_logit[:, 0])        # (2,)
    X = torch.stack([full_agg, feat], dim=0)
    w = torch.stack([1 - wf, wf], dim=0).unsqueeze(1)
    ref = lsp_power_mean(X, a.unsqueeze(0), w, eps=1e-6).clamp(h.eps, 1 - h.eps)
    got = h(x)
    assert torch.allclose(got, ref, atol=1e-6)


def test_gradients_flow_to_spine_and_full():
    h = VectorHybridTreeHead(24, 4, bin_frac=0.25)
    out = h(torch.rand(6, 24))
    out.sum().backward()
    assert h.bin_andness.grad is not None and h.bin_andness.grad.abs().sum() > 0
    assert h.bin_weight_logit.grad is not None
    # at least one full-tree routing parameter receives gradient
    g = h.full.route_logits[0].grad
    assert g is not None and torch.isfinite(g).all()


def test_egress_freeze_delegates():
    h = VectorHybridTreeHead(40, 5, bin_frac=0.25)
    assert not bool(h.egress_frozen)
    h.freeze_egress()
    assert bool(h.egress_frozen)
    h.unfreeze_egress()
    assert not bool(h.egress_frozen)


def test_freeze_scan_delegates_and_freezes():
    h = VectorHybridTreeHead(40, 5, bin_frac=0.25)
    x = torch.rand(64, 40)
    soft = h(x)
    mse = h.freeze_egress_scan(x, num_candidates=16)
    assert bool(h.egress_frozen)
    assert torch.isfinite(mse)
    # frozen forward stays finite and bounded
    hard = h(x)
    assert torch.all(hard > 0) and torch.all(hard < 1)


def test_sparsity_confidence_and_anneal_delegate():
    h = VectorHybridTreeHead(40, 5, bin_frac=0.25)
    s = h.egress_sparsity_loss()
    c = h.egress_confidence()
    assert torch.isfinite(s) and torch.isfinite(c)
    t0 = float(h.full.temperature)
    h.anneal(1.0)
    assert float(h.full.temperature) <= t0


def test_not_permutation_invariant():
    # swapping a binary-spine feature with a full-tree feature changes output
    torch.manual_seed(1)
    h = VectorHybridTreeHead(12, 3, bin_frac=0.25)
    x = torch.rand(4, 12)
    xp = x.clone()
    xp[:, [0, h.n_bin]] = x[:, [h.n_bin, 0]]           # swap across the split
    assert not torch.allclose(h(x), h(xp), atol=1e-5)


def test_max_egress_dag_forward_and_freeze():
    h = VectorHybridTreeHead(40, 5, bin_frac=0.25, branching=8, max_egress=2)
    x = torch.rand(8, 40)
    out = h(x)
    assert out.shape == (8, 5)
    h.freeze_egress()
    assert bool(h.egress_frozen)
    assert h(x).shape == (8, 5)
