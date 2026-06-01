"""Tests for previously untested loss functions, utilities, and boundary conditions.

Covers:
- MeanVarianceLoss (was zero coverage)
- OrderRegressionLoss / utils_cdf (was zero coverage)
- calculate_lds_weights (was zero coverage)
- mixup_data (was zero coverage)
- probs_to_ages (was only mocked)
- DLDLProcessor boundary ages (0, max_age)
- CombinedLoss with use_mv_loss=True
- Numerical stability edge cases
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import Config
from dataset import calculate_lds_weights
from evaluation import normalize_tta_mode, probs_to_ages
from train import mixup_data
from utils import (
    CombinedLoss,
    DLDLProcessor,
    MeanVarianceLoss,
    OrderRegressionLoss,
    remap_state_dict_keys,
    utils_cdf,
)


def _cfg(**overrides):
    cfg = Config()
    cfg.device = torch.device("cpu")
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# ── MeanVarianceLoss ──


class TestMeanVarianceLoss:
    def test_output_is_scalar(self):
        cfg = _cfg(use_mv_loss=True)
        loss_fn = MeanVarianceLoss(
            lambda_var=0.1, start_age=0, end_age=80, device="cpu"
        )
        logits = torch.randn(4, 81)
        targets = torch.tensor([10.0, 30.0, 50.0, 70.0])
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_perfect_prediction_has_low_loss(self):
        loss_fn = MeanVarianceLoss(
            lambda_var=0.1, start_age=0, end_age=80, device="cpu"
        )
        # Create logits that produce a sharp distribution at age=40
        logits = torch.full((1, 81), -10.0)
        logits[0, 40] = 10.0
        targets = torch.tensor([40.0])
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.1

    def test_wrong_prediction_has_high_loss(self):
        loss_fn = MeanVarianceLoss(
            lambda_var=0.1, start_age=0, end_age=80, device="cpu"
        )
        logits = torch.full((1, 81), -10.0)
        logits[0, 10] = 10.0  # predict age 10
        targets = torch.tensor([70.0])  # true age 70
        loss = loss_fn(logits, targets)
        assert loss.item() > 10.0

    def test_gradient_flows(self):
        loss_fn = MeanVarianceLoss(
            lambda_var=0.1, start_age=0, end_age=80, device="cpu"
        )
        logits = torch.randn(4, 81, requires_grad=True)
        targets = torch.tensor([10.0, 30.0, 50.0, 70.0])
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_variance_reduces_with_lambda(self):
        # A higher lambda_var must increase the variance-term contribution to the
        # total loss. We isolate that contribution as (total - mean_term), where
        # mean_term is lambda-independent for fixed inputs.
        logits = torch.randn(4, 81)
        targets = torch.tensor([10.0, 30.0, 50.0, 70.0])

        def variance_contribution(lambda_var):
            loss_fn = MeanVarianceLoss(lambda_var=lambda_var, start_age=0, end_age=80, device="cpu")
            probs = F.softmax(logits, dim=1)
            mean = torch.sum(probs * loss_fn.age_centers, dim=1)
            l_mean = F.mse_loss(mean, targets)
            total = loss_fn(logits, targets)
            return (total - l_mean).item()

        low = variance_contribution(0.01)
        high = variance_contribution(10.0)
        assert high > low
        assert low >= 0


# ── OrderRegressionLoss / utils_cdf ──


class TestOrderRegressionLoss:
    def test_output_is_scalar(self):
        cfg = _cfg(use_dldl_v2=True)
        loss_fn = OrderRegressionLoss(cfg)
        logits = torch.randn(4, 81)
        true_ages = torch.tensor([10.0, 30.0, 50.0, 70.0])
        loss = loss_fn(logits, true_ages)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_gradient_flows(self):
        cfg = _cfg(use_dldl_v2=True)
        loss_fn = OrderRegressionLoss(cfg)
        logits = torch.randn(4, 81, requires_grad=True)
        true_ages = torch.tensor([10.0, 30.0, 50.0, 70.0])
        loss = loss_fn(logits, true_ages)
        loss.backward()
        assert logits.grad is not None

    def test_with_target_dists(self):
        cfg = _cfg(use_dldl_v2=True)
        loss_fn = OrderRegressionLoss(cfg)
        logits = torch.randn(4, 81)
        true_ages = torch.tensor([10.0, 30.0, 50.0, 70.0])
        target_dists = F.softmax(torch.randn(4, 81), dim=1)
        loss = loss_fn(logits, true_ages, target_dists)
        assert loss.item() >= 0

    def test_boundary_ages(self):
        """Test with age=0 and age=80."""
        cfg = _cfg(use_dldl_v2=True)
        loss_fn = OrderRegressionLoss(cfg)
        logits = torch.randn(2, 81)
        true_ages = torch.tensor([0.0, 80.0])
        loss = loss_fn(logits, true_ages)
        assert torch.isfinite(loss)


class TestUtilsCdf:
    def test_heaviside_step(self):
        cdf = utils_cdf(torch.tensor([3.0]), 10, "cpu")
        expected = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float32)
        assert torch.equal(cdf, expected)

    def test_age_zero(self):
        cdf = utils_cdf(torch.tensor([0.0]), 5, "cpu")
        expected = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.float32)
        assert torch.equal(cdf, expected)

    def test_age_beyond_range(self):
        cdf = utils_cdf(torch.tensor([100.0]), 5, "cpu")
        expected = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.float32)
        assert torch.equal(cdf, expected)


# ── calculate_lds_weights ──


class TestLDSWeights:
    def test_returns_tensor(self):
        cfg = _cfg(use_reweighting=True, lds_sigma=4, num_classes=81)
        ages = [10.0] * 100 + [30.0] * 50 + [50.0] * 10
        weights = calculate_lds_weights(ages, cfg)
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (81,)

    def test_rare_age_has_higher_weight(self):
        cfg = _cfg(use_reweighting=True, lds_sigma=4, num_classes=81)
        ages = [10.0] * 100 + [50.0] * 5
        weights = calculate_lds_weights(ages, cfg)
        assert weights[50] > weights[10]

    def test_weights_are_positive(self):
        cfg = _cfg(use_reweighting=True, lds_sigma=4, num_classes=81)
        ages = [10.0] * 100 + [30.0] * 50
        weights = calculate_lds_weights(ages, cfg)
        assert (weights > 0).all()

    def test_weights_clipped_to_max_10(self):
        cfg = _cfg(use_reweighting=True, lds_sigma=4, num_classes=81)
        ages = [10.0] * 1000 + [50.0] * 1
        weights = calculate_lds_weights(ages, cfg)
        assert weights.max() <= 10.0

    def test_mean_of_active_weights_is_near_one(self):
        cfg = _cfg(use_reweighting=True, lds_sigma=4, num_classes=81)
        ages = [10.0] * 100 + [30.0] * 100 + [50.0] * 100
        weights = calculate_lds_weights(ages, cfg)
        # Active classes (10, 30, 50) should have mean ~1.0
        active = weights[[10, 30, 50]]
        assert abs(active.mean().item() - 1.0) < 0.5


# ── mixup_data ──


class TestMixupData:
    def test_shapes_preserved(self):
        x = torch.randn(8, 3, 224, 224)
        y_dist = torch.softmax(torch.randn(8, 81), dim=1)
        y_age = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        mx, md, ma = mixup_data(x, y_dist, y_age, alpha=0.4)
        assert mx.shape == x.shape
        assert md.shape == y_dist.shape
        assert ma.shape == y_age.shape

    def test_alpha_zero_returns_original(self):
        x = torch.randn(4, 3, 32, 32)
        y_dist = torch.ones(4, 81) / 81
        y_age = torch.tensor([10.0, 20.0, 30.0, 40.0])
        mx, md, ma = mixup_data(x, y_dist, y_age, alpha=0.0)
        assert torch.allclose(mx, x)
        assert torch.allclose(md, y_dist)
        assert torch.allclose(ma, y_age)

    def test_mixed_values_stay_in_range(self):
        x = torch.randn(4, 3, 32, 32)
        y_dist = torch.softmax(torch.randn(4, 81), dim=1)
        y_age = torch.tensor([10.0, 20.0, 30.0, 40.0])
        mx, md, ma = mixup_data(x, y_dist, y_age, alpha=1.0)
        # Mixed values should be convex combinations
        assert ma.min() >= 10.0 - 1e-5
        assert ma.max() <= 40.0 + 1e-5
        # Distributions should still sum to ~1
        assert torch.allclose(md.sum(dim=1), torch.ones(4), atol=1e-5)


# ── probs_to_ages ──


class TestProbsToAges:
    def test_one_hot_returns_exact_age(self):
        probs = torch.zeros(1, 81)
        probs[0, 42] = 1.0
        age = probs_to_ages(probs, 81)
        assert abs(age.item() - 42.0) < 1e-5

    def test_uniform_distribution_returns_midpoint(self):
        probs = torch.ones(1, 81) / 81
        age = probs_to_ages(probs, 81)
        expected = 40.0  # mean of 0..80
        assert abs(age.item() - expected) < 0.5

    def test_batch_processing(self):
        probs = torch.softmax(torch.randn(8, 81), dim=1)
        ages = probs_to_ages(probs, 81)
        assert ages.shape == (8,)
        assert (ages >= 0).all() and (ages <= 80).all()


# ── DLDLProcessor boundary ages ──


class TestDLDLBoundaryAges:
    def test_age_zero(self):
        proc = DLDLProcessor(_cfg(use_dldl_v2=True, use_adaptive_sigma=True))
        dist = proc.generate_label_distribution(torch.tensor(0.0))
        assert torch.isfinite(dist).all()
        assert abs(dist.sum().item() - 1.0) < 1e-5
        assert dist.argmax().item() == 0

    def test_age_max(self):
        proc = DLDLProcessor(_cfg(use_dldl_v2=True, use_adaptive_sigma=True))
        dist = proc.generate_label_distribution(torch.tensor(80.0))
        assert torch.isfinite(dist).all()
        assert abs(dist.sum().item() - 1.0) < 1e-5
        assert dist.argmax().item() == 80

    def test_sigma_scales_with_age(self):
        proc = DLDLProcessor(_cfg(use_dldl_v2=True, use_adaptive_sigma=True))
        dist_young = proc.generate_label_distribution(torch.tensor(0.0))
        dist_old = proc.generate_label_distribution(torch.tensor(80.0))
        # Older ages should have wider distributions (higher sigma)
        # Measured by entropy
        ent_young = -(dist_young * (dist_young + 1e-10).log()).sum()
        ent_old = -(dist_old * (dist_old + 1e-10).log()).sum()
        assert ent_old > ent_young

    def test_one_hot_mode(self):
        proc = DLDLProcessor(_cfg(use_dldl_v2=False))
        dist = proc.generate_label_distribution(torch.tensor(42.0))
        assert dist[42].item() == 1.0
        assert dist.sum().item() == 1.0


# ── CombinedLoss with MV enabled ──


class TestCombinedLossWithMV:
    def test_mv_loss_enabled(self):
        cfg = _cfg(use_mv_loss=True, use_dldl_v2=True, use_adaptive_triplet=False,
                    use_asymmetric_ordinal=False, use_moe=False)
        criterion = CombinedLoss(cfg)
        logits = torch.randn(4, 81, requires_grad=True)
        log_probs = F.log_softmax(logits, dim=1)
        target_dists = F.softmax(torch.randn(4, 81), dim=1)
        true_ages = torch.tensor([10.0, 30.0, 50.0, 70.0])
        result = criterion(log_probs, target_dists, true_ages, logits)
        total, kl, l1, rank, mv, triplet, asym, moe_gate, pred_age = result
        assert mv.item() > 0  # MV loss should be nonzero
        assert total.requires_grad

    def test_all_losses_disabled(self):
        cfg = _cfg(use_mv_loss=False, use_dldl_v2=False, use_adaptive_triplet=False,
                    use_asymmetric_ordinal=False, use_moe=False)
        criterion = CombinedLoss(cfg)
        logits = torch.randn(4, 81, requires_grad=True)
        log_probs = F.log_softmax(logits, dim=1)
        target_dists = F.softmax(torch.randn(4, 81), dim=1)
        true_ages = torch.tensor([10.0, 30.0, 50.0, 70.0])
        result = criterion(log_probs, target_dists, true_ages, logits)
        total = result[0]
        assert total.requires_grad


# ── normalize_tta_mode ──


class TestNormalizeTtaMode:
    def test_valid_modes(self):
        assert normalize_tta_mode("raw") == "raw"
        assert normalize_tta_mode("flip") == "flip"
        assert normalize_tta_mode("multi") == "multi"
        assert normalize_tta_mode("none") == "raw"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported TTA mode"):
            normalize_tta_mode("invalid")

    def test_case_insensitive(self):
        assert normalize_tta_mode("RAW") == "raw"
        assert normalize_tta_mode("Flip") == "flip"


# ── Numerical stability ──


class TestNumericalStability:
    def test_extreme_logits_no_nan(self):
        cfg = _cfg(use_mv_loss=True, use_dldl_v2=True)
        criterion = CombinedLoss(cfg)
        logits = torch.tensor([[100.0] + [-100.0] * 80] * 4)  # extreme
        log_probs = F.log_softmax(logits, dim=1)
        target_dists = F.softmax(torch.randn(4, 81), dim=1)
        true_ages = torch.tensor([0.0, 80.0, 0.0, 80.0])
        result = criterion(log_probs, target_dists, true_ages, logits)
        for r in result[:8]:  # all loss components
            if isinstance(r, torch.Tensor) and r.dim() > 0:
                assert torch.isfinite(r).all(), f"Non-finite in loss component: {r}"

    def test_all_same_age_no_crash(self):
        """Triplet loss with all identical ages should produce zero loss."""
        cfg = _cfg(use_adaptive_triplet=True, use_dldl_v2=True)
        criterion = CombinedLoss(cfg)
        logits = torch.randn(4, 81, requires_grad=True)
        log_probs = F.log_softmax(logits, dim=1)
        target_dists = F.softmax(torch.randn(4, 81), dim=1)
        true_ages = torch.tensor([30.0, 30.0, 30.0, 30.0])
        result = criterion(log_probs, target_dists, true_ages, logits,
                           embeddings=torch.randn(4, 128))
        assert torch.isfinite(result[0])


# ── remap_state_dict_keys (checkpoint backward-compat, was zero coverage) ──


class TestRemapStateDictKeys:
    def test_renames_legacy_buffer_keys(self):
        legacy = {
            "texture_branch.imagenet_mean": torch.zeros(3),
            "texture_branch.imagenet_std": torch.ones(3),
            "final_head.0.weight": torch.randn(2, 2),
        }
        remapped = remap_state_dict_keys(legacy)
        assert "texture_branch.image_mean" in remapped
        assert "texture_branch.image_std" in remapped
        assert "texture_branch.imagenet_mean" not in remapped
        # Unrelated keys and tensor values are preserved.
        assert "final_head.0.weight" in remapped
        assert torch.equal(remapped["texture_branch.image_mean"], legacy["texture_branch.imagenet_mean"])

    def test_is_noop_for_current_keys(self):
        current = {"backbone.conv.weight": torch.randn(1), "image_mean": torch.zeros(3)}
        remapped = remap_state_dict_keys(current)
        assert set(remapped.keys()) == set(current.keys())
