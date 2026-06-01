"""Tests for model components, Config validation, SafeRandomErasing, and LossOutput.

Covers:
- CoordAtt (was zero coverage)
- BottleneckSPP (was zero coverage)
- SobelTextureExtractor (was zero coverage)
- SafeRandomErasing (was zero coverage)
- Config.validate() (was missing)
- LossOutput namedtuple (was missing)
- DLDLProcessor expectation_regression
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import Config
from model import BottleneckSPP, CoordAtt, SobelTextureExtractor
from dataset import SafeRandomErasing
from utils import LossOutput, DLDLProcessor, get_dldl_processor


def _cfg(**overrides):
    cfg = Config()
    cfg.device = torch.device("cpu")
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# ── CoordAtt ──


class TestCoordAtt:
    def test_output_shape_matches_input(self):
        attn = CoordAtt(64, 64, reduction=16)
        x = torch.randn(2, 64, 14, 14)
        out = attn(x)
        assert out.shape == x.shape

    def test_output_finite(self):
        attn = CoordAtt(128, 128, reduction=32)
        x = torch.randn(4, 128, 7, 7)
        out = attn(x)
        assert torch.isfinite(out).all()

    def test_asymmetric_spatial(self):
        attn = CoordAtt(32, 32, reduction=8)
        x = torch.randn(1, 32, 10, 20)
        out = attn(x)
        assert out.shape == (1, 32, 10, 20)

    def test_gradient_flows(self):
        attn = CoordAtt(16, 16, reduction=8)
        x = torch.randn(2, 16, 8, 8, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None


# ── BottleneckSPP ──


class TestBottleneckSPP:
    def test_output_shape(self):
        spp = BottleneckSPP(512, 256)
        x = torch.randn(2, 512, 7, 7)
        out = spp(x)
        assert out.shape == (2, 256, 7, 7)

    def test_different_spatial_sizes(self):
        spp = BottleneckSPP(256, 128)
        for size in [4, 7, 14]:
            x = torch.randn(1, 256, size, size)
            out = spp(x)
            assert out.shape == (1, 128, size, size)

    def test_gradient_flows(self):
        spp = BottleneckSPP(64, 32)
        x = torch.randn(2, 64, 7, 7, requires_grad=True)
        out = spp(x)
        out.sum().backward()
        assert x.grad is not None


# ── SobelTextureExtractor ──


class TestSobelTextureExtractor:
    def test_output_shape(self):
        sobel = SobelTextureExtractor()
        x = torch.randn(2, 1, 64, 64)
        out = sobel(x)
        assert out.shape == (2, 1, 64, 64)

    def test_output_non_negative(self):
        sobel = SobelTextureExtractor()
        x = torch.randn(1, 1, 32, 32)
        out = sobel(x)
        assert (out >= 0).all()  # sqrt(gx^2 + gy^2 + eps) >= 0

    def test_flat_image_has_low_response(self):
        sobel = SobelTextureExtractor()
        # Use a larger flat image so boundary effects are proportionally smaller
        x = torch.full((1, 1, 128, 128), 0.5)
        out = sobel(x)
        # Interior should be near zero (boundary padding causes non-zero edges)
        interior = out[:, :, 10:-10, 10:-10]
        assert interior.mean() < 0.01

    def test_no_trainable_params(self):
        sobel = SobelTextureExtractor()
        trainable = sum(p.numel() for p in sobel.parameters() if p.requires_grad)
        assert trainable == 0


# ── SafeRandomErasing ──


class TestSafeRandomErasing:
    def test_output_shape_preserved(self):
        erasing = SafeRandomErasing(p=1.0, scale=(0.1, 0.3), config=_cfg())
        img = torch.randn(3, 224, 224)
        out = erasing(img)
        assert out.shape == img.shape

    def test_p_zero_returns_unchanged(self):
        erasing = SafeRandomErasing(p=0.0, config=_cfg())
        img = torch.randn(3, 224, 224)
        out = erasing(img)
        assert torch.equal(out, img)

    def test_does_not_erase_all(self):
        erasing = SafeRandomErasing(p=1.0, scale=(0.01, 0.05), config=_cfg())
        img = torch.ones(3, 224, 224)
        out = erasing(img)
        # Most of the image should still be 1.0
        assert (out == 1.0).float().mean() > 0.8

    def test_creates_copy_not_inplace(self):
        erasing = SafeRandomErasing(p=1.0, inplace=False, config=_cfg())
        img = torch.randn(3, 224, 224)
        orig = img.clone()
        out = erasing(img)
        assert torch.equal(img, orig)  # original unchanged
        assert not torch.equal(out, img)  # output is different


# ── Config.validate() ──


class TestConfigValidate:
    def test_default_config_valid(self):
        cfg = _cfg()
        cfg.validate()  # should not raise

    def test_dropout_zero_is_valid(self):
        cfg = _cfg(dropout=0.0)
        cfg.validate()

    def test_invalid_num_classes(self):
        cfg = _cfg(min_age=0, max_age=80, num_classes=50)
        with pytest.raises(ValueError, match="num_classes"):
            cfg.validate()

    def test_invalid_sigma_range(self):
        cfg = _cfg(sigma_min=5.0, sigma_max=1.0)
        with pytest.raises(ValueError, match="sigma_min"):
            cfg.validate()

    def test_invalid_lr(self):
        cfg = _cfg(learning_rate=-0.001)
        with pytest.raises(ValueError, match="learning_rate"):
            cfg.validate()

    def test_invalid_batch_size(self):
        cfg = _cfg(batch_size=0)
        with pytest.raises(ValueError, match="batch_size"):
            cfg.validate()

    def test_invalid_dropout(self):
        cfg = _cfg(dropout=1.5)
        with pytest.raises(ValueError, match="dropout"):
            cfg.validate()

    def test_invalid_freeze_epochs(self):
        cfg = _cfg(freeze_backbone_epochs=-1)
        with pytest.raises(ValueError, match="freeze_backbone_epochs"):
            cfg.validate()


# ── LossOutput ──


class TestLossOutput:
    def test_is_tuple_subclass(self):
        assert issubclass(LossOutput, tuple)

    def test_positional_access(self):
        lo = LossOutput(total=1, kl=2, l1=3, rank=4, mv=5, triplet=6, asym=7, moe_gate=8, pred_age=9)
        assert lo[0] == 1
        assert lo[4] == 5
        assert lo[8] == 9

    def test_named_access(self):
        lo = LossOutput(total=1, kl=2, l1=3, rank=4, mv=5, triplet=6, asym=7, moe_gate=8, pred_age=9)
        assert lo.total == 1
        assert lo.moe_gate == 8

    def test_destructuring(self):
        lo = LossOutput(total=1, kl=2, l1=3, rank=4, mv=5, triplet=6, asym=7, moe_gate=8, pred_age=9)
        t, k, l, r, m, tr, a, mg, pa = lo
        assert t == 1 and pa == 9

    def test_len(self):
        lo = LossOutput(*range(9))
        assert len(lo) == 9


# ── DLDLProcessor.expectation_regression ──


class TestDLDLExpectationRegression:
    def test_one_hot_returns_exact_age(self):
        proc = DLDLProcessor(_cfg(use_dldl_v2=True))
        probs = torch.zeros(1, 81)
        probs[0, 42] = 1.0
        age = proc.expectation_regression(probs)
        assert abs(age.item() - 42.0) < 1e-5

    def test_batch_processing(self):
        proc = DLDLProcessor(_cfg(use_dldl_v2=True))
        probs = torch.softmax(torch.randn(4, 81), dim=1)
        ages = proc.expectation_regression(probs)
        assert ages.shape == (4,)
        assert (ages >= 0).all() and (ages <= 80).all()


class TestDLDLProcessorCache:
    def test_cache_reuses_identical_config(self):
        cfg = _cfg(use_dldl_v2=True, use_adaptive_sigma=True, sigma_min=1.0, sigma_max=3.0)

        assert get_dldl_processor(cfg) is get_dldl_processor(cfg)

    def test_cache_separates_sigma_parameters(self):
        cfg_a = _cfg(use_dldl_v2=True, use_adaptive_sigma=True, sigma_min=1.0, sigma_max=3.0)
        cfg_b = _cfg(use_dldl_v2=True, use_adaptive_sigma=True, sigma_min=2.0, sigma_max=4.0)

        proc_a = get_dldl_processor(cfg_a)
        proc_b = get_dldl_processor(cfg_b)

        assert proc_a is not proc_b
        assert proc_b.sigma_min == 2.0
