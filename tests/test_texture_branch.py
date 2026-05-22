import torch
from src.config import Config
from src.model import LightweightAgeEstimator


def test_texture_branch_forward():
    """Texture branch should produce output with correct shape."""
    cfg = Config()
    cfg.use_texture_branch = True
    cfg.use_multi_scale = True
    cfg.use_spp = True
    cfg.img_size = 224
    model = LightweightAgeEstimator(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, cfg.num_classes), f"Expected (2, {cfg.num_classes}), got {out.shape}"


def test_texture_branch_disabled():
    """Without texture branch, model should work normally."""
    cfg = Config()
    cfg.use_texture_branch = False
    cfg.use_multi_scale = True
    cfg.use_spp = True
    cfg.img_size = 224
    model = LightweightAgeEstimator(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, cfg.num_classes)


def test_texture_branch_adds_fusion_dim():
    """Texture branch should increase classifier input dim when enabled."""
    cfg1 = Config()
    cfg1.use_texture_branch = False
    cfg1.use_multi_scale = True
    cfg1.use_spp = True
    cfg1.img_size = 224
    m1 = LightweightAgeEstimator(cfg1)

    cfg2 = Config()
    cfg2.use_texture_branch = True
    cfg2.use_multi_scale = True
    cfg2.use_spp = True
    cfg2.img_size = 224
    m2 = LightweightAgeEstimator(cfg2)

    dim1 = m1.final_head[0].in_features
    dim2 = m2.final_head[0].in_features
    assert dim2 > dim1, f"Texture branch should increase head dim: {dim1} vs {dim2}"
