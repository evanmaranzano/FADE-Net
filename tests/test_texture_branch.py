import torch
from src.config import Config
from src.model import LightweightAgeEstimator, TextureEnhanceBranch


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


def test_texture_branch_uses_configured_normalization_contract():
    """Texture input inversion should follow the same mean/std used by dataset transforms."""
    mean = [0.1, 0.2, 0.3]
    std = [0.4, 0.5, 0.6]
    branch = TextureEnhanceBranch(image_mean=mean, image_std=std)

    assert torch.allclose(branch.image_mean.flatten(), torch.tensor(mean))
    assert torch.allclose(branch.image_std.flatten(), torch.tensor(std))


def test_model_texture_branch_inherits_config_normalization_contract():
    mean = [0.1, 0.2, 0.3]
    std = [0.4, 0.5, 0.6]
    cfg = Config()
    cfg.backbone_pretrained = False
    cfg.use_texture_branch = True
    cfg.use_multi_scale = True
    cfg.use_spp = True
    cfg.image_mean = mean
    cfg.image_std = std

    model = LightweightAgeEstimator(cfg)

    assert torch.allclose(model.texture_branch.image_mean.flatten(), torch.tensor(mean))
    assert torch.allclose(model.texture_branch.image_std.flatten(), torch.tensor(std))
