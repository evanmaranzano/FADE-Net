import torch
from src.config import Config
from src.model import LightweightAgeEstimator, AgeMoEHead


def test_moe_head_shape():
    """MoE head should output correct shape."""
    head = AgeMoEHead(512, num_classes=81, num_experts=3)
    x = torch.randn(4, 512)
    out = head(x)
    assert out.shape == (4, 81)


def test_moe_head_routing():
    """Gate should produce valid probability distribution."""
    head = AgeMoEHead(512, num_classes=81, num_experts=3)
    x = torch.randn(8, 512)
    gate_logits = head.gate(x)
    gate = torch.softmax(gate_logits, dim=1)
    assert torch.allclose(gate.sum(dim=1), torch.ones(8), atol=1e-5)


def test_moe_in_model():
    """Model with MoE should produce correct output."""
    cfg = Config()
    cfg.use_moe = True
    cfg.use_spp = True
    cfg.img_size = 224
    model = LightweightAgeEstimator(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, cfg.num_classes)


def test_moe_model_returns_gate_logits_for_loss():
    """MoE model exposes gate logits when training asks for features."""
    cfg = Config()
    cfg.use_moe = True
    cfg.use_spp = True
    cfg.img_size = 224
    model = LightweightAgeEstimator(cfg)
    x = torch.randn(2, 3, 224, 224)
    logits, features, extras = model(x, return_features=True)
    assert logits.shape == (2, cfg.num_classes)
    assert features.shape[0] == 2
    assert extras["moe_gate_logits"].shape == (2, cfg.moe_num_experts)


def test_moe_disabled():
    """Model without MoE should use standard head."""
    cfg = Config()
    cfg.use_moe = False
    cfg.use_spp = True
    cfg.img_size = 224
    model = LightweightAgeEstimator(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, cfg.num_classes)
