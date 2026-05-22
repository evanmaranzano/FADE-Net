import torch
from src.config import Config
from src.model import LightweightAgeEstimator
from src.utils import CombinedLoss


def _make_cfg(**overrides):
    cfg = Config()
    cfg.img_size = 224
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_all_model_modules_enabled():
    """All 3 model modules (texture, freq, moe) enabled should produce correct output."""
    cfg = _make_cfg(
        use_texture_branch=True,
        use_freq_attention=True,
        use_moe=True,
    )
    model = LightweightAgeEstimator(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, cfg.num_classes)


def test_all_loss_modules_enabled():
    """Both loss modules (triplet, asymmetric) should work with CombinedLoss."""
    cfg = _make_cfg(
        use_adaptive_triplet=True,
        use_asymmetric_ordinal=True,
    )
    criterion = CombinedLoss(cfg)
    logits = torch.randn(4, 81)
    log_probs = torch.log_softmax(logits, dim=1)
    target_dists = torch.softmax(torch.randn(4, 81), dim=1)
    true_ages = torch.tensor([10.0, 30.0, 50.0, 70.0])
    result = criterion(log_probs, target_dists, true_ages, logits)
    assert len(result) == 6, f"Expected 6 return values, got {len(result)}"
    total_loss = result[0]
    assert total_loss.dim() == 0
    assert total_loss.item() > 0


def test_each_model_module_independently():
    """Each model module should work independently."""
    for module in ['use_texture_branch', 'use_freq_attention', 'use_moe']:
        cfg = _make_cfg(**{module: True})
        model = LightweightAgeEstimator(cfg)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, cfg.num_classes), f"{module} failed"


def test_no_new_modules_baseline():
    """With all new modules disabled, model should match original behavior."""
    cfg = _make_cfg()
    model = LightweightAgeEstimator(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, cfg.num_classes)
    # Verify no new modules are attached
    assert not getattr(model, 'use_texture_branch', False)
    assert not getattr(model, 'use_freq_attention', False)
    assert not getattr(model, 'use_moe', False)


def test_model_plus_loss_integration():
    """Full forward+loss pipeline with all modules enabled."""
    cfg = _make_cfg(
        use_texture_branch=True,
        use_freq_attention=True,
        use_moe=True,
        use_adaptive_triplet=True,
        use_asymmetric_ordinal=True,
    )
    model = LightweightAgeEstimator(cfg)
    criterion = CombinedLoss(cfg)
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    log_probs = torch.log_softmax(logits, dim=1)
    target_dists = torch.softmax(torch.randn(4, 81), dim=1)
    true_ages = torch.tensor([10.0, 30.0, 50.0, 70.0])
    result = criterion(log_probs, target_dists, true_ages, logits)
    assert len(result) == 6
    total_loss = result[0]
    total_loss.backward()  # Verify gradients flow
    print(f"Full pipeline loss: {total_loss.item():.4f}")
