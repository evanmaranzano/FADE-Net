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
    embeddings = torch.randn(4, 128)
    result = criterion(log_probs, target_dists, true_ages, logits, embeddings=embeddings)
    assert len(result) == 8, f"Expected 8 return values, got {len(result)}"
    total_loss = result[0]
    assert total_loss.dim() == 0
    assert total_loss.item() > 0
    assert result[6] > 0


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
    logits, embeddings, extras = model(x, return_features=True)
    log_probs = torch.log_softmax(logits, dim=1)
    target_dists = torch.softmax(torch.randn(4, 81), dim=1)
    true_ages = torch.tensor([10.0, 30.0, 50.0, 70.0])
    result = criterion(log_probs, target_dists, true_ages, logits, embeddings=embeddings, extras=extras)
    assert len(result) == 8
    total_loss = result[0]
    total_loss.backward()  # Verify gradients flow
    print(f"Full pipeline loss: {total_loss.item():.4f}")


def test_triplet_uses_feature_embeddings_when_provided():
    """Triplet term should be computed on model embeddings, not class logits."""
    cfg = _make_cfg(use_adaptive_triplet=True)
    criterion = CombinedLoss(cfg)
    logits = torch.zeros(4, 81)
    log_probs = torch.log_softmax(logits, dim=1)
    target_dists = torch.softmax(torch.randn(4, 81), dim=1)
    true_ages = torch.tensor([10.0, 11.0, 30.0, 60.0])
    close_embeddings = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.0],
        [5.0, 0.0],
        [9.0, 0.0],
    ])

    result_with_embeddings = criterion(log_probs, target_dists, true_ages, logits, embeddings=close_embeddings)
    result_with_logits = criterion(log_probs, target_dists, true_ages, logits)

    assert result_with_embeddings[5] != result_with_logits[5]


def test_moe_gate_loss_uses_age_bins():
    """MoE runs should expose a soft age-bin gate loss for age-aware routing."""
    cfg = _make_cfg(use_moe=True)
    criterion = CombinedLoss(cfg)
    model = LightweightAgeEstimator(cfg)
    x = torch.randn(4, 3, 224, 224)
    logits, embeddings, extras = model(x, return_features=True)
    log_probs = torch.log_softmax(logits, dim=1)
    target_dists = torch.softmax(torch.randn(4, 81), dim=1)
    true_ages = torch.tensor([5.0, 25.0, 55.0, 78.0])

    result = criterion(log_probs, target_dists, true_ages, logits, embeddings=embeddings, extras=extras)

    assert "moe_gate_logits" in extras
    assert result[7] > 0


def test_moe_gate_targets_follow_soft_label_distribution():
    cfg = _make_cfg(use_moe=True)
    criterion = CombinedLoss(cfg)
    target_dists = torch.zeros(1, cfg.num_classes)
    target_dists[0, 5] = 0.25
    target_dists[0, 55] = 0.75

    gate_targets = criterion._moe_gate_targets(target_dists, device=target_dists.device)

    assert torch.allclose(gate_targets.sum(dim=1), torch.ones(1))
    assert gate_targets[0, 0] > 0
    assert gate_targets[0, 2] > 0
