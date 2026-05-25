import torch
from src.config import Config
from src.model import LightweightAgeEstimator, AgeMoEHead
from src.utils import CombinedLoss


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


def test_moe_gate_loss_penalizes_batch_expert_collapse():
    """MoE routing loss should discourage sending the whole batch to one expert."""
    cfg = Config()
    cfg.use_moe = True
    cfg.use_dldl_v2 = False
    cfg.use_mv_loss = False
    cfg.lambda_moe_gate = 1.0
    cfg.lambda_moe_balance = 1.0
    criterion = CombinedLoss(cfg)
    logits = torch.zeros(3, cfg.num_classes)
    log_probs = torch.log_softmax(logits, dim=1)
    target_dists = torch.zeros(3, cfg.num_classes)
    target_dists[0, 5] = 1.0
    target_dists[1, 35] = 1.0
    target_dists[2, 70] = 1.0
    true_ages = torch.tensor([5.0, 35.0, 70.0])
    balanced_gate_logits = torch.tensor([
        [8.0, 0.0, 0.0],
        [0.0, 8.0, 0.0],
        [0.0, 0.0, 8.0],
    ])
    collapsed_gate_logits = torch.tensor([
        [8.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
    ])

    balanced = criterion(log_probs, target_dists, true_ages, logits, extras={"moe_gate_logits": balanced_gate_logits})
    collapsed = criterion(log_probs, target_dists, true_ages, logits, extras={"moe_gate_logits": collapsed_gate_logits})

    assert collapsed[7] > balanced[7]


def test_moe_batch_balance_loss_matches_soft_age_bin_usage():
    """Batch-level MoE regularization should match expected expert usage, not a flat prior."""
    cfg = Config()
    cfg.use_moe = True
    criterion = CombinedLoss(cfg)
    target_dists = torch.zeros(3, cfg.num_classes)
    target_dists[0, 5] = 1.0
    target_dists[1, 35] = 1.0
    target_dists[2, 70] = 1.0
    balanced_gate_probs = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    collapsed_gate_probs = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    balanced = criterion._moe_batch_balance_loss(balanced_gate_probs, target_dists, target_dists.device)
    collapsed = criterion._moe_batch_balance_loss(collapsed_gate_probs, target_dists, target_dists.device)

    assert collapsed > balanced


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
