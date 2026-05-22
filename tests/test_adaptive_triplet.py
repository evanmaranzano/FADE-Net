import torch
from src.utils import AdaptiveTripletLoss


def test_adaptive_triplet_basic():
    """Adaptive triplet loss should produce scalar loss."""
    loss_fn = AdaptiveTripletLoss(base_margin=1.0, alpha=0.1)
    embeddings = torch.randn(6, 128)
    ages = torch.tensor([10.0, 12.0, 30.0, 32.0, 60.0, 62.0])
    loss = loss_fn(embeddings, ages)
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"


def test_adaptive_triplet_margin_scales():
    """Larger age difference should produce larger effective margin."""
    loss_fn = AdaptiveTripletLoss(base_margin=1.0, alpha=0.1)
    m1 = loss_fn.base_margin * (1 + loss_fn.alpha * 2)
    m2 = loss_fn.base_margin * (1 + loss_fn.alpha * 20)
    assert m2 > m1, "Margin should scale with age difference"


def test_adaptive_triplet_small_input():
    """Should handle small batch gracefully."""
    loss_fn = AdaptiveTripletLoss(base_margin=1.0, alpha=0.1)
    embeddings = torch.randn(1, 128)
    ages = torch.tensor([30.0])
    loss = loss_fn(embeddings, ages)
    assert loss.item() == 0.0, "Single sample should produce zero loss"
