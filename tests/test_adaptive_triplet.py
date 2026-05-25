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
    loss_fn = AdaptiveTripletLoss(base_margin=0.2, alpha=0.01, max_margin=0.5)
    age_diff = torch.tensor([2.0, 20.0])
    m1, m2 = loss_fn._adaptive_margin(age_diff)
    assert m2 > m1, "Margin should scale with age difference"


def test_adaptive_triplet_default_margin_is_bounded_for_normalized_embeddings():
    """Default dynamic margins must stay feasible on unit-normalized features."""
    loss_fn = AdaptiveTripletLoss()
    margin = loss_fn._adaptive_margin(torch.tensor([0.0, 20.0, 80.0]))

    assert margin.max().item() <= loss_fn.max_margin
    assert loss_fn.max_margin < 2.0


def test_adaptive_triplet_small_input():
    """Should handle small batch gracefully."""
    loss_fn = AdaptiveTripletLoss(base_margin=1.0, alpha=0.1)
    embeddings = torch.randn(1, 128)
    ages = torch.tensor([30.0])
    loss = loss_fn(embeddings, ages)
    assert loss.item() == 0.0, "Single sample should produce zero loss"


def test_adaptive_triplet_negative_mask_uses_anchor_axis():
    """Negative samples must be selected by anchor age, not positive-pair age."""
    loss_fn = AdaptiveTripletLoss(base_margin=1.0, alpha=0.0, age_threshold=3.0)
    embeddings = torch.tensor([
        [0.0],
        [0.5],
        [4.0],
    ])
    ages = torch.tensor([10.0, 11.0, 12.0])

    loss = loss_fn(embeddings, ages)

    assert loss.item() == 0.0, "No anchor has a valid negative sample, so loss must be zero"


def test_adaptive_triplet_is_invariant_to_embedding_scale():
    """Triplet term should not change when upstream feature norms change."""
    loss_fn = AdaptiveTripletLoss(base_margin=0.2, alpha=0.0, age_threshold=3.0)
    embeddings = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],
        [-1.0, 0.0],
        [-0.9, -0.1],
    ])
    ages = torch.tensor([10.0, 11.0, 40.0, 41.0])

    base_loss = loss_fn(embeddings, ages)
    scaled_loss = loss_fn(embeddings * 100.0, ages)

    assert torch.allclose(base_loss, scaled_loss, atol=1e-6)
