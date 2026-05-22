import torch
from src.utils import AsymmetricOrdinalLoss


def test_asymmetric_loss_shape():
    """Asymmetric loss should produce scalar."""
    loss_fn = AsymmetricOrdinalLoss(under_weight=2.0, over_weight=1.0)
    pred = torch.tensor([25.0, 35.0, 50.0])
    true = torch.tensor([30.0, 30.0, 30.0])
    loss = loss_fn(pred, true)
    assert loss.dim() == 0


def test_asymmetric_underestimate_penalty():
    """Underestimation should be penalized more heavily when under_weight > over_weight."""
    loss_fn = AsymmetricOrdinalLoss(under_weight=3.0, over_weight=1.0)
    pred_under = torch.tensor([20.0])
    pred_over = torch.tensor([40.0])
    true_age = torch.tensor([30.0])
    loss_under = loss_fn(pred_under, true_age)
    loss_over = loss_fn(pred_over, true_age)
    assert loss_under.item() > loss_over.item(), "Underestimate should have higher loss"


def test_asymmetric_symmetric_mode():
    """When weights are equal, loss should be symmetric."""
    loss_fn = AsymmetricOrdinalLoss(under_weight=1.0, over_weight=1.0)
    pred_under = torch.tensor([20.0])
    pred_over = torch.tensor([40.0])
    true_age = torch.tensor([30.0])
    loss_under = loss_fn(pred_under, true_age)
    loss_over = loss_fn(pred_over, true_age)
    assert abs(loss_under.item() - loss_over.item()) < 1e-5
