import copy
import sys
from pathlib import Path

import torch
from torch import nn


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from config import Config
from dcsr_cgbr import CGBR, DCSR, FADELoss
from fade_net import FADENet
from eval_fade_net_tta import TTA_VIEW_ORDER, make_ordered_views
from train_fade_net import (
    build_adaptive_sigma_table,
    get_transforms,
    restore_training_state,
    update_ema,
)


def test_ema_updates_parameters_and_batchnorm_buffers():
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)

    with torch.no_grad():
        model[0].weight.add_(1.0)
        model[1].running_mean.add_(2.0)

    updates = update_ema(ema_model, model, decay=0.999, num_updates=0)

    assert updates == 1
    assert not torch.equal(ema_model[0].weight, model[0].weight)
    assert torch.equal(ema_model[1].running_mean, model[1].running_mean)


def test_restore_training_state_restores_ema_optimizer_and_resume_lrs(tmp_path):
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    ema_model = copy.deepcopy(model)
    optimizer = torch.optim.AdamW([
        {'params': model[0].parameters(), 'lr': 3e-5},
        {'params': model[1].parameters(), 'lr': 3e-4},
    ])
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        for parameter in ema_model.parameters():
            parameter.add_(2.0)

    expected_model = copy.deepcopy(model.state_dict())
    expected_ema = copy.deepcopy(ema_model.state_dict())
    checkpoint_path = tmp_path / 'resume.pth'
    torch.save({
        'epoch': 21,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_mae': 3.2827,
        'ema_updates': 123,
    }, checkpoint_path)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for parameter in ema_model.parameters():
            parameter.zero_()

    checkpoint = restore_training_state(
        checkpoint_path, model, ema_model, optimizer, 3e-6, 3e-5
    )

    assert checkpoint['epoch'] == 21
    assert checkpoint['ema_updates'] == 123
    assert optimizer.param_groups[0]['lr'] == 3e-6
    assert optimizer.param_groups[1]['lr'] == 3e-5
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected_model[name])
    for name, value in ema_model.state_dict().items():
        assert torch.equal(value, expected_ema[name])


def test_fade_net_forward_contract_without_pretrained_download():
    config = Config()
    config.validate()
    assert (config.min_age, config.max_age, config.num_classes) == (0, 80, 81)
    assert (config.data_min_age, config.data_max_age) == (15, 72)
    config.backbone_pretrained = False
    config.img_size = 64
    config.device = torch.device("cpu")

    model = FADENet(config).eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 64, 64))

    assert outputs["age"].shape == (1,)
    assert outputs["base_age"].shape == (1,)
    assert outputs["main_prob"].shape == (1, config.num_classes)
    assert outputs["coarse_prob"].shape == (1, config.num_classes)
    assert torch.isfinite(outputs["age"]).all()


def test_fade_net_probes_medium_backbone_channels():
    config = Config()
    config.backbone_pretrained = False
    config.backbone_name = "mobilenetv4_conv_medium"
    config.img_size = 64

    model = FADENet(config).eval()

    assert model.feature_indices == [1, 3]
    assert model.feature_channels == [48, 160, 960]
    assert model.adapters[0].conv1x1.in_channels == 48
    assert model.adapters[1].conv1x1.in_channels == 160


def test_distribution_encoders_receive_gradients_after_probability_detach():
    dcsr = DCSR(fusion_channels=8, route_groups=2, num_classes=81, min_age=0, max_age=80)
    coarse_probs = torch.softmax(torch.randn(2, 81), dim=1)
    f1 = torch.randn(2, 8, 8, 8)
    f2 = torch.randn(2, 8, 4, 4)
    f3 = torch.randn(2, 8, 2, 2)
    _, route_weights = dcsr(f1, f2, f3, coarse_probs)
    route_weights[..., 0].mean().backward()
    assert dcsr.dist_encoder.mlp[0].weight.grad is not None

    cgbr = CGBR(in_channels=8, num_classes=81, min_age=0, max_age=80)
    fused = torch.randn(2, 8, 4, 4, requires_grad=True)
    main_probs = torch.softmax(torch.randn(2, 81), dim=1)
    gate, residual, refined_age, _ = cgbr(fused, main_probs)
    (gate.mean() + residual.mean() + refined_age.mean()).backward()
    assert cgbr.main_dist_encoder.mlp[0].weight.grad is not None


def test_cdf_distance_is_zero_for_match_and_larger_for_farther_mass():
    target = torch.zeros(1, 5)
    target[0, 2] = 1.0
    exact = target.clone()
    near = torch.zeros(1, 5)
    near[0, 3] = 1.0
    far = torch.zeros(1, 5)
    far[0, 4] = 1.0

    assert FADELoss._cdf_distance(exact, target).item() == 0.0
    assert FADELoss._cdf_distance(far, target) > FADELoss._cdf_distance(near, target)


def test_tta_view_order_preserves_1x_2x_and_builds_six_views():
    images = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    views = make_ordered_views(images, image_size=8)

    assert len(views) == 6
    assert TTA_VIEW_ORDER[0] == {"scale": 1.0, "flip": False}
    assert TTA_VIEW_ORDER[1] == {"scale": 1.0, "flip": True}
    assert torch.equal(views[0], images)
    assert torch.equal(views[1], torch.flip(images, dims=[3]))
    assert all(view.shape == images.shape for view in views)


def test_adaptive_sigma_uses_training_frequency_and_broadens_rare_age():
    samples = [
        {"age": 15}, {"age": 15}, {"age": 15}, {"age": 15}, {"age": 16}
    ]
    sigmas, details = build_adaptive_sigma_table(
        samples, range(len(samples)), 0, 80, base_sigma=2.0, max_sigma=3.0
    )

    assert sigmas[15] == 2.0
    assert sigmas[16] == 3.0
    assert sigmas[14] == 2.0
    assert details["train_count_by_age"] == {"15": 4, "16": 1}

    criterion = FADELoss(min_age=0, max_age=80, label_sigma_by_age=sigmas)
    distributions = criterion._gaussian_label(torch.tensor([15.0, 16.0]))
    age_values = torch.arange(81, dtype=torch.float32)
    variances = (
        distributions * (age_values.unsqueeze(0) - torch.tensor([[15.0], [16.0]])) ** 2
    ).sum(dim=1)
    assert variances[1] > variances[0]


def test_random_erasing_probability_is_explicitly_configurable():
    default_transform = get_transforms(img_size=64, is_train=True)
    disabled_transform = get_transforms(
        img_size=64, is_train=True, random_erasing_p=0.0
    )

    assert default_transform.transforms[-1].p == 0.1
    assert disabled_transform.transforms[-1].p == 0.0


def test_training_crop_scale_is_explicitly_configurable():
    default_transform = get_transforms(img_size=64, is_train=True)
    wider_scale_transform = get_transforms(
        img_size=64, is_train=True, train_crop_scale_min=0.7
    )

    assert default_transform.transforms[0].scale == (0.8, 1.0)
    assert wider_scale_transform.transforms[0].scale == (0.7, 1.0)
