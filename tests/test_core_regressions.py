import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import Config
import dataset as dataset_module
from dataset import AFADDataset, get_dataloaders, get_stratified_split, my_collate_fn
from utils import CombinedLoss, DLDLProcessor, EMAModel


class _TinyDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx


def _cfg(**overrides):
    cfg = Config()
    cfg.device = torch.device("cpu")
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_stratified_split_keeps_rare_age_in_test_set(tmp_path):
    ages = [10.0, 10.0, 20.0, 20.0]
    train, val, test = get_stratified_split(
        _TinyDataset(len(ages)),
        ages,
        split_ratios=(0.72, 0.08, 0.20),
        save_path=str(tmp_path / "split.json"),
        dataset_hash="tiny-v1",
    )

    assert len(train.indices) == 2
    assert len(val.indices) == 0
    assert len(test.indices) == 2
    assert {int(ages[idx]) for idx in test.indices} == {10, 20}


def test_dldl_tensor_sigma_is_clamped_and_finite():
    processor = DLDLProcessor(_cfg(use_dldl_v2=True, use_adaptive_sigma=True, sigma_min=1.0, sigma_max=3.0))

    dist = processor.generate_label_distribution(torch.tensor(0.0), sigma_offset=torch.tensor(-1.0))

    assert torch.isfinite(dist).all()
    assert torch.isclose(dist.sum(), torch.tensor(1.0), atol=1e-6)


def test_dldl_label_distribution_uses_age_scalar_device():
    processor = DLDLProcessor(_cfg(use_dldl_v2=True, use_adaptive_sigma=True, sigma_min=1.0, sigma_max=3.0))

    dist = processor.generate_label_distribution(torch.tensor(10.0, device="meta"))

    assert dist.device.type == "meta"


def test_ema_registers_frozen_params_before_unfreeze():
    model = torch.nn.Linear(2, 1)
    model.weight.requires_grad = False
    ema = EMAModel(model, decay=0.9)
    initial_shadow = ema.shadow["weight"].clone()

    model.weight.requires_grad = True
    model.weight.data.add_(1.0)
    ema.update()

    assert "weight" in ema.shadow
    assert not torch.equal(ema.shadow["weight"], model.weight.data)
    assert torch.allclose(ema.shadow["weight"], 0.1 * model.weight.data + 0.9 * initial_shadow)


def test_ema_apply_shadow_rejects_double_apply():
    model = torch.nn.Linear(2, 1)
    ema = EMAModel(model)

    ema.apply_shadow()
    with pytest.raises(RuntimeError):
        ema.apply_shadow()
    ema.restore()


def test_asymmetric_ordinal_reports_l1_contribution_not_raw_l1():
    cfg = _cfg(use_asymmetric_ordinal=True)
    criterion = CombinedLoss(cfg)
    logits = torch.randn(4, cfg.num_classes)
    log_probs = torch.log_softmax(logits, dim=1)
    target_dists = torch.softmax(torch.randn(4, cfg.num_classes), dim=1)
    true_ages = torch.tensor([10.0, 30.0, 50.0, 70.0])

    result = criterion(log_probs, target_dists, true_ages, logits)

    assert result[2] == 0.0


def _afad_root_with_images(tmp_path, *images):
    image_dir = tmp_path / "10" / "111"
    image_dir.mkdir(parents=True)
    for name, good in images:
        path = image_dir / name
        if good:
            Image.new("RGB", (2, 2), color=(255, 0, 0)).save(path)
        else:
            path.write_bytes(b"not an image")
    return tmp_path


def test_afad_dataset_retries_on_failed_image(monkeypatch, tmp_path):
    import random as _random_mod
    root = _afad_root_with_images(tmp_path, ("bad.jpg", False), ("good.jpg", True))
    cfg = _cfg(min_age=0, max_age=100, num_classes=101, use_dldl_v2=False)
    dataset = AFADDataset(str(root), transform=lambda image: torch.ones(1), config=cfg)

    def always_return_1(low, high):
        return 1

    monkeypatch.setattr(_random_mod, "randint", always_return_1)

    result = dataset[0]
    assert result is not None
    image, label_dist, age = result
    assert image.shape == (1,)


def test_afad_dataset_returns_none_after_bounded_failed_fallbacks(monkeypatch, tmp_path):
    root = _afad_root_with_images(tmp_path, ("bad_a.jpg", False), ("bad_b.jpg", False))
    cfg = _cfg(min_age=0, max_age=100, num_classes=101, use_dldl_v2=False)
    dataset = AFADDataset(str(root), transform=lambda image: torch.ones(1), config=cfg)

    with pytest.warns(UserWarning, match="Failed to load image"):
        assert dataset[0] is None


def test_collate_filters_none_and_handles_all_empty_batches():
    sample = (torch.ones(1), torch.ones(3), torch.tensor(10.0))

    images, labels, ages = my_collate_fn([None, sample])
    empty_images, empty_labels, empty_ages = my_collate_fn([None, None])

    assert images.shape == (1, 1)
    assert labels.shape == (1, 3)
    assert ages.shape == (1,)
    assert empty_images.numel() == 0
    assert empty_labels.numel() == 0
    assert empty_ages.numel() == 0


def test_validation_transform_uses_config_img_size(monkeypatch, tmp_path):
    root = _afad_root_with_images(
        tmp_path,
        ("a.jpg", True),
        ("b.jpg", True),
        ("c.jpg", True),
    )
    cfg = _cfg(
        afad_dir=str(root),
        img_size=4,
        min_age=0,
        max_age=100,
        num_classes=101,
        batch_size=2,
        num_workers=0,
        use_dldl_v2=False,
        split_protocol="72-8-20",
        split_file_tag="pytest_imgsize",
        allow_legacy_split_upgrade=False,
    )
    monkeypatch.setattr(dataset_module, "ROOT_DIR", str(tmp_path))

    _train_loader, val_loader, test_loader, _class_weights = get_dataloaders(cfg)

    val_resize = next(t for t in val_loader.dataset.transform.transforms if isinstance(t, transforms.Resize))
    val_crop = next(t for t in val_loader.dataset.transform.transforms if isinstance(t, transforms.CenterCrop))
    test_resize = next(t for t in test_loader.dataset.transform.transforms if isinstance(t, transforms.Resize))
    test_crop = next(t for t in test_loader.dataset.transform.transforms if isinstance(t, transforms.CenterCrop))

    assert val_crop.size == (4, 4)
    assert val_resize.size == 4
    assert test_crop.size == (4, 4)
    assert test_resize.size == 4
