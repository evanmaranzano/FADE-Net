import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import Config
from dataset import get_stratified_split
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
