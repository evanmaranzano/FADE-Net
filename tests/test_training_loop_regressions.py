import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import train as train_module
from train import (
    SchedulerStepController,
    backbone_learning_rate,
    hard_distillation_start_epoch,
    model_forward_for_loss,
)


def test_backbone_learning_rate_uses_full_lr_without_loaded_pretraining():
    assert backbone_learning_rate(3e-4, effective_pretrained=False) == pytest.approx(3e-4)
    assert backbone_learning_rate(3e-4, effective_pretrained=True) == pytest.approx(3e-5)


class _RecorderModel:
    def __init__(self):
        self.calls = []

    def __call__(self, images, return_features=False):
        self.calls.append(return_features)
        logits = torch.zeros(images.size(0), 3)
        if return_features:
            return logits, torch.ones(images.size(0), 2), {"moe_gate_logits": torch.zeros(images.size(0), 2)}
        return logits


def test_model_forward_for_loss_avoids_aux_outputs_when_not_needed():
    model = _RecorderModel()
    cfg = SimpleNamespace(use_adaptive_triplet=False, use_moe=False)

    logits, embeddings, extras = model_forward_for_loss(model, cfg, torch.zeros(2, 1))

    assert model.calls == [False]
    assert logits.shape == (2, 3)
    assert embeddings is None
    assert extras is None


def test_model_forward_for_loss_requests_aux_outputs_for_triplet_or_moe():
    for cfg in (
        SimpleNamespace(use_adaptive_triplet=True, use_moe=False),
        SimpleNamespace(use_adaptive_triplet=False, use_moe=True),
    ):
        model = _RecorderModel()

        logits, embeddings, extras = model_forward_for_loss(model, cfg, torch.zeros(2, 1))

        assert model.calls == [True]
        assert logits.shape == (2, 3)
        assert embeddings.shape == (2, 2)
        assert "moe_gate_logits" in extras


def test_hard_distillation_start_epoch_scales_for_short_runs():
    assert hard_distillation_start_epoch(120) == 105
    assert hard_distillation_start_epoch(100) == 85
    assert hard_distillation_start_epoch(10) == 0


class _Scheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def test_scheduler_controller_replays_deferred_steps_after_amp_skip():
    scheduler = _Scheduler()
    controller = SchedulerStepController(scheduler, max_epochs=100)

    assert controller.step_epoch(epoch=0, optimizer_stepped=False) == 0
    assert scheduler.steps == 0
    assert controller.step_epoch(epoch=1, optimizer_stepped=True) == 2
    assert scheduler.steps == 2
    assert controller.pending_steps == 0


def test_scheduler_controller_ignores_stable_phase_epochs():
    scheduler = _Scheduler()
    controller = SchedulerStepController(scheduler, max_epochs=100)

    assert controller.step_epoch(epoch=100, optimizer_stepped=True) == 0
    assert scheduler.steps == 0


class _SmokeDataset:
    transform = "eval-transform"
    augment_label = True

    def __len__(self):
        return 2


class _SmokeLoader:
    def __init__(self):
        self.dataset = _SmokeDataset()
        self.collate_fn = None

    def __iter__(self):
        images = torch.ones(2, 1)
        target_dists = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        true_ages = torch.zeros(2)
        yield images, target_dists, true_ages

    def __len__(self):
        return 1


class _SmokeConfig:
    ACADEMIC_SEEDS = {}

    def __init__(self):
        self.epochs = 1
        self.batch_size = 2
        self.split_protocol = "72-8-20"
        self.freeze_backbone_epochs = 0
        self.backbone_source = "torchvision"
        self.backbone_name = "mobilenet_v3_small"
        self.backbone_pretrained = True
        self.use_ema = False
        self.ema_decay = 0.999
        self.learning_rate = 3e-4
        self.weight_decay = 0.0
        self.device = torch.device("cpu")
        self.num_classes = 3
        self.min_age = 0
        self.max_age = 2
        self.img_size = 1
        self.use_mixup = False
        self.mixup_prob = 0.0
        self.mixup_alpha = 0.5
        self.use_sigma_jitter = True
        self.use_adaptive_triplet = False
        self.use_moe = False
        self.use_asymmetric_ordinal = False
        self.use_mv_loss = False
        self.use_dldl_v2 = False


class _SmokeModel(torch.nn.Module):
    instances = []

    def __init__(self, cfg):
        super().__init__()
        self.backbone = torch.nn.Linear(1, 1)
        self.head = torch.nn.Linear(1, cfg.num_classes)
        self.backbone.pretrained_loaded = False
        self.return_features_calls = []
        _SmokeModel.instances.append(self)

    def forward(self, images, return_features=False):
        self.return_features_calls.append(return_features)
        logits = self.head(images.float())
        if return_features:
            return logits, logits, {}
        return logits


class _SmokeCriterion(torch.nn.Module):
    def to(self, device):
        return self

    def forward(self, log_probs, target_dists, true_ages, logits, embeddings=None, extras=None):
        loss = logits.sum() * 0 + 1.0
        return loss, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


class _SmokeWriter:
    instances = []

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.closed = False
        _SmokeWriter.instances.append(self)

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        self.closed = True


def _args(**overrides):
    values = {
        "seed": 42,
        "epochs": 1,
        "batch_size": None,
        "split": None,
        "freeze": None,
        "overwrite_artifacts": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_smoke_training(monkeypatch, tmp_path):
    cfg = _SmokeConfig()
    train_loader = _SmokeLoader()
    val_loader = _SmokeLoader()
    test_loader = _SmokeLoader()
    recorded_lrs = []
    hard_calls = []
    controller_calls = []

    _SmokeModel.instances = []
    _SmokeWriter.instances = []

    monkeypatch.setattr(train_module, "Config", lambda: cfg)
    monkeypatch.setattr(train_module, "get_dataloaders", lambda cfg: (train_loader, val_loader, test_loader, None))
    monkeypatch.setattr(train_module, "LightweightAgeEstimator", _SmokeModel)
    monkeypatch.setattr(train_module, "CombinedLoss", lambda cfg, weights=None: _SmokeCriterion())
    monkeypatch.setattr(train_module, "SummaryWriter", _SmokeWriter)
    monkeypatch.setattr(train_module, "build_training_metadata", lambda cfg, seed: {
        "experiment_id": "smoke",
        "backbone": {"pretrained": cfg.backbone_pretrained},
        "selection_metric": {"tta": "multi"},
    })
    monkeypatch.setattr(train_module, "artifact_path", lambda root, name, cfg, seed, suffix: str(tmp_path / f"{name}_{seed}{suffix}"))
    monkeypatch.setattr(train_module, "save_model_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "predict_probs", lambda model, images, mode, base_size: torch.tensor([[1.0, 0.0, 0.0]]).repeat(images.size(0), 1))
    monkeypatch.setattr(train_module, "probs_to_ages", lambda probs, num_classes: torch.zeros(probs.size(0)))
    monkeypatch.setattr(train_module, "evaluate_mae", lambda *args, **kwargs: {"raw": 0.0, "flip": 0.0, "multi": 0.0})

    original_adamw = train_module.optim.AdamW

    def recording_adamw(params, *args, **kwargs):
        params = list(params)
        recorded_lrs.append([group["lr"] for group in params])
        return original_adamw(params, *args, **kwargs)

    monkeypatch.setattr(train_module.optim, "AdamW", recording_adamw)

    class RecordingSchedulerController:
        def __init__(self, scheduler, max_epochs):
            self.max_epochs = max_epochs

        def step_epoch(self, epoch, optimizer_stepped):
            controller_calls.append((epoch, optimizer_stepped))
            return 1

    monkeypatch.setattr(train_module, "SchedulerStepController", RecordingSchedulerController)

    def recording_hard_mode(cfg, train_loader_arg, val_loader_arg):
        hard_calls.append((cfg, train_loader_arg, val_loader_arg))
        return train_loader_arg

    monkeypatch.setattr(train_module, "apply_hard_distillation_mode", recording_hard_mode)
    return SimpleNamespace(cfg=cfg, recorded_lrs=recorded_lrs, hard_calls=hard_calls, controller_calls=controller_calls)


def test_train_uses_full_backbone_lr_and_avoids_aux_outputs(monkeypatch, tmp_path):
    state = _patch_smoke_training(monkeypatch, tmp_path)

    train_module.train(_args())

    assert state.recorded_lrs[0][0] == pytest.approx(state.cfg.learning_rate)
    assert _SmokeModel.instances[0].return_features_calls == [False]


def test_train_uses_short_run_hard_distillation_and_scheduler_controller(monkeypatch, tmp_path):
    state = _patch_smoke_training(monkeypatch, tmp_path)

    train_module.train(_args())

    assert len(state.hard_calls) == 1
    assert state.controller_calls == [(0, True)]


def test_train_closes_summary_writer_when_training_raises(monkeypatch, tmp_path):
    _patch_smoke_training(monkeypatch, tmp_path)

    class RaisingCriterion(_SmokeCriterion):
        def forward(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(train_module, "CombinedLoss", lambda cfg, weights=None: RaisingCriterion())

    with pytest.raises(RuntimeError, match="boom"):
        train_module.train(_args())

    assert _SmokeWriter.instances[0].closed
