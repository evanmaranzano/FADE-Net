import math
import sys
from pathlib import Path

import pytest
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import Config  # noqa: E402
from dcsr_cgbr import FADELoss  # noqa: E402
from fade_net import FADENet  # noqa: E402
from teacher_vit import build_teacher  # noqa: E402
from train_fade_net import (  # noqa: E402
    CLIP_NORM_MEAN,
    CLIP_NORM_STD,
    IMAGENET_NORM_MEAN,
    IMAGENET_NORM_STD,
    TEACHER_INPUT_SIZE,
    compute_kd_loss,
    student_to_teacher_input,
    train_one_epoch,
)

TEACHER_CHECKPOINT = (
    ROOT_DIR
    / "server_recovery"
    / "2026-07-17"
    / "host_restore_20260718"
    / "exp030_final"
    / "best_checkpoint.pth"
)


class _UniformTeacher(torch.nn.Module):
    """Mock teacher returning a uniform main distribution for any input."""

    def forward(self, images):
        probs = torch.full((images.shape[0], 81), 1.0 / 81.0)
        return {"main_prob": probs}


def _run_one_training_batch(teacher_model, lambda_kd):
    """Run a single light CPU training batch and return the mean loss."""
    torch.manual_seed(0)
    config = Config()
    config.validate()
    config.backbone_pretrained = False
    config.img_size = 64
    model = FADENet(config)
    criterion = FADELoss(min_age=0, max_age=80, label_sigma=2.0)
    optimizer = torch.optim.AdamW(model.get_params_groups(3e-5, 3e-4))
    images = torch.randn(2, 3, 64, 64)
    ages = torch.tensor([30.0, 45.0])
    loader = [(images, ages, torch.tensor([0, 1]))]

    torch.manual_seed(1)
    train_loss, _ = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        torch.device("cpu"),
        epoch=0,
        cgbr_start_epoch=16,
        cgbr_full_epoch=26,
        teacher_model=teacher_model,
        lambda_kd=lambda_kd,
    )
    return train_loss


def test_student_to_teacher_input_denormalizes_and_renormalizes():
    pixels = torch.rand(2, 3, 224, 224)  # already in [0, 1]
    imagenet_mean = torch.tensor(IMAGENET_NORM_MEAN).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_NORM_STD).view(1, 3, 1, 1)
    clip_mean = torch.tensor(CLIP_NORM_MEAN).view(1, 3, 1, 1)
    clip_std = torch.tensor(CLIP_NORM_STD).view(1, 3, 1, 1)
    student_input = (pixels - imagenet_mean) / imagenet_std

    teacher_input = student_to_teacher_input(student_input)

    # 224 -> 224 bilinear interpolation is the identity, so only the
    # normalization constants change the values.
    expected = (pixels.clamp(0.0, 1.0) - clip_mean) / clip_std
    assert teacher_input.shape == (2, 3, TEACHER_INPUT_SIZE, TEACHER_INPUT_SIZE)
    assert torch.allclose(teacher_input, expected, atol=1e-6)


def test_student_to_teacher_input_resizes_256_to_224_with_constant_image():
    pixel_value = 0.4
    pixels = torch.full((1, 3, 256, 256), pixel_value)
    imagenet_mean = torch.tensor(IMAGENET_NORM_MEAN).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_NORM_STD).view(1, 3, 1, 1)
    clip_mean = torch.tensor(CLIP_NORM_MEAN).view(1, 3, 1, 1)
    clip_std = torch.tensor(CLIP_NORM_STD).view(1, 3, 1, 1)
    student_input = (pixels - imagenet_mean) / imagenet_std

    teacher_input = student_to_teacher_input(student_input)

    # A constant image stays constant under bilinear resize, so each channel
    # must equal the hand-computed renormalized value everywhere.
    expected_value = (pixel_value - clip_mean) / clip_std
    assert teacher_input.shape == (1, 3, TEACHER_INPUT_SIZE, TEACHER_INPUT_SIZE)
    assert torch.allclose(
        teacher_input,
        expected_value.expand_as(teacher_input),
        atol=1e-6,
    )


def test_lambda_kd_zero_matches_path_without_teacher():
    loss_without_teacher = _run_one_training_batch(None, 0.0)
    loss_with_mock_teacher = _run_one_training_batch(_UniformTeacher(), 0.0)

    assert loss_without_teacher == pytest.approx(
        loss_with_mock_teacher, rel=0, abs=1e-7
    )


def test_lambda_kd_positive_adds_kl_to_total():
    loss_without_teacher = _run_one_training_batch(None, 0.0)
    loss_with_kd = _run_one_training_batch(_UniformTeacher(), 1.0)

    # The mock teacher is uniform, so KL(uniform || p_student) >= 0 and the
    # distilled loss must be strictly larger than the base loss.
    assert loss_with_kd > loss_without_teacher


def test_kd_loss_direction_and_hand_computed_value():
    teacher_probs = torch.tensor([[0.5, 0.5]])
    student_probs = torch.tensor([[0.25, 0.75]])
    student_logits = torch.log(student_probs).requires_grad_(True)

    kd = compute_kd_loss(student_logits, teacher_probs)

    forward_kl = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    reverse_kl = 0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5)
    assert kd.item() == pytest.approx(forward_kl, rel=1e-6)
    assert kd.item() != pytest.approx(reverse_kl, rel=1e-3)

    kd.backward()
    assert student_logits.grad is not None
    assert student_logits.grad.abs().sum() > 0


def test_kd_loss_is_zero_for_identical_distributions():
    teacher_probs = torch.softmax(torch.randn(4, 81), dim=1)
    kd = compute_kd_loss(torch.log(teacher_probs), teacher_probs)
    assert kd.item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(
    not TEACHER_CHECKPOINT.is_file(),
    reason="EXP-030 teacher checkpoint not available locally",
)
def test_real_teacher_checkpoint_forward_is_valid_on_student_input():
    try:
        checkpoint = torch.load(
            TEACHER_CHECKPOINT, map_location="cpu", weights_only=False
        )
    except Exception as exc:  # checkpoint may still be transferring
        pytest.skip(f"teacher checkpoint unreadable (transfer in progress?): {exc}")

    assert "model_state_dict" in checkpoint
    assert "ema_state_dict" in checkpoint

    teacher = build_teacher(weights_path=None)
    teacher.load_state_dict(checkpoint["ema_state_dict"])
    teacher.eval()
    teacher.requires_grad_(False)

    student_input = torch.randn(1, 3, 256, 256)
    teacher_input = student_to_teacher_input(student_input)
    assert teacher_input.shape == (1, 3, TEACHER_INPUT_SIZE, TEACHER_INPUT_SIZE)

    with torch.no_grad():
        outputs = teacher(teacher_input)
    probs = outputs["main_prob"]
    assert probs.shape == (1, 81)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5)
