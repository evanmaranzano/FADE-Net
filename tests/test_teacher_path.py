import sys
from pathlib import Path

import pytest
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from teacher_vit import (  # noqa: E402
    FARL_AUX_KEY_PREFIXES,
    FaRLVisualEncoder,
    build_teacher,
    load_farl_visual_weights,
)
from train_farl_teacher import (  # noqa: E402
    TeacherLoss,
    build_arg_parser,
    get_teacher_transforms,
    validate_args,
)

FARL_WEIGHTS = ROOT_DIR / "pretrained" / "FaRL-Base-Patch16-LAIONFace20M-ep16.pth"


def test_teacher_forward_shapes_and_distribution_contract():
    model = build_teacher(weights_path=None).eval()
    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 224, 224))

    assert outputs["main_prob"].shape == (2, 81)
    assert outputs["main_logits"].shape == (2, 81)
    assert outputs["age"].shape == (2,)
    assert torch.allclose(
        outputs["main_prob"].sum(dim=1), torch.ones(2), atol=1e-5
    )
    assert (outputs["age"] >= 0).all() and (outputs["age"] <= 80).all()
    assert torch.isfinite(outputs["age"]).all()
    assert torch.equal(outputs["age"], outputs["base_age"])


def test_visual_key_roundtrip_via_visual_prefix():
    source = FaRLVisualEncoder()
    with torch.no_grad():
        for param in source.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    prefixed = {f"visual.{k}": v for k, v in source.state_dict().items()}
    # FaRL aux keys plus the text tower must be tolerated.
    prefixed["visual.lm_head.weight"] = torch.randn(8192, 768)
    prefixed["visual.mask_token"] = torch.randn(768)
    prefixed["text_side.weight"] = torch.randn(4, 4)

    target = FaRLVisualEncoder()
    stats = load_farl_visual_weights(target, prefixed, source="synthetic")

    assert stats["consumed"] == len(source.state_dict())
    assert stats["missing"] == 0
    assert stats["aux_ignored"] == 2
    for name, value in target.state_dict().items():
        assert torch.equal(value, source.state_dict()[name])


def test_load_farl_visual_weights_rejects_missing_encoder_keys():
    encoder = FaRLVisualEncoder()
    broken = {f"visual.{k}": v for k, v in encoder.state_dict().items()}
    del broken["visual.ln_post.weight"]
    with pytest.raises(KeyError, match="ln_post"):
        load_farl_visual_weights(FaRLVisualEncoder(), broken, source="broken")


@pytest.mark.skipif(not FARL_WEIGHTS.is_file(),
                    reason="FaRL weights download not finished")
def test_real_farl_weights_load_and_forward_is_finite():
    checkpoint = torch.load(FARL_WEIGHTS, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    visual_keys = {k for k in state_dict if k.startswith("visual.")}

    model = build_teacher(str(FARL_WEIGHTS)).eval()
    consumed = {
        f"visual.{k}" for k in model.visual.state_dict()
    }
    missing = consumed - visual_keys
    aux = {
        k for k in visual_keys
        if k[len("visual."):].startswith(FARL_AUX_KEY_PREFIXES)
    }
    unexpected = visual_keys - consumed - aux

    assert not missing, f"missing encoder keys: {sorted(missing)}"
    assert not unexpected, f"unconsumed non-aux keys: {sorted(unexpected)}"
    assert consumed <= visual_keys
    assert len(consumed) == 152  # 12 blocks * 12 + 8 top-level visual keys

    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 224, 224))
    assert torch.isfinite(outputs["main_logits"]).all()
    assert torch.isfinite(outputs["age"]).all()


def test_teacher_loss_matches_student_main_path_form():
    criterion = TeacherLoss(min_age=0, max_age=80, label_sigma=2.0)
    logits = torch.randn(4, 81, requires_grad=True)
    ages = torch.tensor([20.0, 35.0, 50.0, 65.0])
    outputs = {
        "main_logits": logits,
        "base_age": torch.softmax(logits, dim=1).mul(
            torch.arange(81, dtype=torch.float32)
        ).sum(dim=1),
    }
    losses = criterion(outputs, ages)

    assert set(losses) == {"total", "main_kl", "main_reg"}
    assert torch.isclose(
        losses["total"], losses["main_kl"] + losses["main_reg"]
    )
    losses["total"].backward()


def test_argparser_defaults_match_exp030_spec(tmp_path):
    parser = build_arg_parser()
    args = parser.parse_args(["--farl_weights", str(tmp_path / "w.pth")])
    (tmp_path / "w.pth").touch()
    validate_args(args, parser)

    assert args.input_size == 224
    assert args.batch_size == 64
    assert args.epochs == 55
    assert args.warmup_epochs == 5
    assert args.early_stopping_patience == 20
    assert args.ema_decay == 0.999
    assert args.seed == 42
    assert args.num_workers == 4
    assert args.backbone_lr == 1e-5
    assert args.head_lr == 3e-4
    assert args.weight_decay == 5e-4
    assert args.gradient_clip == 5.0
    assert args.label_sigma == 2.0
    assert args.train_crop_scale_min == 0.7
    assert args.random_erasing_p == 0.1

    train_tf = get_teacher_transforms(224, is_train=True)
    val_tf = get_teacher_transforms(224, is_train=False)
    assert train_tf.transforms[0].scale == (0.7, 1.0)
    assert train_tf.transforms[-1].p == 0.1
    assert list(train_tf.transforms[4].mean) == pytest.approx(
        [0.48145466, 0.4578275, 0.40821073]
    )
    assert list(val_tf.transforms[3].std) == pytest.approx(
        [0.26862954, 0.26130258, 0.27577711]
    )
