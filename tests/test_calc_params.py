import sys
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import calc_params
from src.config import Config


def _args(ablation_id):
    return SimpleNamespace(
        ablation_id=ablation_id,
        backbone_source=None,
        backbone_name=None,
        no_pretrained=False,
    )


def test_calc_params_applies_a9_ablation_profile_overrides():
    cfg = Config()

    calc_params.apply_common_overrides(cfg, _args("A9"))

    assert cfg.use_multi_scale is True
    assert cfg.use_spp is True
    assert cfg.use_texture_branch is True
    assert cfg.use_freq_attention is True
    assert cfg.use_moe is True
    assert cfg.use_adaptive_triplet is True
    assert cfg.use_asymmetric_ordinal is True


def test_calc_params_applies_a0_ablation_profile_overrides():
    cfg = Config()

    calc_params.apply_common_overrides(cfg, _args("A0"))

    assert cfg.use_multi_scale is False
    assert cfg.use_spp is False
    assert cfg.use_texture_branch is False
    assert cfg.use_freq_attention is False
    assert cfg.use_moe is False
    assert cfg.use_adaptive_triplet is False
    assert cfg.use_asymmetric_ordinal is False


def test_calc_params_cli_passes_ablation_id_to_model(monkeypatch, capsys):
    captured = {}

    class FakeModel:
        def __init__(self, cfg):
            captured["cfg"] = cfg

        def eval(self):
            return self

        def parameters(self):
            return []

        def __call__(self, _inputs):
            return None

    monkeypatch.setattr(calc_params, "LightweightAgeEstimator", FakeModel)
    monkeypatch.setattr(
        calc_params,
        "profile",
        lambda _model, inputs, verbose=False: (0, 0),
    )
    monkeypatch.setattr(
        calc_params,
        "clever_format",
        lambda values, _fmt: tuple(str(value) for value in values),
    )
    monkeypatch.setattr(sys, "argv", ["calc_params.py", "--no_pretrained", "--ablation_id", "A0"])

    calc_params.main()

    assert captured["cfg"].backbone_pretrained is False
    assert captured["cfg"].use_multi_scale is False
    assert captured["cfg"].use_spp is False
    assert captured["cfg"].use_texture_branch is False
    assert "Total Parameters: 0" in capsys.readouterr().out
