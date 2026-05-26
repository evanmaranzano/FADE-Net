import ast
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from experiment import build_model_for_checkpoint_load, load_model_state_package


def _source(path):
    return (ROOT_DIR / path).read_text(encoding="utf-8")


def test_web_demo_live_video_has_no_blocking_toggle_loop():
    tree = ast.parse(_source("src/web_demo.py"))
    while_tests = [
        ast.unparse(node.test)
        for node in ast.walk(tree)
        if isinstance(node, ast.While)
    ]

    assert "run_video" not in while_tests


def test_gui_demo_has_no_hidden_keyboard_calibration_offset():
    source = _source("src/gui_demo.py")

    assert "secret_calibration" not in source
    assert "set_calibration_offset" not in source


def test_gui_demo_image_mode_stops_after_single_frame():
    source = _source("src/gui_demo.py")
    assert "self._run_flag = False  # single-image mode finishes after one inference pass" in source


def test_demo_checkpoint_loader_uses_weights_only_when_available(monkeypatch):
    calls = []

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return {"model_state_dict": {"weight": 1}, "metadata": {"experiment_id": "safe"}}

    monkeypatch.setattr("experiment.torch.load", fake_load)

    state_dict, checkpoint = load_model_state_package("demo.pth", "cpu")

    assert state_dict == {"weight": 1}
    assert checkpoint["metadata"]["experiment_id"] == "safe"
    assert calls[0][1]["weights_only"] is True


def test_demo_checkpoint_loader_rejects_unsafe_legacy_torch_load(monkeypatch):
    calls = []

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        if kwargs.get("weights_only") is True:
            raise TypeError("weights_only is not supported")
        return {"model_state_dict": {"unsafe": 1}}

    monkeypatch.setattr("experiment.torch.load", fake_load)

    try:
        load_model_state_package("untrusted.pth", "cpu")
    except RuntimeError as exc:
        assert "weights_only=True" in str(exc)
    else:
        raise AssertionError("unsafe legacy torch.load fallback was not rejected")

    assert len(calls) == 1


def test_demos_use_inference_metadata_compatibility_contract():
    for path in ("src/web_demo.py", "src/gui_demo.py"):
        source = _source(path)

        assert "inference_checkpoint_metadata_mismatches" in source


def test_demos_use_configured_normalization_for_inference():
    web_source = _source("src/web_demo.py")
    gui_source = _source("src/gui_demo.py")

    assert "transforms.Normalize(mean=cfg.image_mean, std=cfg.image_std)" in web_source
    assert "transforms.Normalize(mean=self.cfg.image_mean, std=self.cfg.image_std)" in gui_source
    assert "transforms.Normalize(mean=[0.485" not in web_source
    assert "transforms.Normalize(mean=[0.485" not in gui_source


def test_demos_do_not_bypass_metadata_scan_for_exact_checkpoint_path():
    web_source = _source("src/web_demo.py")
    gui_source = _source("src/gui_demo.py")

    assert "model_path = exact_path" not in web_source
    assert "return exact_path" not in gui_source


def test_checkpoint_model_builder_disables_pretrained_only_during_instantiation(monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, cfg):
            calls.append(cfg.backbone_pretrained)
            if cfg.backbone_pretrained:
                raise RuntimeError("pretrained download attempted")

    monkeypatch.setattr("model.LightweightAgeEstimator", FakeModel)

    class Cfg:
        backbone_pretrained = True

    cfg = Cfg()
    model = build_model_for_checkpoint_load(cfg)

    assert isinstance(model, FakeModel)
    assert calls == [False]
    assert cfg.backbone_pretrained is True


def test_eval_and_demo_checkpoint_paths_build_without_pretrained_download():
    for path in ("scripts/advanced_eval.py", "src/web_demo.py", "src/gui_demo.py"):
        source = _source(path)

        assert "build_model_for_checkpoint_load" in source


def test_thesis_draft_handles_missing_afad_directory():
    source = _source("scripts/generate_thesis_draft.py")

    assert "if not dataset_dir.exists():" in source
    assert "AFAD dataset directory not found" in source


def test_preprocess_uses_cli_or_environment_dataset_path():
    source = _source("scripts/preprocess.py")

    assert "argparse" in source
    assert "FADE_NET_AFAD_DIR" in source
    assert "F:\\QQFiles" not in source
