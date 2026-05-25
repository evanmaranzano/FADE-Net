import ast
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from experiment import load_model_state_package


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
