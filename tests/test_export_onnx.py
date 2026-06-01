import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from config import Config
from experiment import build_training_metadata, populate_runtime_model_metadata
from export_onnx import apply_common_overrides, export_onnx


class ExportOnnxTests(unittest.TestCase):
    def test_apply_common_overrides_supports_ablation(self):
        class Args:
            ablation_id = "A6"
            backbone_source = None
            backbone_name = None
            no_pretrained = False
            split_file_tag = "formal_v1"

        cfg = Config()
        apply_common_overrides(cfg, Args())

        self.assertTrue(cfg.use_moe)
        self.assertEqual("formal_v1", cfg.split_file_tag)

    def test_export_creates_output_parent_dir(self):
        cfg = Config()
        cfg.device = torch.device("cpu")
        cfg.backbone_pretrained = False
        cfg.num_classes = 1
        cfg.img_size = 1
        populate_runtime_model_metadata(cfg)
        metadata = build_training_metadata(cfg, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"
            output_path = Path(tmpdir) / "missing" / "model.onnx"
            torch.save({"model_state_dict": {"weight": torch.ones(1)}, "metadata": metadata}, checkpoint_path)

            fake_model = torch.nn.Linear(1, 1)
            fake_model.load_state_dict = mock.Mock()
            fake_model.eval = mock.Mock()

            def fake_export(_model, _dummy, path, **_kwargs):
                Path(path).write_bytes(b"onnx")

            with mock.patch("export_onnx.build_model_for_checkpoint_load", return_value=fake_model), \
                 mock.patch("export_onnx.torch.randn", return_value=torch.zeros(1, 1)), \
                 mock.patch("export_onnx.torch.onnx.export", side_effect=fake_export):
                export_onnx(str(checkpoint_path), str(output_path), cfg)

            self.assertTrue(output_path.exists())

    def test_export_accepts_checkpoint_with_runtime_metadata(self):
        cfg = Config()
        cfg.device = torch.device("cpu")
        cfg.backbone_pretrained = False
        populate_runtime_model_metadata(cfg)
        metadata = build_training_metadata(cfg, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"
            output_path = Path(tmpdir) / "model.onnx"
            torch.save({"model_state_dict": {"weight": torch.ones(1)}, "metadata": metadata}, checkpoint_path)

            fake_model = torch.nn.Linear(1, 1)
            fake_model.load_state_dict = mock.Mock()
            fake_model.eval = mock.Mock()

            with mock.patch("export_onnx.build_model_for_checkpoint_load", return_value=fake_model), \
                 mock.patch("export_onnx.torch.randn", return_value=torch.zeros(1, 1)), \
                 mock.patch("export_onnx.torch.onnx.export", side_effect=lambda _m, _d, path, **_k: Path(path).write_bytes(b"onnx")):
                export_onnx(str(checkpoint_path), str(output_path), cfg)

            self.assertTrue(output_path.exists())
            fake_model.load_state_dict.assert_called_once()

    def test_export_rejects_metadata_mismatch_before_state_dict_load(self):
        cfg = Config()
        cfg.device = torch.device("cpu")
        cfg.backbone_pretrained = False
        metadata = build_training_metadata(cfg, seed=42)
        metadata = dict(metadata)
        metadata["test_tta"] = "raw"

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"
            torch.save({"model_state_dict": {}, "metadata": metadata}, checkpoint_path)

            with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
                export_onnx(str(checkpoint_path), str(Path(tmpdir) / "model.onnx"), cfg)


if __name__ == "__main__":
    unittest.main()
