import sys
import unittest
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import Config
from experiment import build_training_metadata, checkpoint_metadata_mismatches
from model import LightweightAgeEstimator


class ResearchRefactorTests(unittest.TestCase):
    def make_fast_config(self):
        cfg = Config()
        cfg.backbone_pretrained = False
        cfg.device = torch.device("cpu")
        cfg.use_hybrid_attention = True
        cfg.use_multi_scale = True
        cfg.use_spp = True
        cfg.use_mv_loss = True
        cfg.num_workers = 0
        return cfg

    def test_default_backbone_forward_contract(self):
        cfg = self.make_fast_config()
        model = LightweightAgeEstimator(cfg)
        model.eval()

        with torch.no_grad():
            logits = model(torch.zeros(1, 3, cfg.img_size, cfg.img_size))

        self.assertEqual((1, cfg.num_classes), tuple(logits.shape))
        self.assertEqual(40, model.feature_spec.shallow_channels)
        self.assertEqual(112, model.feature_spec.mid_channels)
        self.assertEqual(960, model.feature_spec.out_channels)

    def test_training_metadata_distinguishes_backbones(self):
        cfg_a = self.make_fast_config()
        cfg_b = self.make_fast_config()
        cfg_b.backbone_source = "timm"
        cfg_b.backbone_name = "repvit_m0_9"

        meta_a = build_training_metadata(cfg_a, seed=42)
        meta_b = build_training_metadata(cfg_b, seed=42)

        self.assertNotEqual(meta_a["experiment_id"], meta_b["experiment_id"])
        self.assertNotEqual(meta_a["backbone"], meta_b["backbone"])

    def test_checkpoint_metadata_rejects_mismatched_backbone(self):
        cfg_current = self.make_fast_config()
        cfg_checkpoint = self.make_fast_config()
        cfg_checkpoint.backbone_source = "timm"
        cfg_checkpoint.backbone_name = "mobilenetv4_conv_small"

        expected = build_training_metadata(cfg_current, seed=42)
        checkpoint = {"metadata": build_training_metadata(cfg_checkpoint, seed=42)}
        mismatches = checkpoint_metadata_mismatches(checkpoint, expected)

        mismatch_keys = {key for key, _, _ in mismatches}
        self.assertIn("experiment_id", mismatch_keys)
        self.assertIn("backbone", mismatch_keys)

    def test_timm_stage_index_fallback_uses_valid_indices(self):
        try:
            from backbones import TimmFeatureBackbone
        except ImportError:
            self.skipTest("timm is not installed")

        cfg = self.make_fast_config()
        cfg.backbone_source = "timm"
        cfg.backbone_name = "mobilenetv4_conv_small"
        cfg.backbone_pretrained = False

        try:
            model = LightweightAgeEstimator(cfg)
        except ImportError:
            self.skipTest("timm is not installed")
        except RuntimeError as exc:
            self.skipTest(f"timm model is unavailable in this environment: {exc}")
        except ValueError as exc:
            self.skipTest(f"timm model is unavailable in this environment: {exc}")

        self.assertIsInstance(model.backbone, TimmFeatureBackbone)
        self.assertLess(max(model.msff_indices), model.backbone.feature_count)
        meta = build_training_metadata(cfg, seed=42)
        self.assertTrue(meta["ablations"]["use_hybrid_attention"])
        self.assertFalse(meta["ablations"]["effective_hybrid_attention"])
        self.assertNotIn("_HA_", cfg.project_name)


if __name__ == "__main__":
    unittest.main()
