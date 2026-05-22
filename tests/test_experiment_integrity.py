import os
import sys
import tempfile
import unittest
import json
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from config import Config
from dataset import (
    CachedSplitMetadataMismatchError,
    LegacySplitMetadataError,
    dataset_fingerprint,
    get_stratified_split,
    split_filename_with_tag,
)
from evaluation import TTA_MODES, evaluate_mae, normalize_tta_mode, predict_probs
from experiment import artifact_path, build_training_metadata, checkpoint_metadata_mismatches
from swa_average import average_checkpoints, discover_checkpoint_seeds
from train import _guard_fresh_artifact_overwrite, amp_step_was_skipped, make_grad_scaler, parse_selected_test_mae


class ConstantLogitModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, images):
        return torch.zeros(images.size(0), self.num_classes, device=images.device)


class RecordingLogitModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.calls = []

    def forward(self, images):
        self.calls.append(images.detach().clone())
        return torch.zeros(images.size(0), self.num_classes, device=images.device)


class ExperimentIntegrityTests(unittest.TestCase):
    def make_config(self):
        cfg = Config()
        cfg.device = torch.device("cpu")
        cfg.backbone_pretrained = False
        cfg.num_workers = 0
        cfg.img_size = 4
        cfg.num_classes = 3
        cfg.min_age = 0
        cfg.max_age = 2
        return cfg

    def test_metadata_declares_tta_protocol_and_rejects_split_drift(self):
        cfg = self.make_config()
        meta_a = build_training_metadata(cfg, seed=42, split_metadata={
            "split_file": "split_a.json",
            "fingerprint": "aaa",
        })
        meta_b = build_training_metadata(cfg, seed=42, split_metadata={
            "split_file": "split_a.json",
            "fingerprint": "bbb",
        })

        self.assertEqual(["raw", "flip", "multi"], meta_a["reported_tta_modes"])
        self.assertEqual("multi", meta_a["selection_metric"]["tta"])

        mismatches = checkpoint_metadata_mismatches({"metadata": meta_a}, meta_b)
        mismatch_keys = {key for key, _, _ in mismatches}
        self.assertIn("split_fingerprint", mismatch_keys)

    def test_cached_split_rejects_dataset_fingerprint_drift(self):
        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self, paths, ages):
                self.image_paths = paths
                self.ages = ages

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, index):
                return self.image_paths[index], self.ages[index]

        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = os.path.join(tmpdir, "split.json")
            first = TinyDataset(["/data/18/111/a.jpg", "/data/18/112/b.jpg", "/data/19/111/c.jpg"], [18, 18, 19])
            second = TinyDataset(["/data/18/111/a.jpg", "/data/18/112/changed.jpg", "/data/19/111/c.jpg"], [18, 18, 19])

            get_stratified_split(
                first,
                first.ages,
                split_ratios=(0.34, 0.33, 0.33),
                save_path=split_path,
                dataset_hash=dataset_fingerprint(first),
            )

            with self.assertRaises(CachedSplitMetadataMismatchError):
                get_stratified_split(
                    second,
                    second.ages,
                    split_ratios=(0.34, 0.33, 0.33),
                    save_path=split_path,
                    dataset_hash=dataset_fingerprint(second),
                )
            with open(split_path, encoding="utf-8") as f:
                unchanged_split = json.load(f)

            self.assertEqual(
                dataset_fingerprint(first),
                unchanged_split["_metadata"]["dataset_fingerprint"],
            )

    def test_legacy_split_is_rejected_by_default(self):
        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.image_paths = ["/data/18/111/a.jpg", "/data/19/111/b.jpg", "/data/20/111/c.jpg"]
                self.ages = [18, 19, 20]

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, index):
                return self.image_paths[index], self.ages[index]

        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = os.path.join(tmpdir, "split.json")
            legacy_split = {"train": [0], "val": [1], "test": [2]}
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(legacy_split, f)

            dataset = TinyDataset()
            with self.assertRaises(LegacySplitMetadataError):
                get_stratified_split(
                    dataset,
                    dataset.ages,
                    split_ratios=(0.34, 0.33, 0.33),
                    save_path=split_path,
                    dataset_hash=dataset_fingerprint(dataset),
                )
            with open(split_path, encoding="utf-8") as f:
                unchanged_split = json.load(f)

            self.assertNotIn("_metadata", unchanged_split)

    def test_legacy_split_can_be_explicitly_upgraded_without_changing_indices(self):
        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.image_paths = ["/data/18/111/a.jpg", "/data/19/111/b.jpg", "/data/20/111/c.jpg"]
                self.ages = [18, 19, 20]

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, index):
                return self.image_paths[index], self.ages[index]

        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = os.path.join(tmpdir, "split.json")
            legacy_split = {"train": [0], "val": [1], "test": [2]}
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(legacy_split, f)

            dataset = TinyDataset()
            get_stratified_split(
                dataset,
                dataset.ages,
                split_ratios=(0.34, 0.33, 0.33),
                save_path=split_path,
                dataset_hash=dataset_fingerprint(dataset),
                allow_legacy_split_upgrade=True,
            )
            with open(split_path, encoding="utf-8") as f:
                upgraded_split = json.load(f)

            self.assertEqual(legacy_split["train"], upgraded_split["train"])
            self.assertEqual(legacy_split["val"], upgraded_split["val"])
            self.assertEqual(legacy_split["test"], upgraded_split["test"])
            self.assertEqual(dataset_fingerprint(dataset), upgraded_split["_metadata"]["dataset_fingerprint"])
            self.assertTrue(upgraded_split["_metadata"]["legacy_upgraded"])

    def test_artifact_path_uses_experiment_identity(self):
        cfg = self.make_config()
        path = artifact_path(str(ROOT_DIR), "final_result", cfg, seed=42, extension=".txt")

        self.assertTrue(path.endswith(".txt"))
        self.assertIn("final_result_", os.path.basename(path))
        self.assertIn("seed42", os.path.basename(path))

    def test_experiment_id_separates_backbone_source_and_pretraining(self):
        cfg_scratch = self.make_config()
        cfg_scratch.backbone_source = "timm"
        cfg_scratch.backbone_name = "mobilenetv4_conv_small"
        cfg_scratch.backbone_pretrained = False

        cfg_pretrained = self.make_config()
        cfg_pretrained.backbone_source = "timm"
        cfg_pretrained.backbone_name = "mobilenetv4_conv_small"
        cfg_pretrained.backbone_pretrained = True

        scratch_meta = build_training_metadata(cfg_scratch, seed=42)
        pretrained_meta = build_training_metadata(cfg_pretrained, seed=42)

        self.assertNotEqual(scratch_meta["experiment_id"], pretrained_meta["experiment_id"])
        self.assertIn("timm", scratch_meta["experiment_id"])
        self.assertIn("scratch", scratch_meta["experiment_id"])
        self.assertIn("pretrained", pretrained_meta["experiment_id"])

        scratch_path = artifact_path(str(ROOT_DIR), "final_result", cfg_scratch, seed=42, extension=".txt")
        pretrained_path = artifact_path(str(ROOT_DIR), "final_result", cfg_pretrained, seed=42, extension=".txt")
        self.assertNotEqual(scratch_path, pretrained_path)

    def test_experiment_tag_separates_smoke_artifacts(self):
        cfg_full = self.make_config()
        cfg_smoke = self.make_config()
        cfg_smoke.experiment_tag = "smoke"

        full_meta = build_training_metadata(cfg_full, seed=42)
        smoke_meta = build_training_metadata(cfg_smoke, seed=42)

        self.assertNotEqual(full_meta["experiment_id"], smoke_meta["experiment_id"])
        self.assertEqual("smoke", smoke_meta["experiment_tag"])
        mismatches = checkpoint_metadata_mismatches({"metadata": full_meta}, smoke_meta)
        self.assertIn("experiment_tag", {key for key, _, _ in mismatches})

    def test_split_file_tag_separates_formal_split_artifacts(self):
        cfg_default = self.make_config()
        cfg_tagged = self.make_config()
        cfg_tagged.split_file_tag = "formal_v1"

        default_meta = build_training_metadata(cfg_default, seed=42)
        tagged_meta = build_training_metadata(cfg_tagged, seed=42)

        self.assertNotEqual(default_meta["experiment_id"], tagged_meta["experiment_id"])
        self.assertIsNone(default_meta["split_file_tag"])
        self.assertEqual("formal_v1", tagged_meta["split_file_tag"])
        self.assertIn("splitfile-formal_v1", tagged_meta["experiment_id"])

        default_path = artifact_path(str(ROOT_DIR), "final_result", cfg_default, seed=42, extension=".txt")
        tagged_path = artifact_path(str(ROOT_DIR), "final_result", cfg_tagged, seed=42, extension=".txt")
        self.assertNotEqual(default_path, tagged_path)

        mismatches = checkpoint_metadata_mismatches({"metadata": default_meta}, tagged_meta)
        self.assertIn("split_file_tag", {key for key, _, _ in mismatches})

    def test_split_filename_with_tag_preserves_default_and_sanitizes_tag(self):
        self.assertEqual(
            "dataset_split_AFAD_72_8_20.json",
            split_filename_with_tag("dataset_split_AFAD_72_8_20.json", None),
        )
        self.assertEqual(
            "dataset_split_AFAD_72_8_20_formal-v1.json",
            split_filename_with_tag("dataset_split_AFAD_72_8_20.json", "formal v1"),
        )

    def test_evaluate_mae_reports_all_modes_and_skips_empty_batches(self):
        cfg = self.make_config()
        model = ConstantLogitModel(cfg.num_classes)
        empty_batch = (torch.tensor([]), torch.tensor([]), torch.tensor([]))
        valid_batch = (
            torch.zeros(2, 3, cfg.img_size, cfg.img_size),
            torch.zeros(2, cfg.num_classes),
            torch.ones(2),
        )

        metrics = evaluate_mae(model, [empty_batch, valid_batch], cfg, cfg.device, modes=TTA_MODES)

        self.assertEqual(set(TTA_MODES), set(metrics))
        for mode in TTA_MODES:
            self.assertAlmostEqual(0.0, metrics[mode], places=6)
        self.assertEqual("raw", normalize_tta_mode("none"))

    def test_tta_modes_use_expected_number_of_forward_passes(self):
        cfg = self.make_config()
        images = torch.arange(3 * cfg.img_size * cfg.img_size, dtype=torch.float32).reshape(
            1, 3, cfg.img_size, cfg.img_size
        )

        for mode, expected_calls in {"raw": 1, "flip": 2, "multi": 6}.items():
            model = RecordingLogitModel(cfg.num_classes)
            predict_probs(model, images, mode=mode, base_size=cfg.img_size)
            self.assertEqual(expected_calls, len(model.calls), mode)

        model = RecordingLogitModel(cfg.num_classes)
        predict_probs(model, images, mode="flip", base_size=cfg.img_size)
        self.assertTrue(torch.equal(torch.flip(model.calls[0], dims=[3]), model.calls[1]))

    def test_evaluate_mae_rejects_all_empty_batches(self):
        cfg = self.make_config()
        model = ConstantLogitModel(cfg.num_classes)
        empty_batch = (torch.tensor([]), torch.tensor([]), torch.tensor([]))

        with self.assertRaises(RuntimeError):
            evaluate_mae(model, [empty_batch], cfg, cfg.device)

    def test_evaluate_mae_can_limit_batches_for_smoke_runs(self):
        cfg = self.make_config()
        model = ConstantLogitModel(cfg.num_classes)
        first_batch = (
            torch.zeros(1, 3, cfg.img_size, cfg.img_size),
            torch.zeros(1, cfg.num_classes),
            torch.ones(1),
        )
        second_batch = (
            torch.zeros(1, 3, cfg.img_size, cfg.img_size),
            torch.zeros(1, cfg.num_classes),
            torch.zeros(1),
        )

        metrics = evaluate_mae(model, [first_batch, second_batch], cfg, cfg.device, modes=("raw",), max_batches=1)

        self.assertEqual({"raw": 0.0}, metrics)

    def test_swa_rejects_metadata_mismatch(self):
        cfg = self.make_config()
        meta_a = build_training_metadata(cfg, seed=42, split_metadata={"fingerprint": "aaa"})
        meta_b = build_training_metadata(cfg, seed=42, split_metadata={"fingerprint": "bbb"})

        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "a.pth")
            path_b = os.path.join(tmpdir, "b.pth")
            torch.save({"model_state_dict": {"w": torch.ones(1)}, "metadata": meta_a}, path_a)
            torch.save({"model_state_dict": {"w": torch.ones(1)}, "metadata": meta_b}, path_b)

            with self.assertRaises(RuntimeError):
                average_checkpoints([path_a, path_b], device="cpu")

    def test_swa_returns_checkpoint_metadata(self):
        cfg = self.make_config()
        meta = build_training_metadata(cfg, seed=42, split_metadata={"fingerprint": "aaa"})

        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "a.pth")
            path_b = os.path.join(tmpdir, "b.pth")
            torch.save({"model_state_dict": {"w": torch.ones(1)}, "metadata": meta}, path_a)
            torch.save({"model_state_dict": {"w": torch.full((1,), 3.0)}, "metadata": meta}, path_b)

            avg_state, returned_meta = average_checkpoints([path_a, path_b], device="cpu")

        self.assertEqual(meta, returned_meta)
        self.assertTrue(torch.equal(torch.full((1,), 2.0), avg_state["w"]))

    def test_swa_auto_detection_includes_formal_seed_2026(self):
        cfg = self.make_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = artifact_path(tmpdir, "checkpoint_epoch_111", cfg, 2026, ".pth")
            torch.save({"model_state_dict": {"w": torch.ones(1)}, "metadata": {}}, checkpoint_path)

            seeds = discover_checkpoint_seeds(tmpdir, cfg, epoch=111)

        self.assertIn(2026, seeds)

    def test_checkpoint_missing_dataset_fingerprint_is_a_mismatch(self):
        cfg = self.make_config()
        expected = build_training_metadata(cfg, seed=42, split_metadata={
            "fingerprint": "split-hash",
            "dataset_fingerprint": "dataset-hash",
        })
        checkpoint_meta = dict(expected)
        checkpoint_meta.pop("dataset_fingerprint")

        mismatches = checkpoint_metadata_mismatches({"metadata": checkpoint_meta}, expected)

        self.assertIn("dataset_fingerprint", {key for key, _, _ in mismatches})

    def test_fresh_artifact_guard_refuses_existing_non_checkpoint_artifacts(self):
        cfg = self.make_config()
        expected = build_training_metadata(cfg, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = os.path.join(tmpdir, "final_result.txt")
            with open(result_path, "w", encoding="utf-8") as f:
                f.write("Selected_Test_MAE: 3.21\n")

            with self.assertRaises(RuntimeError):
                _guard_fresh_artifact_overwrite([result_path], expected, cfg.device)

    def test_selected_test_mae_parser_prefers_new_key(self):
        self.assertEqual(3.21, parse_selected_test_mae("Selected_Test_MAE: 3.21\nTest MAE: 9.99\n"))
        self.assertEqual(3.22, parse_selected_test_mae("Final Test MAE (multi): 3.22"))
        self.assertEqual(3.23, parse_selected_test_mae("Test MAE: 3.23"))

    def test_make_grad_scaler_supports_current_torch_amp_api(self):
        scaler = make_grad_scaler("cuda")

        self.assertTrue(hasattr(scaler, "scale"))
        self.assertTrue(hasattr(scaler, "step"))

    def test_amp_step_skip_detection_uses_scale_drop(self):
        self.assertTrue(amp_step_was_skipped(1024.0, 512.0))
        self.assertFalse(amp_step_was_skipped(1024.0, 1024.0))
        self.assertFalse(amp_step_was_skipped(1024.0, 2048.0))


if __name__ == "__main__":
    unittest.main()
