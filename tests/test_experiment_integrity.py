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
from evaluation import TTA_MODES, evaluate_mae, normalize_tta_mode, predict_age_with_uncertainty, predict_probs
from experiment import (
    artifact_path,
    build_training_metadata,
    checkpoint_metadata_mismatches,
    compatible_best_model_paths,
    inference_checkpoint_metadata_mismatches,
    is_compatible_checkpoint,
)
from advanced_eval import evaluate_uncertainty
from swa_average import average_checkpoints, discover_checkpoint_seeds
from train import _guard_fresh_artifact_overwrite, amp_step_was_skipped, make_grad_scaler, parse_selected_test_mae
from train import hard_distillation_schedule_metadata


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


class SumLogitModel(torch.nn.Module):
    def forward(self, images):
        values = images.flatten(1).sum(dim=1)
        return torch.stack([values, values * 0.5, -values], dim=1)


class FlipSensitiveLogitModel(torch.nn.Module):
    def forward(self, images):
        left = images[:, :, :, 0].flatten(1).sum(dim=1)
        right = images[:, :, :, -1].flatten(1).sum(dim=1)
        return torch.stack([left, torch.zeros_like(left), right], dim=1)


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

    def test_new_loss_modules_separate_artifacts_and_metadata(self):
        cfg_full = self.make_config()
        cfg_triplet = self.make_config()
        cfg_triplet.use_adaptive_triplet = True
        cfg_asym = self.make_config()
        cfg_asym.use_asymmetric_ordinal = True

        full_meta = build_training_metadata(cfg_full, seed=42)
        triplet_meta = build_training_metadata(cfg_triplet, seed=42)
        asym_meta = build_training_metadata(cfg_asym, seed=42)

        self.assertNotEqual(full_meta["experiment_id"], triplet_meta["experiment_id"])
        self.assertNotEqual(full_meta["experiment_id"], asym_meta["experiment_id"])
        self.assertIn("TRIPLET", triplet_meta["experiment_id"])
        self.assertIn("ASYM", asym_meta["experiment_id"])
        self.assertTrue(triplet_meta["ablations"]["use_adaptive_triplet"])
        self.assertTrue(asym_meta["ablations"]["use_asymmetric_ordinal"])
        self.assertEqual(cfg_triplet.lambda_triplet, triplet_meta["loss"]["lambda_triplet"])
        self.assertEqual(cfg_asym.lambda_asym, asym_meta["loss"]["lambda_asym"])

        full_path = artifact_path(str(ROOT_DIR), "final_result", cfg_full, seed=42, extension=".txt")
        triplet_path = artifact_path(str(ROOT_DIR), "final_result", cfg_triplet, seed=42, extension=".txt")
        asym_path = artifact_path(str(ROOT_DIR), "final_result", cfg_asym, seed=42, extension=".txt")

        self.assertNotEqual(full_path, triplet_path)
        self.assertNotEqual(full_path, asym_path)

    def test_image_normalization_is_recorded_and_checked_in_metadata(self):
        cfg_default = self.make_config()
        cfg_custom = self.make_config()
        cfg_custom.image_mean = [0.1, 0.2, 0.3]
        cfg_custom.image_std = [0.4, 0.5, 0.6]

        default_meta = build_training_metadata(cfg_default, seed=42)
        custom_meta = build_training_metadata(cfg_custom, seed=42)

        self.assertEqual([0.485, 0.456, 0.406], default_meta["augmentation"]["image_mean"])
        self.assertEqual([0.229, 0.224, 0.225], default_meta["augmentation"]["image_std"])
        self.assertEqual([0.1, 0.2, 0.3], custom_meta["augmentation"]["image_mean"])
        self.assertEqual([0.4, 0.5, 0.6], custom_meta["augmentation"]["image_std"])

        mismatches = checkpoint_metadata_mismatches({"metadata": default_meta}, custom_meta)
        self.assertIn("augmentation", {key for key, _, _ in mismatches})

    def test_missing_normalization_metadata_is_default_only_compatible(self):
        cfg_default = self.make_config()
        cfg_custom = self.make_config()
        cfg_custom.image_mean = [0.1, 0.2, 0.3]
        cfg_custom.image_std = [0.4, 0.5, 0.6]

        legacy_meta = build_training_metadata(cfg_default, seed=42)
        legacy_meta["augmentation"].pop("image_mean")
        legacy_meta["augmentation"].pop("image_std")

        default_mismatches = checkpoint_metadata_mismatches(
            {"metadata": legacy_meta},
            build_training_metadata(cfg_default, seed=42),
        )
        custom_mismatches = checkpoint_metadata_mismatches(
            {"metadata": legacy_meta},
            build_training_metadata(cfg_custom, seed=42),
        )

        self.assertNotIn("augmentation", {key for key, _, _ in default_mismatches})
        self.assertIn("augmentation", {key for key, _, _ in custom_mismatches})

    def test_model_weight_schema_changes_are_versioned_in_metadata(self):
        cfg = self.make_config()
        expected = build_training_metadata(cfg, seed=42)
        legacy_meta = build_training_metadata(cfg, seed=42)
        legacy_meta["experiment_id"] = expected["experiment_id"]
        legacy_meta["backbone"]["head_version"] = "fade-head-v1"

        self.assertNotEqual("fade-head-v1", expected["backbone"]["head_version"])
        mismatches = checkpoint_metadata_mismatches({"metadata": legacy_meta}, expected)
        self.assertIn("backbone", {key for key, _, _ in mismatches})

        missing_version_meta = build_training_metadata(cfg, seed=42)
        missing_version_meta["experiment_id"] = expected["experiment_id"]
        missing_version_meta["backbone"].pop("head_version")
        missing_version_mismatches = checkpoint_metadata_mismatches({"metadata": missing_version_meta}, expected)
        self.assertIn("backbone", {key for key, _, _ in missing_version_mismatches})

    def test_missing_required_nested_metadata_keys_are_mismatches(self):
        cfg = self.make_config()
        expected = build_training_metadata(cfg, seed=42)
        missing_loss_meta = build_training_metadata(cfg, seed=42)
        missing_loss_meta["loss"].pop("lambda_moe_balance")
        missing_ablation_meta = build_training_metadata(cfg, seed=42)
        missing_ablation_meta["ablations"].pop("use_freq_attention")
        missing_backbone_meta = build_training_metadata(cfg, seed=42)
        missing_backbone_meta["backbone"].pop("effective_pretrained")

        loss_mismatches = checkpoint_metadata_mismatches({"metadata": missing_loss_meta}, expected)
        ablation_mismatches = checkpoint_metadata_mismatches({"metadata": missing_ablation_meta}, expected)
        backbone_mismatches = checkpoint_metadata_mismatches({"metadata": missing_backbone_meta}, expected)

        self.assertIn("loss", {key for key, _, _ in loss_mismatches})
        self.assertIn("ablations", {key for key, _, _ in ablation_mismatches})
        self.assertIn("backbone", {key for key, _, _ in backbone_mismatches})

    def test_inference_metadata_can_ignore_split_and_effective_pretraining_provenance(self):
        train_cfg = self.make_config()
        train_cfg.split_file_tag = "formal_v1"
        train_cfg.effective_pretrained = False
        train_cfg.split_metadata = {
            "split_file": "dataset_split_AFAD_72_8_20_formal_v1.json",
            "fingerprint": "split-hash",
            "dataset_fingerprint": "dataset-hash",
        }
        demo_cfg = self.make_config()
        demo_cfg.effective_pretrained = True

        mismatches = inference_checkpoint_metadata_mismatches(
            {"metadata": build_training_metadata(train_cfg, seed=42)},
            build_training_metadata(demo_cfg, seed=42),
        )

        self.assertEqual([], mismatches)

    def test_regularization_schedule_is_recorded_in_metadata(self):
        cfg = self.make_config()
        cfg.regularization_schedule = {
            "hard_distillation": {
                "start_epoch": 105,
                "disables_mixup": True,
                "disables_sigma_jitter": True,
                "uses_eval_transform": True,
            }
        }

        metadata = build_training_metadata(cfg, seed=42)

        self.assertEqual(cfg.regularization_schedule, metadata["regularization_schedule"])

    def test_default_regularization_schedule_matches_training_schedule(self):
        cfg = self.make_config()
        expected_schedule = {
            "hard_distillation": hard_distillation_schedule_metadata(cfg.epochs),
        }

        metadata = build_training_metadata(cfg, seed=42)

        self.assertEqual(expected_schedule, metadata["regularization_schedule"])

    def test_demo_expected_metadata_accepts_training_default_regularization_schedule(self):
        train_cfg = self.make_config()
        demo_cfg = self.make_config()
        train_cfg.regularization_schedule = {
            "hard_distillation": hard_distillation_schedule_metadata(train_cfg.epochs),
        }
        checkpoint = {"metadata": build_training_metadata(train_cfg, seed=42)}
        expected_metadata = build_training_metadata(demo_cfg, seed=42)

        mismatches = checkpoint_metadata_mismatches(checkpoint, expected_metadata)

        self.assertNotIn("regularization_schedule", {key for key, _, _ in mismatches})

    def test_regularization_schedule_drift_is_a_metadata_mismatch(self):
        cfg_a = self.make_config()
        cfg_b = self.make_config()
        cfg_a.regularization_schedule = {
            "hard_distillation": {
                "start_epoch": 105,
                "disables_mixup": True,
                "disables_sigma_jitter": True,
                "uses_eval_transform": True,
            }
        }
        cfg_b.regularization_schedule = {
            "hard_distillation": {
                "start_epoch": 85,
                "disables_mixup": True,
                "disables_sigma_jitter": True,
                "uses_eval_transform": True,
            }
        }

        mismatches = checkpoint_metadata_mismatches(
            {"metadata": build_training_metadata(cfg_a, seed=42)},
            build_training_metadata(cfg_b, seed=42),
        )

        self.assertIn("regularization_schedule", {key for key, _, _ in mismatches})

    def test_missing_regularization_schedule_is_compatible_with_default_only(self):
        cfg_default = self.make_config()
        default_meta = build_training_metadata(cfg_default, seed=42)
        default_meta.pop("regularization_schedule")

        default_mismatches = checkpoint_metadata_mismatches(
            {"metadata": default_meta},
            build_training_metadata(cfg_default, seed=42),
        )

        cfg_scheduled = self.make_config()
        cfg_scheduled.regularization_schedule = {
            "hard_distillation": {
                "start_epoch": 85,
                "disables_mixup": True,
                "disables_sigma_jitter": True,
                "uses_eval_transform": True,
            }
        }
        scheduled_mismatches = checkpoint_metadata_mismatches(
            {"metadata": default_meta},
            build_training_metadata(cfg_scheduled, seed=42),
        )

        self.assertNotIn("regularization_schedule", {key for key, _, _ in default_mismatches})
        self.assertIn("regularization_schedule", {key for key, _, _ in scheduled_mismatches})

    def test_missing_regularization_schedule_is_compatible_with_non_120_epoch_default(self):
        cfg_default = self.make_config()
        cfg_default.epochs = 10
        legacy_meta = build_training_metadata(cfg_default, seed=42)
        legacy_meta.pop("regularization_schedule")

        mismatches = checkpoint_metadata_mismatches(
            {"metadata": legacy_meta},
            build_training_metadata(cfg_default, seed=42),
        )

        self.assertNotIn("regularization_schedule", {key for key, _, _ in mismatches})

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

        for mode, expected_calls in {"raw": 1, "flip": 1, "multi": 1}.items():
            model = RecordingLogitModel(cfg.num_classes)
            predict_probs(model, images, mode=mode, base_size=cfg.img_size)
            self.assertEqual(expected_calls, len(model.calls), mode)

        model = RecordingLogitModel(cfg.num_classes)
        predict_probs(model, images, mode="flip", base_size=cfg.img_size)
        augmented = model.calls[0]
        self.assertEqual(2, augmented.size(0))
        self.assertTrue(torch.equal(torch.flip(augmented[0:1], dims=[3]), augmented[1:2]))
        self.assertTrue(torch.equal(images, augmented[0:1]))

    def test_tta_can_chunk_augmented_forward_passes(self):
        cfg = self.make_config()
        images = torch.arange(3 * cfg.img_size * cfg.img_size, dtype=torch.float32).reshape(
            1, 3, cfg.img_size, cfg.img_size
        )

        model = RecordingLogitModel(cfg.num_classes)
        predict_probs(model, images, mode="multi", base_size=cfg.img_size, max_augmented_batch_size=2)

        self.assertEqual(3, len(model.calls))
        self.assertEqual([2, 2, 2], [call.size(0) for call in model.calls])

    def test_tta_chunked_and_batched_outputs_match(self):
        cfg = self.make_config()
        images = torch.arange(2 * 3 * cfg.img_size * cfg.img_size, dtype=torch.float32).reshape(
            2, 3, cfg.img_size, cfg.img_size
        )

        batched = predict_probs(SumLogitModel(), images, mode="multi", base_size=cfg.img_size)
        chunked = predict_probs(
            SumLogitModel(),
            images,
            mode="multi",
            base_size=cfg.img_size,
            max_augmented_batch_size=1,
        )

        self.assertTrue(torch.equal(batched, chunked))

    def test_tta_chunked_and_batched_outputs_match_flip(self):
        cfg = self.make_config()
        images = torch.arange(2 * 3 * cfg.img_size * cfg.img_size, dtype=torch.float32).reshape(
            2, 3, cfg.img_size, cfg.img_size
        )

        batched = predict_probs(SumLogitModel(), images, mode="flip", base_size=cfg.img_size)
        chunked = predict_probs(
            SumLogitModel(),
            images,
            mode="flip",
            base_size=cfg.img_size,
            max_augmented_batch_size=1,
        )

        self.assertTrue(torch.equal(batched, chunked))

    def test_tta_uncertainty_reports_view_disagreement(self):
        cfg = self.make_config()
        images = torch.arange(3 * cfg.img_size * cfg.img_size, dtype=torch.float32).reshape(
            1, 3, cfg.img_size, cfg.img_size
        )

        raw_probs, raw_ages, raw_std = predict_age_with_uncertainty(
            FlipSensitiveLogitModel(), images, mode="raw", base_size=cfg.img_size
        )
        flip_probs, flip_ages, flip_std = predict_age_with_uncertainty(
            FlipSensitiveLogitModel(), images, mode="flip", base_size=cfg.img_size
        )

        self.assertEqual((1, cfg.num_classes), tuple(raw_probs.shape))
        self.assertEqual((1,), tuple(raw_ages.shape))
        self.assertTrue(torch.equal(torch.zeros_like(raw_std), raw_std))
        self.assertEqual((1, cfg.num_classes), tuple(flip_probs.shape))
        self.assertEqual((1,), tuple(flip_ages.shape))
        self.assertGreater(flip_std.item(), 0.0)

    def test_advanced_eval_uncertainty_summary(self):
        cfg = self.make_config()
        valid_batch = (
            torch.arange(3 * cfg.img_size * cfg.img_size, dtype=torch.float32).reshape(
                1, 3, cfg.img_size, cfg.img_size
            ),
            torch.zeros(1, cfg.num_classes),
            torch.ones(1),
        )

        summary = evaluate_uncertainty(FlipSensitiveLogitModel(), [valid_batch], cfg, cfg.device, mode="flip")

        self.assertEqual("flip", summary["mode"])
        self.assertGreaterEqual(summary["mae"], 0.0)
        self.assertGreater(summary["mean_age_std"], 0.0)

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

    def test_compatible_best_model_paths_filters_metadata_mismatch(self):
        cfg = self.make_config()
        matching_meta = build_training_metadata(cfg, seed=42)
        mismatched_cfg = self.make_config()
        mismatched_cfg.use_spp = not cfg.use_spp
        mismatched_meta = build_training_metadata(mismatched_cfg, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            good_path = artifact_path(tmpdir, "best_model", cfg, 42, ".pth")
            bad_path = os.path.join(tmpdir, "best_model_wrong_seed42.pth")
            torch.save({"model_state_dict": {"w": torch.ones(1)}, "metadata": matching_meta}, good_path)
            torch.save({"model_state_dict": {"w": torch.ones(1)}, "metadata": mismatched_meta}, bad_path)

            compatible, incompatible = compatible_best_model_paths(tmpdir, cfg, seed=42, device="cpu")
            ok, reason = is_compatible_checkpoint(good_path, cfg, seed=42, device="cpu")

        self.assertTrue(ok, reason)
        self.assertEqual([good_path], compatible)
        self.assertEqual(1, len(incompatible))
        self.assertIn("best_model_wrong_seed42.pth", incompatible[0][0])

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
