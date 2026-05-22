import argparse
import csv
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_backbone_screening
from run_backbone_screening import (
    build_command,
    config_for_candidate,
    log_paths_for_candidate,
    parse_selected_test_mae,
    write_manifest,
)


class BackboneScreeningRunnerTests(unittest.TestCase):
    def make_args(self):
        return argparse.Namespace(
            seed=42,
            epochs=20,
            split="72-8-20",
            afad_dir="F:/data/AFAD",
            no_pretrained=False,
            max_train_batches=None,
            max_val_batches=None,
            max_test_batches=None,
            experiment_tag=None,
            resume=False,
            overwrite_artifacts=False,
            allow_legacy_split_upgrade=False,
            split_file_tag=None,
            ablation_id=None,
            log_dir="F:/FADE-Net/docs/backbone_screening_logs",
        )

    def test_build_command_includes_dataset_override_and_timm_backbone(self):
        command = build_command(self.make_args(), "timm", "mobilenetv4_conv_small")

        self.assertIn("--afad_dir", command)
        self.assertIn("F:/data/AFAD", command)
        self.assertIn("--backbone_source", command)
        self.assertIn("timm", command)
        self.assertIn("--backbone_name", command)
        self.assertIn("mobilenetv4_conv_small", command)
        self.assertEqual("-u", command[1])
        self.assertIn("--fresh", command)
        self.assertNotIn("--resume", command)
        self.assertNotIn("--no_pretrained", command)

    def test_build_command_can_resume_matching_checkpoint(self):
        args = self.make_args()
        args.resume = True

        command = build_command(args, "timm", "mobilenetv4_conv_small")

        self.assertIn("--resume", command)
        self.assertNotIn("--fresh", command)

    def test_build_command_can_disable_pretrained_for_smoke_or_ablation(self):
        args = self.make_args()
        args.no_pretrained = True

        command = build_command(args, "timm", "mobilenetv4_conv_small")

        self.assertIn("--no_pretrained", command)

    def test_build_command_can_pass_smoke_batch_limits(self):
        args = self.make_args()
        args.max_train_batches = 1
        args.max_val_batches = 1
        args.max_test_batches = 1

        command = build_command(args, "timm", "repvit_m0_9")

        self.assertIn("--max_train_batches", command)
        self.assertIn("--max_val_batches", command)
        self.assertIn("--max_test_batches", command)

    def test_build_command_can_pass_reproducibility_override_flags(self):
        args = self.make_args()
        args.allow_legacy_split_upgrade = True
        args.overwrite_artifacts = True

        command = build_command(args, "timm", "repvit_m0_9")

        self.assertIn("--allow_legacy_split_upgrade", command)
        self.assertIn("--overwrite_artifacts", command)

    def test_build_command_passes_experiment_tag(self):
        args = self.make_args()
        args.experiment_tag = "smoke"

        command = build_command(args, "timm", "mobilenetv4_conv_small_050")

        self.assertIn("--experiment_tag", command)
        self.assertIn("smoke", command)

    def test_build_command_passes_split_file_tag(self):
        args = self.make_args()
        args.split_file_tag = "formal_v1"

        command = build_command(args, "timm", "mobilenetv4_conv_small")

        self.assertIn("--split_file_tag", command)
        self.assertIn("formal_v1", command)

    def test_build_command_passes_ablation_profile_flags(self):
        args = self.make_args()
        args.ablation_id = "A7"

        command = build_command(args, "timm", "mobilenetv4_conv_small")

        self.assertIn("--msff", command)
        self.assertIn("--spp", command)
        self.assertIn("--triplet", command)
        self.assertIn("--no-texture", command)
        self.assertIn("--no-freq", command)
        self.assertIn("--no-moe", command)
        self.assertIn("--no-asym", command)

    def test_build_command_keeps_default_torchvision_backbone_implicit(self):
        command = build_command(self.make_args(), "torchvision", "mobilenet_v3_large")

        self.assertNotIn("--backbone_source", command)
        self.assertNotIn("--backbone_name", command)

    def test_config_for_candidate_resolves_absolute_dataset_path(self):
        args = self.make_args()
        args.afad_dir = "."

        cfg = config_for_candidate(args, "timm", "repvit_m0_9")

        self.assertEqual("timm", cfg.backbone_source)
        self.assertEqual("repvit_m0_9", cfg.backbone_name)
        self.assertTrue(cfg.backbone_pretrained)
        self.assertEqual(os.path.abspath("."), cfg.afad_dir)

    def test_config_for_candidate_sets_split_file_tag(self):
        args = self.make_args()
        args.split_file_tag = "formal_v1"

        cfg = config_for_candidate(args, "timm", "repvit_m0_9")

        self.assertEqual("formal_v1", cfg.split_file_tag)

    def test_config_for_candidate_applies_ablation_profile(self):
        args = self.make_args()
        args.ablation_id = "A7"

        cfg = config_for_candidate(args, "timm", "mobilenetv4_conv_small")

        self.assertTrue(cfg.use_multi_scale)
        self.assertTrue(cfg.use_spp)
        self.assertTrue(cfg.use_adaptive_triplet)
        self.assertFalse(cfg.use_texture_branch)

    def test_select_candidates_filters_by_backbone_name(self):
        self.assertTrue(hasattr(run_backbone_screening, "select_candidates"))

        selected = run_backbone_screening.select_candidates("mobilenetv4_conv_small,repvit_m0_9")

        self.assertEqual((
            ("timm", "mobilenetv4_conv_small"),
            ("timm", "repvit_m0_9"),
        ), selected)

    def test_select_candidates_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            run_backbone_screening.select_candidates("not_a_real_backbone")

    def test_parse_selected_test_mae_uses_explicit_result_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stale = Path(tmpdir) / "final_result_stale_seed42.txt"
            expected = Path(tmpdir) / "final_result_expected_seed42.txt"
            stale.write_text("Selected_Test_MAE: 9.99\n", encoding="utf-8")
            expected.write_text("MAE_raw: 4.0\nSelected_Test_MAE: 3.21\n", encoding="utf-8")

            self.assertEqual("3.21", parse_selected_test_mae(expected))

    def test_parse_result_metrics_reads_all_tta_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "final_result_expected_seed42.txt"
            result_path.write_text(
                "\n".join([
                    "MAE_raw: 3.40",
                    "MAE_flip: 3.35",
                    "MAE_multi: 3.30",
                    "Selected_TTA: multi",
                    "Selected_Test_MAE: 3.30",
                    "Experiment ID: FADE-Net_seed42",
                ]),
                encoding="utf-8",
            )

            self.assertTrue(hasattr(run_backbone_screening, "parse_result_metrics"))
            metrics = run_backbone_screening.parse_result_metrics(result_path)

            self.assertEqual("3.40", metrics["mae_raw"])
            self.assertEqual("3.35", metrics["mae_flip"])
            self.assertEqual("3.30", metrics["mae_multi"])
            self.assertEqual("multi", metrics["selected_tta"])
            self.assertEqual("3.30", metrics["selected_test_mae"])
            self.assertEqual("FADE-Net_seed42", metrics["experiment_id"])

    def test_parse_result_metrics_returns_blank_fields_for_missing_or_directory_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_metrics = run_backbone_screening.parse_result_metrics(Path(tmpdir) / "missing.txt")
            directory_metrics = run_backbone_screening.parse_result_metrics(Path(tmpdir))

            for metrics in (missing_metrics, directory_metrics):
                self.assertEqual("", metrics["mae_raw"])
                self.assertEqual("", metrics["mae_flip"])
                self.assertEqual("", metrics["mae_multi"])
                self.assertEqual("", metrics["selected_tta"])
                self.assertEqual("", metrics["selected_test_mae"])
                self.assertEqual("", metrics["experiment_id"])

    def test_manifest_includes_tta_metrics_and_result_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "screen.csv"
            write_manifest([
                {
                    "source": "timm",
                    "backbone": "mobilenetv4_conv_small",
                    "seed": 42,
                    "epochs": 20,
                    "split": "72-8-20",
                    "split_file_tag": "",
                    "status": "ok",
                    "returncode": 0,
                    "selected_test_mae": "3.30",
                    "mae_raw": "3.40",
                    "mae_flip": "3.35",
                    "mae_multi": "3.30",
                    "selected_tta": "multi",
                    "experiment_id": "FADE-Net_seed42",
                    "result_path": "F:/FADE-Net/final_result.txt",
                    "stdout_log": "F:/FADE-Net/docs/backbone_screening_logs/run.out.log",
                    "stderr_log": "F:/FADE-Net/docs/backbone_screening_logs/run.err.log",
                    "command": "python src/train.py",
                }
            ], output)

            with output.open(encoding="utf-8", newline="") as f:
                row = next(csv.DictReader(f))

            self.assertEqual("3.40", row["mae_raw"])
            self.assertEqual("3.35", row["mae_flip"])
            self.assertEqual("3.30", row["mae_multi"])
            self.assertEqual("multi", row["selected_tta"])
            self.assertEqual("", row["split_file_tag"])
            self.assertEqual("FADE-Net_seed42", row["experiment_id"])
            self.assertEqual("F:/FADE-Net/final_result.txt", row["result_path"])
            self.assertEqual("F:/FADE-Net/docs/backbone_screening_logs/run.out.log", row["stdout_log"])
            self.assertEqual("F:/FADE-Net/docs/backbone_screening_logs/run.err.log", row["stderr_log"])

    def test_log_paths_include_candidate_weight_split_seed_and_tag(self):
        args = self.make_args()
        args.log_dir = "F:/logs"
        args.experiment_tag = "smoke"

        stdout_log, stderr_log = log_paths_for_candidate(args, "timm", "mobilenetv4_conv_small")

        self.assertEqual(Path("F:/logs/timm_mobilenetv4_conv_small_pretrained_72-8-20_seed42_smoke.out.log"), stdout_log)
        self.assertEqual(Path("F:/logs/timm_mobilenetv4_conv_small_pretrained_72-8-20_seed42_smoke.err.log"), stderr_log)

    def test_log_paths_include_split_file_tag(self):
        args = self.make_args()
        args.log_dir = "F:/logs"
        args.split_file_tag = "formal v1"

        stdout_log, stderr_log = log_paths_for_candidate(args, "timm", "mobilenetv4_conv_small")

        self.assertEqual(
            Path("F:/logs/timm_mobilenetv4_conv_small_pretrained_72-8-20_splitfile-formal-v1_seed42.out.log"),
            stdout_log,
        )
        self.assertEqual(
            Path("F:/logs/timm_mobilenetv4_conv_small_pretrained_72-8-20_splitfile-formal-v1_seed42.err.log"),
            stderr_log,
        )

    def test_log_paths_include_ablation_id(self):
        args = self.make_args()
        args.log_dir = "F:/logs"
        args.ablation_id = "A9"

        stdout_log, stderr_log = log_paths_for_candidate(args, "timm", "mobilenetv4_conv_small")

        self.assertEqual(Path("F:/logs/timm_mobilenetv4_conv_small_pretrained_72-8-20_A9_seed42.out.log"), stdout_log)
        self.assertEqual(Path("F:/logs/timm_mobilenetv4_conv_small_pretrained_72-8-20_A9_seed42.err.log"), stderr_log)

    def test_main_flushes_manifest_after_each_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "screen.csv"
            calls = []

            def fake_run(_command, cwd, stdout, stderr, text, check):
                calls.append((cwd, check))
                stdout.write("stdout proof")
                stderr.write("stderr proof")
                if len(calls) == 2:
                    self.assertTrue(output.exists())
                    with output.open(encoding="utf-8", newline="") as f:
                        rows = list(csv.DictReader(f))
                    self.assertEqual(1, len(rows))
                    self.assertEqual("mobilenet_v3_large", rows[0]["backbone"])
                return mock.Mock(returncode=0)

            argv = [
                "run_backbone_screening.py",
                "--run",
                "--overwrite_artifacts",
                "--output",
                str(output),
                "--epochs",
                "1",
                "--log_dir",
                str(Path(tmpdir) / "logs"),
            ]
            with mock.patch.object(run_backbone_screening, "DEFAULT_CANDIDATES", (
                ("torchvision", "mobilenet_v3_large"),
                ("timm", "mobilenetv4_conv_small"),
            )):
                with mock.patch.object(run_backbone_screening.subprocess, "run", side_effect=fake_run):
                    with mock.patch.object(run_backbone_screening.sys, "argv", argv):
                        run_backbone_screening.main()

            self.assertEqual(2, len(calls))

            baseline_out = Path(tmpdir) / "logs" / "torchvision_mobilenet_v3_large_pretrained_72-8-20_seed42.out.log"
            baseline_err = Path(tmpdir) / "logs" / "torchvision_mobilenet_v3_large_pretrained_72-8-20_seed42.err.log"
            self.assertEqual("stdout proof", baseline_out.read_text(encoding="utf-8"))
            self.assertEqual("stderr proof", baseline_err.read_text(encoding="utf-8"))

    def test_main_append_preserves_existing_manifest_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "screen.csv"
            write_manifest([
                {
                    "source": "torchvision",
                    "backbone": "mobilenet_v3_large",
                    "seed": 42,
                    "epochs": 20,
                    "split": "72-8-20",
                    "split_file_tag": "",
                    "status": "ok",
                    "returncode": 0,
                    "selected_test_mae": "3.50",
                    "mae_raw": "3.60",
                    "mae_flip": "3.55",
                    "mae_multi": "3.50",
                    "selected_tta": "multi",
                    "experiment_id": "baseline_seed42",
                    "result_path": "F:/FADE-Net/final_result_baseline.txt",
                    "stdout_log": "F:/FADE-Net/docs/backbone_screening_logs/baseline.out.log",
                    "stderr_log": "F:/FADE-Net/docs/backbone_screening_logs/baseline.err.log",
                    "command": "python src/train.py",
                }
            ], output)

            argv = [
                "run_backbone_screening.py",
                "--run",
                "--append",
                "--overwrite_artifacts",
                "--output",
                str(output),
                "--epochs",
                "1",
                "--candidates",
                "mobilenetv4_conv_small",
                "--log_dir",
                str(Path(tmpdir) / "logs"),
            ]
            def fake_run(_command, cwd, stdout, stderr, text, check):
                stdout.write("")
                stderr.write("")
                return mock.Mock(returncode=0)

            with mock.patch.object(run_backbone_screening.subprocess, "run", side_effect=fake_run):
                with mock.patch.object(run_backbone_screening.sys, "argv", argv):
                    run_backbone_screening.main()

            with output.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(2, len(rows))
            self.assertEqual("mobilenet_v3_large", rows[0]["backbone"])
            self.assertEqual("mobilenetv4_conv_small", rows[1]["backbone"])
            self.assertIn("stdout_log", rows[1])
            self.assertIn("stderr_log", rows[1])

    def test_main_manifest_records_split_file_tag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "screen.csv"
            argv = [
                "run_backbone_screening.py",
                "--output",
                str(output),
                "--candidates",
                "mobilenetv4_conv_small",
                "--split_file_tag",
                "formal_v1",
            ]
            with mock.patch.object(run_backbone_screening.sys, "argv", argv):
                run_backbone_screening.main()

            with output.open(encoding="utf-8", newline="") as f:
                row = next(csv.DictReader(f))

            self.assertEqual("formal_v1", row["split_file_tag"])
            self.assertIn("--split_file_tag formal_v1", row["command"])
            self.assertIn("splitfile-formal_v1", row["result_path"])

    def test_main_manifest_records_ablation_id_and_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "screen.csv"
            argv = [
                "run_backbone_screening.py",
                "--output",
                str(output),
                "--candidates",
                "mobilenetv4_conv_small",
                "--ablation_id",
                "A7",
            ]
            with mock.patch.object(run_backbone_screening.sys, "argv", argv):
                run_backbone_screening.main()

            with output.open(encoding="utf-8", newline="") as f:
                row = next(csv.DictReader(f))

            self.assertEqual("A7", row["ablation_id"])
            self.assertEqual("True", row["use_adaptive_triplet"])
            self.assertEqual("False", row["use_texture_branch"])
            self.assertIn("--triplet", row["command"])
            self.assertIn("TRIPLET", row["result_path"])

    def test_main_refuses_existing_candidate_artifacts_without_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "screen.csv"
            log_dir = Path(tmpdir) / "logs"
            log_dir.mkdir()
            existing_log = log_dir / "timm_mobilenetv4_conv_small_pretrained_72-8-20_seed42.out.log"
            existing_log.write_text("old log", encoding="utf-8")

            argv = [
                "run_backbone_screening.py",
                "--run",
                "--output",
                str(output),
                "--log_dir",
                str(log_dir),
                "--candidates",
                "mobilenetv4_conv_small",
            ]

            with mock.patch.object(run_backbone_screening.sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    run_backbone_screening.main()

    def test_main_default_screening_uses_pretrained_backbone(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "screen.csv"
            argv = [
                "run_backbone_screening.py",
                "--output",
                str(output),
                "--candidates",
                "mobilenetv4_conv_small",
            ]
            with mock.patch.object(run_backbone_screening.sys, "argv", argv):
                run_backbone_screening.main()

            with output.open(encoding="utf-8", newline="") as f:
                row = next(csv.DictReader(f))

            self.assertNotIn("--no_pretrained", row["command"])

    def test_main_marks_successful_run_without_experiment_id_incomplete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "screen.csv"
            argv = [
                "run_backbone_screening.py",
                "--run",
                "--overwrite_artifacts",
                "--output",
                str(output),
                "--epochs",
                "1",
                "--candidates",
                "mobilenetv4_conv_small",
                "--log_dir",
                str(Path(tmpdir) / "logs"),
            ]

            def fake_run(_command, cwd, stdout, stderr, text, check):
                return mock.Mock(returncode=0)

            metrics_without_experiment_id = {
                "mae_raw": "3.40",
                "mae_flip": "3.35",
                "mae_multi": "3.30",
                "selected_tta": "multi",
                "selected_test_mae": "3.30",
                "experiment_id": "",
            }

            with mock.patch.object(run_backbone_screening.subprocess, "run", side_effect=fake_run):
                with mock.patch.object(run_backbone_screening, "parse_result_metrics", return_value=metrics_without_experiment_id):
                    with mock.patch.object(run_backbone_screening.sys, "argv", argv):
                        run_backbone_screening.main()

            with output.open(encoding="utf-8", newline="") as f:
                row = next(csv.DictReader(f))

            self.assertEqual("incomplete", row["status"])
            self.assertEqual("3.30", row["selected_test_mae"])
            self.assertEqual("", row["experiment_id"])


if __name__ == "__main__":
    unittest.main()
