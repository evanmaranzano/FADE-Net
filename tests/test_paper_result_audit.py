import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from config import Config
from dataset import file_sha256, split_filename_with_tag
from experiment import artifact_path, build_training_metadata
from audit_paper_results import (
    apply_ablation_profile,
    audit_candidate,
    load_checkpoint_metadata,
    load_split_payload,
    parse_ablation_ids,
    parse_seeds,
    populate_runtime_model_metadata,
    write_audit,
)


class PaperResultAuditTests(unittest.TestCase):
    def make_config(self):
        cfg = Config()
        cfg.backbone_source = "timm"
        cfg.backbone_name = "mobilenetv4_conv_small"
        cfg.backbone_pretrained = True
        cfg.split_protocol = "72-8-20"
        cfg.use_hybrid_attention = True
        cfg.use_dldl_v2 = True
        cfg.use_multi_scale = True
        cfg.use_spp = True
        cfg.use_mv_loss = True
        cfg.hybrid_attention_replaced_blocks = ()
        return cfg

    def write_paper_ready_artifacts(
        self,
        root_dir,
        cfg,
        seed=42,
        split_legacy=False,
        experiment_tag=None,
        split_file_tag=None,
    ):
        cfg.experiment_tag = experiment_tag
        cfg.split_file_tag = split_file_tag
        split_file = split_filename_with_tag("dataset_split_AFAD_72_8_20.json", split_file_tag)
        split_payload = {
            "_metadata": {
                "version": 2,
                "num_samples": 3,
                "split_ratios": [0.72, 0.08, 0.20],
                "dataset_fingerprint": "dataset-hash",
                "legacy_upgraded": split_legacy,
            },
            "train": [0],
            "val": [1],
            "test": [2],
        }
        split_path = Path(root_dir) / split_file
        split_path.write_text(json.dumps(split_payload), encoding="utf-8")
        split_fingerprint = file_sha256(split_path)
        cfg.split_metadata = {
            "split_file": split_file,
            "split_file_tag": split_file_tag,
            "fingerprint": split_fingerprint,
            "dataset_fingerprint": "dataset-hash",
            "legacy_upgraded": split_legacy,
        }
        populate_runtime_model_metadata(cfg)
        metadata = build_training_metadata(cfg, seed)

        state = {"w": torch.ones(1)}
        for kind in ("best_model", "last_checkpoint"):
            torch.save(
                {"model_state_dict": state, "metadata": metadata},
                artifact_path(str(root_dir), kind, cfg, seed, ".pth"),
            )

        result_path = Path(artifact_path(str(root_dir), "final_result", cfg, seed, ".txt"))
        result_path.write_text(
            "\n".join([
                "MAE_raw: 3.40",
                "MAE_flip: 3.35",
                "MAE_multi: 3.30",
                "Selected_TTA: multi",
                "Selected_Test_MAE: 3.30",
                f"Experiment ID: {metadata['experiment_id']}",
            ]),
            encoding="utf-8",
        )
        return metadata

    def test_parse_seeds_trims_empty_values(self):
        self.assertEqual([42, 3407, 2026], parse_seeds("42, 3407,,2026"))

    def test_parse_ablation_ids_validates_known_ids(self):
        self.assertEqual(["A0", "A7", "A9"], parse_ablation_ids("a0,A7, a9"))
        with self.assertRaises(ValueError):
            parse_ablation_ids("A10")

    def test_audit_blocks_untagged_complete_metadata_chain_for_paper_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self.make_config()
            self.write_paper_ready_artifacts(tmpdir, cfg)

            row = audit_candidate(Path(tmpdir), cfg, seed=42)

        self.assertEqual("blocked", row["status"])
        self.assertIn("formal_v1", row["reasons"])
        self.assertEqual("3.30", row["selected_test_mae"])
        self.assertEqual("multi", row["selected_tta"])
        self.assertEqual("False", str(row["legacy_split_upgraded"]))

    def test_audit_accepts_tagged_split_metadata_chain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self.make_config()
            metadata = self.write_paper_ready_artifacts(tmpdir, cfg, split_file_tag="formal_v1")

            row = audit_candidate(Path(tmpdir), cfg, seed=42, ablation_id="A3")

        self.assertEqual("paper-ready", row["status"])
        self.assertEqual("single-row evidence; not final mean/std paper table", row["scope"])
        self.assertEqual("formal_v1", row["split_file_tag"])
        self.assertEqual("formal_v1", metadata["split_file_tag"])
        self.assertEqual("A3", row["ablation_id"])
        self.assertIn("dataset_split_AFAD_72_8_20_formal_v1.json", row["split_file"])
        self.assertIn("splitfile-formal_v1", row["experiment_id"])

    def test_audit_distinguishes_loss_ablation_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_full = self.make_config()
            cfg_triplet = self.make_config()
            apply_ablation_profile(cfg_triplet, "A7")
            cfg_asym = self.make_config()
            apply_ablation_profile(cfg_asym, "A8")

            self.write_paper_ready_artifacts(tmpdir, cfg_full, split_file_tag="formal_v1")
            self.write_paper_ready_artifacts(tmpdir, cfg_triplet, split_file_tag="formal_v1")
            self.write_paper_ready_artifacts(tmpdir, cfg_asym, split_file_tag="formal_v1")

            full_row = audit_candidate(Path(tmpdir), cfg_full, seed=42, ablation_id="A3")
            triplet_row = audit_candidate(Path(tmpdir), cfg_triplet, seed=42, ablation_id="A7")
            asym_row = audit_candidate(Path(tmpdir), cfg_asym, seed=42, ablation_id="A8")

        self.assertEqual("paper-ready", full_row["status"])
        self.assertEqual("paper-ready", triplet_row["status"])
        self.assertEqual("paper-ready", asym_row["status"])
        self.assertNotEqual(full_row["result_path"], triplet_row["result_path"])
        self.assertNotEqual(full_row["result_path"], asym_row["result_path"])
        self.assertTrue(triplet_row["use_adaptive_triplet"])
        self.assertTrue(asym_row["use_asymmetric_ordinal"])
        self.assertIn("TRIPLET", triplet_row["experiment_id"])
        self.assertIn("ASYM", asym_row["experiment_id"])

    def test_audit_blocks_checkpoint_metadata_mismatch_with_same_artifact_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self.make_config()
            self.write_paper_ready_artifacts(tmpdir, cfg, split_file_tag="formal_v1")

            for kind in ("best_model", "last_checkpoint"):
                checkpoint_path = Path(artifact_path(str(tmpdir), kind, cfg, 42, ".pth"))
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                checkpoint["metadata"] = dict(checkpoint["metadata"])
                checkpoint["metadata"]["loss"] = dict(checkpoint["metadata"]["loss"])
                checkpoint["metadata"]["loss"]["lambda_mv"] = 999
                torch.save(checkpoint, checkpoint_path)

            row = audit_candidate(Path(tmpdir), cfg, seed=42)

        self.assertEqual("blocked", row["status"])
        self.assertIn("checkpoint metadata mismatch", row["reasons"])
        self.assertIn("loss", row["reasons"])

    def test_audit_blocks_runtime_backbone_metadata_drift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self.make_config()
            self.write_paper_ready_artifacts(tmpdir, cfg, split_file_tag="formal_v1")

            for kind in ("best_model", "last_checkpoint"):
                checkpoint_path = Path(artifact_path(str(tmpdir), kind, cfg, 42, ".pth"))
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                checkpoint["metadata"] = dict(checkpoint["metadata"])
                checkpoint["metadata"]["backbone"] = dict(checkpoint["metadata"]["backbone"])
                checkpoint["metadata"]["backbone"]["effective_msff_channels"] = [1, 2]
                torch.save(checkpoint, checkpoint_path)

            row = audit_candidate(Path(tmpdir), cfg, seed=42)

        self.assertEqual("blocked", row["status"])
        self.assertIn("checkpoint metadata mismatch", row["reasons"])
        self.assertIn("backbone", row["reasons"])

    def test_audit_blocks_smoke_and_legacy_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self.make_config()
            self.write_paper_ready_artifacts(tmpdir, cfg, split_legacy=True, experiment_tag="smoke")

            row = audit_candidate(Path(tmpdir), cfg, seed=42)

        self.assertEqual("blocked", row["status"])
        self.assertIn("smoke experiment_tag", row["reasons"])
        self.assertIn("legacy split", row["reasons"])

    def test_audit_blocks_missing_split_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self.make_config()
            self.write_paper_ready_artifacts(tmpdir, cfg)
            split_path = Path(tmpdir) / "dataset_split_AFAD_72_8_20.json"
            split_path.write_text(json.dumps({"train": [0], "val": [1], "test": [2]}), encoding="utf-8")

            row = audit_candidate(Path(tmpdir), cfg, seed=42)

        self.assertEqual("blocked", row["status"])
        self.assertIn("split file has no _metadata", row["reasons"])

    def test_write_audit_preserves_status_and_reasons(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "audit.csv"
            write_audit([{"status": "blocked", "reasons": "missing result"}], output)

            content = output.read_text(encoding="utf-8")

        self.assertIn("status,scope,reasons", content)
        self.assertIn("blocked,,missing result", content)

    def test_load_checkpoint_metadata_uses_weights_only_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            checkpoint_path.write_bytes(b"placeholder")
            calls = []

            def fake_load(path, **kwargs):
                calls.append((path, kwargs))
                return {"model_state_dict": {}, "metadata": {"experiment_id": "safe"}}

            with unittest.mock.patch("audit_paper_results.torch.load", side_effect=fake_load):
                metadata, error = load_checkpoint_metadata(checkpoint_path)

        self.assertIsNone(error)
        self.assertEqual({"experiment_id": "safe"}, metadata)
        self.assertTrue(calls[0][1]["weights_only"])

    def test_load_checkpoint_metadata_rejects_unsafe_legacy_torch_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            checkpoint_path.write_bytes(b"placeholder")
            calls = []

            def fake_load(path, **kwargs):
                calls.append((path, kwargs))
                if kwargs.get("weights_only") is True:
                    raise TypeError("weights_only is not supported")
                return {"model_state_dict": {}, "metadata": {"experiment_id": "unsafe"}}

            with unittest.mock.patch("audit_paper_results.torch.load", side_effect=fake_load):
                metadata, error = load_checkpoint_metadata(checkpoint_path)

        self.assertIsNone(metadata)
        self.assertIn("weights_only=True", error)
        self.assertEqual(1, len(calls))

    def test_load_split_payload_rejects_paths_outside_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "root"
            root.mkdir()
            outside = Path(tmpdir) / "outside.json"
            outside.write_text(json.dumps({"_metadata": {}}), encoding="utf-8")
            reasons = []

            split_path, payload = load_split_payload(root, "../outside.json", reasons)

        self.assertIsNone(split_path)
        self.assertIsNone(payload)
        self.assertIn("outside project root", "; ".join(reasons))

    def test_write_audit_refuses_to_overwrite_without_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "audit.csv"
            output.write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                write_audit([{"status": "blocked"}], output)

            self.assertEqual("existing", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
