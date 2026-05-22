import argparse
import csv
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import Config
from dataset import file_sha256
from experiment import artifact_path, build_training_metadata, checkpoint_metadata_mismatches
from model import LightweightAgeEstimator
from run_backbone_screening import parse_result_metrics, select_candidates


AUDIT_FIELDS = [
    "source",
    "backbone",
    "seed",
    "split",
    "split_file_tag",
    "status",
    "reasons",
    "selected_test_mae",
    "mae_raw",
    "mae_flip",
    "mae_multi",
    "selected_tta",
    "experiment_id",
    "result_path",
    "best_model_path",
    "last_checkpoint_path",
    "split_file",
    "split_fingerprint",
    "dataset_fingerprint",
    "legacy_split_upgraded",
]

PAPER_SPLIT_FILE_TAG = "formal_v1"


def parse_seeds(seeds_arg):
    return [int(seed.strip()) for seed in seeds_arg.split(",") if seed.strip()]


def config_for_candidate(args, source, name):
    cfg = Config()
    cfg.split_protocol = args.split
    cfg.backbone_source = source
    cfg.backbone_name = name
    cfg.backbone_pretrained = not args.no_pretrained
    cfg.experiment_tag = args.experiment_tag
    cfg.split_file_tag = getattr(args, "split_file_tag", None)
    return cfg


def populate_runtime_model_metadata(cfg):
    """Mirror train.py metadata setup without downloading pretrained weights."""
    original_pretrained = cfg.backbone_pretrained
    cfg.backbone_pretrained = False
    try:
        with torch.no_grad(), redirect_stdout(io.StringIO()):
            model = LightweightAgeEstimator(cfg)
        del model
    finally:
        cfg.backbone_pretrained = original_pretrained


def load_checkpoint_metadata(path):
    if not path.is_file():
        return None, f"missing checkpoint: {path}"
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        return None, f"not a packaged checkpoint: {path}"
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict) or not metadata:
        return None, f"checkpoint has no metadata: {path}"
    return metadata, None


def validate_float_field(metrics, key, reasons):
    value = metrics.get(key, "")
    if value == "":
        reasons.append(f"missing result metric: {key}")
        return ""
    try:
        float(value)
    except ValueError:
        reasons.append(f"invalid numeric metric {key}: {value!r}")
    return value


def load_split_payload(root_dir, split_file, reasons):
    if not split_file:
        reasons.append("missing split_file in checkpoint metadata")
        return None, None

    split_path = Path(root_dir) / split_file
    if not split_path.is_file():
        reasons.append(f"missing split file: {split_path}")
        return None, None

    with split_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return split_path, payload


def audit_candidate(root_dir, cfg, seed):
    reasons = []
    result_path = Path(artifact_path(str(root_dir), "final_result", cfg, seed, ".txt"))
    best_model_path = Path(artifact_path(str(root_dir), "best_model", cfg, seed, ".pth"))
    last_checkpoint_path = Path(artifact_path(str(root_dir), "last_checkpoint", cfg, seed, ".pth"))

    metrics = parse_result_metrics(result_path)
    for key in ("selected_test_mae", "mae_raw", "mae_flip", "mae_multi"):
        validate_float_field(metrics, key, reasons)
    if metrics.get("selected_tta") != "multi":
        reasons.append(f"selected_tta must be 'multi', got {metrics.get('selected_tta')!r}")

    best_metadata, error = load_checkpoint_metadata(best_model_path)
    if error:
        reasons.append(error)
    last_metadata, error = load_checkpoint_metadata(last_checkpoint_path)
    if error:
        reasons.append(error)

    metadata = best_metadata or last_metadata or {}
    if best_metadata and last_metadata and best_metadata != last_metadata:
        reasons.append("best_model and last_checkpoint metadata differ")

    experiment_id = metadata.get("experiment_id", metrics.get("experiment_id", ""))
    if metrics.get("experiment_id") != experiment_id:
        reasons.append(
            f"result experiment_id mismatch: result={metrics.get('experiment_id')!r}, checkpoint={experiment_id!r}"
        )

    split_file = metadata.get("split_file")
    split_file_tag = metadata.get("split_file_tag")
    split_fingerprint = metadata.get("split_fingerprint")
    dataset_fingerprint = metadata.get("dataset_fingerprint")
    legacy_split_upgraded = bool(metadata.get("legacy_split_upgraded", False))

    requested_split_file_tag = getattr(cfg, "split_file_tag", None)
    if requested_split_file_tag != PAPER_SPLIT_FILE_TAG:
        reasons.append(
            f"paper audit requires --split_file_tag {PAPER_SPLIT_FILE_TAG}, got {requested_split_file_tag!r}"
        )
    if split_file_tag != PAPER_SPLIT_FILE_TAG:
        reasons.append(f"checkpoint split_file_tag must be {PAPER_SPLIT_FILE_TAG!r}, got {split_file_tag!r}")

    for key, value in (
        ("split_fingerprint", split_fingerprint),
        ("dataset_fingerprint", dataset_fingerprint),
    ):
        if not value:
            reasons.append(f"missing {key} in checkpoint metadata")

    if legacy_split_upgraded:
        reasons.append("legacy split upgrade is not paper-ready")

    if metadata.get("experiment_tag") and str(metadata.get("experiment_tag")).startswith("smoke"):
        reasons.append(f"smoke experiment_tag is not paper-ready: {metadata.get('experiment_tag')!r}")

    split_path, split_payload = load_split_payload(root_dir, split_file, reasons)
    if split_path is not None:
        actual_split_fingerprint = file_sha256(split_path)
        if actual_split_fingerprint != split_fingerprint:
            reasons.append(
                "split fingerprint mismatch: "
                f"file={actual_split_fingerprint}, checkpoint={split_fingerprint}"
            )

    split_metadata = split_payload.get("_metadata", {}) if isinstance(split_payload, dict) else {}
    if split_payload is not None and not split_metadata:
        reasons.append("split file has no _metadata")
    if split_metadata:
        if split_metadata.get("dataset_fingerprint") != dataset_fingerprint:
            reasons.append(
                "dataset fingerprint mismatch: "
                f"split={split_metadata.get('dataset_fingerprint')!r}, checkpoint={dataset_fingerprint!r}"
            )
        if split_metadata.get("legacy_upgraded"):
            reasons.append("split file is marked legacy_upgraded")

    if split_metadata:
        cfg.split_metadata = {
            "split_file": split_file,
            "fingerprint": split_fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
            "legacy_upgraded": bool(split_metadata.get("legacy_upgraded", False)),
        }
    populate_runtime_model_metadata(cfg)
    expected_metadata = build_training_metadata(cfg, seed)
    for label, checkpoint_metadata in (("best_model", best_metadata), ("last_checkpoint", last_metadata)):
        if not checkpoint_metadata:
            continue
        mismatches = checkpoint_metadata_mismatches({"metadata": checkpoint_metadata}, expected_metadata)
        if mismatches:
            details = ", ".join(
                f"{key}: checkpoint={actual!r}, expected={expected!r}"
                for key, actual, expected in mismatches
            )
            reasons.append(f"{label} checkpoint metadata mismatch: {details}")

    expected_experiment_id = expected_metadata["experiment_id"]
    if experiment_id and experiment_id != expected_experiment_id:
        reasons.append(
            f"unexpected experiment_id for requested candidate: "
            f"checkpoint={experiment_id!r}, expected={expected_experiment_id!r}"
        )

    return {
        "source": cfg.backbone_source,
        "backbone": cfg.backbone_name,
        "seed": seed,
        "split": cfg.split_protocol,
        "split_file_tag": split_file_tag or "",
        "status": "paper-ready" if not reasons else "blocked",
        "reasons": "; ".join(reasons),
        "selected_test_mae": metrics.get("selected_test_mae", ""),
        "mae_raw": metrics.get("mae_raw", ""),
        "mae_flip": metrics.get("mae_flip", ""),
        "mae_multi": metrics.get("mae_multi", ""),
        "selected_tta": metrics.get("selected_tta", ""),
        "experiment_id": experiment_id,
        "result_path": str(result_path),
        "best_model_path": str(best_model_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "split_file": split_file or "",
        "split_fingerprint": split_fingerprint or "",
        "dataset_fingerprint": dataset_fingerprint or "",
        "legacy_split_upgraded": legacy_split_upgraded,
    }


def write_audit(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in AUDIT_FIELDS} for row in rows])


def main():
    parser = argparse.ArgumentParser(description="Audit FADE-Net results before using them in paper tables")
    parser.add_argument("--seeds", type=str, default="42,3407,2026", help="Comma-separated seeds to audit")
    parser.add_argument("--split", type=str, default="72-8-20", choices=["80-10-10", "90-5-5", "72-8-20"])
    parser.add_argument("--candidates", type=str, help="Comma-separated backbone names or source/name pairs")
    parser.add_argument("--no_pretrained", dest="no_pretrained", action="store_true", default=False)
    parser.add_argument("--pretrained", dest="no_pretrained", action="store_false", help="Enable pretrained backbones")
    parser.add_argument("--experiment_tag", type=str, help="Audit a tagged side run instead of formal untagged artifacts")
    parser.add_argument("--split_file_tag", type=str, help="Audit tagged split file/artifact identity")
    parser.add_argument("--output", type=str, default=str(ROOT_DIR / "docs" / "paper_result_audit.csv"))
    args = parser.parse_args()

    rows = []
    for source, name in select_candidates(args.candidates):
        cfg = config_for_candidate(args, source, name)
        for seed in parse_seeds(args.seeds):
            rows.append(audit_candidate(ROOT_DIR, cfg, seed))

    output_path = Path(args.output)
    write_audit(rows, output_path)
    print(f"Audit written: {output_path}")


if __name__ == "__main__":
    main()
