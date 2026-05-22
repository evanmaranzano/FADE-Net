import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import Config
from ablation_profiles import (
    ABLATION_FIELDS,
    ablation_cli_flags,
    ablation_row_flags,
    apply_ablation_profile,
    parse_ablation_ids,
)
from experiment import artifact_path, build_training_metadata, optional_sanitize_token

DEFAULT_CANDIDATES = (
    ("torchvision", "mobilenet_v3_large"),
    ("timm", "mobilenetv4_conv_small"),
    ("timm", "mobilenetv4_conv_small_050"),
    ("timm", "repvit_m0_9"),
)

MANIFEST_FIELDS = [
    "source",
    "backbone",
    "seed",
    "epochs",
    "split",
    "split_file_tag",
    "ablation_id",
    *ABLATION_FIELDS,
    "status",
    "returncode",
    "selected_test_mae",
    "mae_raw",
    "mae_flip",
    "mae_multi",
    "selected_tta",
    "experiment_id",
    "result_path",
    "stdout_log",
    "stderr_log",
    "command",
]


def select_candidates(candidates_arg=None):
    if not candidates_arg:
        return DEFAULT_CANDIDATES

    available = {}
    for source, name in DEFAULT_CANDIDATES:
        available[name] = (source, name)
        available[f"{source}/{name}"] = (source, name)

    selected = []
    unknown = []
    for token in candidates_arg.split(","):
        key = token.strip()
        if not key:
            continue
        candidate = available.get(key)
        if candidate is None:
            unknown.append(key)
        elif candidate not in selected:
            selected.append(candidate)

    if unknown:
        allowed = ", ".join(name for _source, name in DEFAULT_CANDIDATES)
        raise ValueError(f"Unknown backbone candidate(s): {', '.join(unknown)}. Allowed: {allowed}")
    if not selected:
        raise ValueError("No backbone candidates selected.")
    return tuple(selected)


def build_command(args, source, name):
    command = [
        sys.executable,
        "-u",
        str(ROOT_DIR / "src" / "train.py"),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--split",
        args.split,
    ]

    command.append("--resume" if args.resume else "--fresh")
    if args.no_pretrained:
        command.append("--no_pretrained")
    if args.afad_dir:
        command.extend(["--afad_dir", args.afad_dir])
    if args.max_train_batches is not None:
        command.extend(["--max_train_batches", str(args.max_train_batches)])
    if args.max_val_batches is not None:
        command.extend(["--max_val_batches", str(args.max_val_batches)])
    if args.max_test_batches is not None:
        command.extend(["--max_test_batches", str(args.max_test_batches)])
    if args.experiment_tag:
        command.extend(["--experiment_tag", args.experiment_tag])
    if getattr(args, "split_file_tag", None):
        command.extend(["--split_file_tag", args.split_file_tag])
    if args.allow_legacy_split_upgrade:
        command.append("--allow_legacy_split_upgrade")
    if args.overwrite_artifacts:
        command.append("--overwrite_artifacts")
    command.extend(ablation_cli_flags(getattr(args, "ablation_id", None)))
    if source != "torchvision" or name != "mobilenet_v3_large":
        command.extend(["--backbone_source", source, "--backbone_name", name])
    return command


def parse_selected_test_mae(result_path):
    return parse_result_metrics(result_path)["selected_test_mae"]


def parse_result_metrics(result_path):
    metrics = {
        "mae_raw": "",
        "mae_flip": "",
        "mae_multi": "",
        "selected_tta": "",
        "selected_test_mae": "",
        "experiment_id": "",
    }
    if not result_path.is_file():
        return metrics

    field_map = {
        "MAE_raw": "mae_raw",
        "MAE_flip": "mae_flip",
        "MAE_multi": "mae_multi",
        "Selected_TTA": "selected_tta",
        "Selected_Test_MAE": "selected_test_mae",
        "Experiment ID": "experiment_id",
    }

    for line in result_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metric_name = field_map.get(key.strip())
        if metric_name:
            metrics[metric_name] = value.strip()
    return metrics


def empty_result_metrics():
    return {
        "mae_raw": "",
        "mae_flip": "",
        "mae_multi": "",
        "selected_tta": "",
        "selected_test_mae": "",
        "experiment_id": "",
    }


def config_for_candidate(args, source, name):
    cfg = Config()
    cfg.split_protocol = args.split
    cfg.backbone_source = source
    cfg.backbone_name = name
    cfg.backbone_pretrained = not args.no_pretrained
    cfg.experiment_tag = args.experiment_tag
    cfg.split_file_tag = getattr(args, "split_file_tag", None)
    cfg.allow_legacy_split_upgrade = bool(args.allow_legacy_split_upgrade)
    apply_ablation_profile(cfg, getattr(args, "ablation_id", None))
    if args.afad_dir:
        cfg.afad_dir = os.path.abspath(args.afad_dir)
    return cfg


def log_paths_for_candidate(args, source, name):
    log_dir = Path(args.log_dir)
    weight_tag = "scratch" if args.no_pretrained else "pretrained"
    split_file_tag = optional_sanitize_token(getattr(args, "split_file_tag", None))
    split_tag = f"_splitfile-{split_file_tag}" if split_file_tag else ""
    ablation_tag = f"_{args.ablation_id}" if getattr(args, "ablation_id", None) else ""
    tag = f"_{args.experiment_tag}" if args.experiment_tag else ""
    stem = f"{source}_{name}_{weight_tag}_{args.split}{split_tag}{ablation_tag}_seed{args.seed}{tag}"
    return log_dir / f"{stem}.out.log", log_dir / f"{stem}.err.log"


def write_manifest(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in MANIFEST_FIELDS} for row in rows])


def read_manifest(output_path):
    if not output_path.is_file():
        return []
    with output_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Run or print the FADE-Net backbone short-screen matrix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--split", type=str, default="72-8-20", choices=["80-10-10", "90-5-5", "72-8-20"])
    parser.add_argument("--afad_dir", type=str, default=os.environ.get("FADE_NET_AFAD_DIR"))
    parser.add_argument("--no_pretrained", dest="no_pretrained", action="store_true", default=False)
    parser.add_argument("--pretrained", dest="no_pretrained", action="store_false", help="Enable pretrained backbones")
    parser.add_argument("--max_train_batches", type=int, help="Limit valid train batches for pipeline smoke tests")
    parser.add_argument("--max_val_batches", type=int, help="Limit valid validation batches for pipeline smoke tests")
    parser.add_argument("--max_test_batches", type=int, help="Limit valid test batches for pipeline smoke tests")
    parser.add_argument("--experiment_tag", type=str, help="Append tag to experiment id for smoke or side runs")
    parser.add_argument("--split_file_tag", type=str, help="Use a tagged split file/artifact identity, e.g. formal_v1")
    parser.add_argument("--ablation_id", type=str, choices=[item for item in parse_ablation_ids("A0,A1,A2,A3,A4,A5,A6,A7,A8,A9")], help="Apply one A0-A9 ablation profile")
    parser.add_argument("--candidates", type=str, help="Comma-separated backbone names or source/name pairs to run")
    parser.add_argument("--run", action="store_true", help="Actually run training; default only prints commands")
    parser.add_argument("--append", action="store_true", help="Preserve existing manifest rows and append new rows")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing manifest instead of refusing")
    parser.add_argument("--overwrite_artifacts", action="store_true", help="Allow training artifacts and per-candidate logs to be overwritten")
    parser.add_argument("--allow_legacy_split_upgrade", action="store_true", help="Pass through legacy split metadata upgrade approval")
    parser.add_argument("--resume", action="store_true", help="Resume matching checkpoints instead of forcing --fresh")
    parser.add_argument("--output", type=str, default=str(ROOT_DIR / "docs" / "backbone_screening_runs.csv"))
    parser.add_argument("--log_dir", type=str, default=str(ROOT_DIR / "docs" / "backbone_screening_logs"))
    args = parser.parse_args()
    if (
        args.experiment_tag is None
        and (args.max_train_batches is not None or args.max_val_batches is not None or args.max_test_batches is not None)
    ):
        args.experiment_tag = "smoke"

    output_path = Path(args.output)
    if output_path.exists() and not args.append and not args.overwrite:
        raise SystemExit(f"Manifest already exists: {output_path}. Use --append or --overwrite explicitly.")
    rows = read_manifest(output_path) if args.append else []
    for source, name in select_candidates(args.candidates):
        command = build_command(args, source, name)
        cfg = config_for_candidate(args, source, name)
        result_path = Path(artifact_path(str(ROOT_DIR), "final_result", cfg, args.seed, ".txt"))
        stdout_log, stderr_log = log_paths_for_candidate(args, source, name)
        print(" ".join(f'"{part}"' if " " in part else part for part in command), flush=True)

        if args.run and not args.overwrite_artifacts:
            existing_artifacts = [path for path in (result_path, stdout_log, stderr_log) if path.exists()]
            if existing_artifacts:
                formatted = ", ".join(str(path) for path in existing_artifacts)
                raise SystemExit(
                    f"Candidate artifacts already exist: {formatted}. "
                    "Use --overwrite_artifacts only after archiving or intentionally replacing them."
                )

        status = "dry-run"
        returncode = ""
        if args.run:
            stdout_log.parent.mkdir(parents=True, exist_ok=True)
            print(f"stdout log: {stdout_log}", flush=True)
            print(f"stderr log: {stderr_log}", flush=True)
            with stdout_log.open("w", encoding="utf-8", errors="replace") as stdout_f:
                with stderr_log.open("w", encoding="utf-8", errors="replace") as stderr_f:
                    completed = subprocess.run(
                        command,
                        cwd=str(ROOT_DIR),
                        stdout=stdout_f,
                        stderr=stderr_f,
                        text=True,
                        check=False,
                    )
            returncode = completed.returncode
            status = "ok" if completed.returncode == 0 else "failed"
            if completed.returncode != 0:
                rows.append({
                    "source": source,
                    "backbone": name,
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "split": args.split,
                    "split_file_tag": getattr(args, "split_file_tag", "") or "",
                    "ablation_id": args.ablation_id or "",
                    **ablation_row_flags(cfg),
                    "status": status,
                    "returncode": returncode,
                    "selected_test_mae": "",
                    "mae_raw": "",
                    "mae_flip": "",
                    "mae_multi": "",
                    "selected_tta": "",
                    "experiment_id": "",
                    "result_path": str(result_path),
                    "stdout_log": str(stdout_log),
                    "stderr_log": str(stderr_log),
                    "command": " ".join(command),
                })
                write_manifest(rows, output_path)
                break

        metrics = parse_result_metrics(result_path) if args.run else empty_result_metrics()
        if args.run and status == "ok":
            expected_experiment_id = build_training_metadata(cfg, args.seed)["experiment_id"]
            if not metrics["selected_test_mae"] or not metrics["experiment_id"]:
                status = "incomplete"
            elif metrics["experiment_id"] and metrics["experiment_id"] != expected_experiment_id:
                status = "metadata-mismatch"
        rows.append({
            "source": source,
            "backbone": name,
            "seed": args.seed,
            "epochs": args.epochs,
            "split": args.split,
            "split_file_tag": getattr(args, "split_file_tag", "") or "",
            "ablation_id": args.ablation_id or "",
            **ablation_row_flags(cfg),
            "status": status,
            "returncode": returncode,
            "selected_test_mae": metrics["selected_test_mae"],
            "mae_raw": metrics["mae_raw"],
            "mae_flip": metrics["mae_flip"],
            "mae_multi": metrics["mae_multi"],
            "selected_tta": metrics["selected_tta"],
            "experiment_id": metrics["experiment_id"],
            "result_path": str(result_path),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "command": " ".join(command),
        })
        write_manifest(rows, output_path)

    print(f"Manifest written: {args.output}")


if __name__ == "__main__":
    main()
