"""Audit and summarize the final five-fold FADE-Net experiment artifacts.

The script intentionally reads only archived JSON/status/checkpoint metadata.  It
does not load model weights or touch training outputs.  Test-time augmentation
is selected independently for every fold from the symmetric validation
candidates {2, 4, 6}; the test set is never used for that choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "server_recovery" / "2026-07-17" / "host_restore_20260718"
LEGACY_OUTPUTS = (
    ROOT
    / "server_recovery"
    / "2026-07-17"
    / "files"
    / "data"
    / "outputs"
)
EXPECTED_FINGERPRINT = (
    "8813b83131df5e09ccfeb9d513abaa72906da9f816e500dabe7a69e95f086375"
)
SYMMETRIC_TTA_VIEWS = (2, 4, 6)

MODEL_RUNS = {
    "small": ("exp031_final", "exp045_final", "exp047_final", "exp049_final", "exp051_final"),
    "medium": ("exp033_final", "exp044_final", "exp046_final", "exp048_final", "exp050_final"),
}
TEACHER_RUNS = ("exp030_final", "exp040_final", "exp041_final", "exp042_final", "exp043_final")
ENSEMBLE_RUNS = ("ensemble_final", "ensemble_fold1", "ensemble_fold2", "ensemble_fold3", "ensemble_fold4")

EXPECTED_SPLITS = {
    0: {"train": [0, 1, 2, 3, 4, 5], "val": [6, 7], "test": [8, 9]},
    1: {"train": [2, 3, 4, 5, 6, 7], "val": [8, 9], "test": [0, 1]},
    2: {"train": [4, 5, 6, 7, 8, 9], "val": [0, 1], "test": [2, 3]},
    3: {"train": [5, 6, 7, 8, 9, 0], "val": [1, 2], "test": [3, 4]},
    4: {"train": [6, 7, 8, 9, 0, 1], "val": [2, 3], "test": [4, 5]},
}
EXPECTED_TEST_SIZES = (33161, 33067, 33085, 33182, 33062)
EXPECTED_BACKBONES = {
    "small": "mobilenetv4_conv_small",
    "medium": "mobilenetv4_conv_medium",
}
EXPECTED_CONFIG = {
    "input_size": 256,
    "fusion_channels": 96,
    "route_groups": 8,
    "residual_bound": 3.0,
    "label_sigma": 2.0,
    "epochs": 55,
    "batch_size": 64,
    "backbone_lr": 3e-5,
    "head_lr": 3e-4,
    "weight_decay": 5e-4,
    "warmup_epochs": 5,
    "early_stopping_patience": 20,
    "train_crop_scale_min": 0.7,
    "gradient_clip": 5.0,
    "cgbr_start_epoch": 16,
    "cgbr_full_epoch": 26,
    "lambda_coarse": 0.3,
    "lambda_refine": 0.5,
    "lambda_gate": 0.1,
    "lambda_kd": 1.0,
    "use_ema": True,
    "ema_decay": 0.999,
    "seed": 42,
    "skip_final_test": True,
}
NULL_TEST_FIELDS = (
    "test_mae",
    "test_base_mae",
    "evaluation_model",
    "raw_test_mae",
    "raw_test_base_mae",
    "ema_test_mae",
    "ema_test_base_mae",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def metric_value(payload: dict[str, Any], prefix: str, views: int) -> float:
    return float(payload["metrics"][f"{prefix}_{views}x"]["mae"])


def tta_paths(run_dir: Path, fold: int) -> tuple[Path, Path]:
    if fold == 0:
        return run_dir / "val_tta_1to6.json", run_dir / "test_tta_1to6.json"
    return run_dir / "val_tta.json", run_dir / "test_tta.json"


def check_status_if_present(run_dir: Path) -> dict[str, Any] | None:
    statuses = list(run_dir.glob("*.status"))
    if not statuses:
        return None
    require(len(statuses) == 1, f"ambiguous status files in {run_dir}")
    values: dict[str, str] = {}
    for line in statuses[0].read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            key, value = line.split("\t", 1)
            values[key] = value
    require(values.get("exit_code") == "0", f"non-zero status in {statuses[0]}")
    return {"path": rel(statuses[0]), "exit_code": 0}


def audit_model(model_name: str, run_names: tuple[str, ...]) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for fold, run_name in enumerate(run_names):
        run_dir = ARCHIVE / run_name
        result_path = run_dir / "results.json"
        checkpoint_path = run_dir / "best_checkpoint.pth"
        require(result_path.is_file(), f"missing {result_path}")
        require(checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0, f"missing checkpoint in {run_dir}")

        result = load_json(result_path)
        require(result["fold"] == fold, f"fold mismatch in {result_path}")
        require(result["split"]["split_id"] == fold, f"split id mismatch in {result_path}")
        require(result["split"]["split_fingerprint"] == EXPECTED_FINGERPRINT, f"fingerprint mismatch in {result_path}")
        expected_split = EXPECTED_SPLITS[fold]
        require(result["split"]["train_folders"] == expected_split["train"], f"train folders mismatch in {result_path}")
        require(result["split"]["val_folders"] == expected_split["val"], f"val folders mismatch in {result_path}")
        require(result["split"]["test_folders"] == expected_split["test"], f"test folders mismatch in {result_path}")
        require(result["split"]["age_range"] == [15, 72], f"age range mismatch in {result_path}")
        require(result["split"]["output_range"] == [0, 80], f"output range mismatch in {result_path}")
        require(result["split"]["missing_entries_in_age_range"] == 0, f"missing AFAD files in {result_path}")
        for key in NULL_TEST_FIELDS:
            require(result.get(key) is None, f"training-time Test field {key} is populated in {result_path}")

        config = result["config"]
        for key, expected in EXPECTED_CONFIG.items():
            require(config.get(key) == expected, f"config {key} mismatch in {result_path}: {config.get(key)!r}")
        backbone = config.get("backbone_name", EXPECTED_BACKBONES[model_name])
        require(backbone == EXPECTED_BACKBONES[model_name], f"backbone mismatch in {result_path}")
        require(config.get("use_cgbr") is True, f"CGBR disabled in {result_path}")
        require(bool(config.get("teacher_checkpoint")), f"missing fold-specific teacher in {result_path}")

        val_path, test_path = tta_paths(run_dir, fold)
        require(val_path.is_file() and test_path.is_file(), f"missing TTA JSON in {run_dir}")
        val = load_json(val_path)
        test = load_json(test_path)
        for payload, subset, path in ((val, "val", val_path), (test, "test", test_path)):
            require(payload["split_id"] == fold, f"split id mismatch in {path}")
            require(payload["split_fingerprint"] == EXPECTED_FINGERPRINT, f"fingerprint mismatch in {path}")
            require(payload["evaluation_subset"] == subset, f"subset mismatch in {path}")
        require(test["evaluation_size"] == EXPECTED_TEST_SIZES[fold], f"test size mismatch in {test_path}")
        require(val["tta_view_order"] == test["tta_view_order"], f"view order mismatch in {run_dir}")

        selected_views = min(SYMMETRIC_TTA_VIEWS, key=lambda n: metric_value(val, "ema", n))
        folds.append(
            {
                "fold": fold,
                "run": run_name,
                "best_epoch": int(result["best_epoch"]),
                "best_val_mae": float(result["best_val_mae"]),
                "test_size": int(test["evaluation_size"]),
                "test_1x_mae": metric_value(test, "ema", 1),
                "selected_tta_views": selected_views,
                "selected_val_mae": metric_value(val, "ema", selected_views),
                "selected_test_mae": metric_value(test, "ema", selected_views),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "status": check_status_if_present(run_dir),
                "sources": {
                    "results": rel(result_path),
                    "results_sha256": sha256(result_path),
                    "val_tta": rel(val_path),
                    "val_tta_sha256": sha256(val_path),
                    "test_tta": rel(test_path),
                    "test_tta_sha256": sha256(test_path),
                    "checkpoint": rel(checkpoint_path),
                },
            }
        )
    return summarize_folds(model_name, folds)


def audit_ensemble() -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for fold, run_name in enumerate(ENSEMBLE_RUNS):
        run_dir = ARCHIVE / run_name
        if fold == 0:
            val_path = run_dir / "ensemble_small_medium_val_w05.json"
            test_path = run_dir / "ensemble_small_medium_test_w05.json"
        else:
            val_path = run_dir / "val_tta_w05.json"
            test_path = run_dir / "test_tta_w05.json"
        require(val_path.is_file() and test_path.is_file(), f"missing ensemble JSON in {run_dir}")
        val = load_json(val_path)
        test = load_json(test_path)
        for payload, subset, path in ((val, "val", val_path), (test, "test", test_path)):
            require(payload["split_id"] == fold, f"split id mismatch in {path}")
            require(payload["split_fingerprint"] == EXPECTED_FINGERPRINT, f"fingerprint mismatch in {path}")
            require(payload["evaluation_subset"] == subset, f"subset mismatch in {path}")
            require(float(payload["weight1"]) == 0.5 and float(payload["weight2"]) == 0.5, f"weights mismatch in {path}")
        require(test["evaluation_size"] == EXPECTED_TEST_SIZES[fold], f"test size mismatch in {test_path}")
        require(val["tta_view_order"] == test["tta_view_order"], f"view order mismatch in {run_dir}")
        selected_views = min(SYMMETRIC_TTA_VIEWS, key=lambda n: metric_value(val, "ensemble", n))
        folds.append(
            {
                "fold": fold,
                "run": run_name,
                "test_size": int(test["evaluation_size"]),
                "test_1x_mae": metric_value(test, "ensemble", 1),
                "selected_tta_views": selected_views,
                "selected_val_mae": metric_value(val, "ensemble", selected_views),
                "selected_test_mae": metric_value(test, "ensemble", selected_views),
                "sources": {
                    "val_tta": rel(val_path),
                    "val_tta_sha256": sha256(val_path),
                    "test_tta": rel(test_path),
                    "test_tta_sha256": sha256(test_path),
                },
            }
        )
    return summarize_folds("ensemble", folds)


def summarize_folds(name: str, folds: list[dict[str, Any]]) -> dict[str, Any]:
    one_view = [fold["test_1x_mae"] for fold in folds]
    selected = [fold["selected_test_mae"] for fold in folds]
    return {
        "name": name,
        "folds": folds,
        "test_1x": {"mean": statistics.fmean(one_view), "population_std": statistics.pstdev(one_view)},
        "val_selected_tta": {"mean": statistics.fmean(selected), "population_std": statistics.pstdev(selected)},
    }


def audit_teachers() -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for fold, run_name in enumerate(TEACHER_RUNS):
        run_dir = ARCHIVE / run_name
        result_path = run_dir / "results.json"
        checkpoint_path = run_dir / "best_checkpoint.pth"
        require(result_path.is_file(), f"missing {result_path}")
        require(checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0, f"missing checkpoint in {run_dir}")
        result = load_json(result_path)
        require(result["fold"] == fold, f"teacher fold mismatch in {result_path}")
        require(result["split"]["split_fingerprint"] == EXPECTED_FINGERPRINT, f"teacher fingerprint mismatch in {result_path}")
        for key in NULL_TEST_FIELDS:
            require(result.get(key) is None, f"teacher Test field {key} is populated in {result_path}")
        folds.append(
            {
                "fold": fold,
                "run": run_name,
                "best_epoch": int(result["best_epoch"]),
                "best_val_mae": float(result["best_val_mae"]),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "status": check_status_if_present(run_dir),
                "sources": {
                    "results": rel(result_path),
                    "results_sha256": sha256(result_path),
                    "checkpoint": rel(checkpoint_path),
                },
            }
        )
    values = [fold["best_val_mae"] for fold in folds]
    return {
        "name": "farl_teacher",
        "folds": folds,
        "validation": {"mean": statistics.fmean(values), "population_std": statistics.pstdev(values)},
    }


def load_ablation(name: str, path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing ablation result {path}")
    payload = load_json(path)
    require(payload["fold"] == 0, f"ablation is not Fold0: {path}")
    require(payload["split"]["split_fingerprint"] == EXPECTED_FINGERPRINT, f"ablation fingerprint mismatch: {path}")
    return {
        "name": name,
        "best_val_mae": float(payload["best_val_mae"]),
        "best_epoch": int(payload["best_epoch"]),
        "source": rel(path),
        "source_sha256": sha256(path),
    }


def audit_ablations() -> list[dict[str, Any]]:
    paths = {
        "基础配置（含CGBR）": LEGACY_OUTPUTS / "fade_net_exp003_0_80_encoderfix_gpu3" / "fold0" / "results.json",
        "关闭CGBR": LEGACY_OUTPUTS / "fade_net_exp004_0_80_cgbr_ablation_gpu3" / "fold0" / "results.json",
        "Small，55轮，无知识蒸馏": ARCHIVE / "exp029_final" / "results.json",
        "Small，55轮，FaRL知识蒸馏": ARCHIVE / "exp031_final" / "results.json",
        "Small，FaRL知识蒸馏，关闭CGBR": ARCHIVE / "exp037_final" / "results.json",
        "Medium，FaRL知识蒸馏，启用CGBR": ARCHIVE / "exp033_final" / "results.json",
        "Medium，FaRL知识蒸馏，关闭CGBR": ARCHIVE / "exp036_final" / "results.json",
        "路由分组4": ARCHIVE / "exp024_final" / "fold0" / "results.json",
        "路由分组8": LEGACY_OUTPUTS / "fade_net_exp003_0_80_encoderfix_gpu3" / "fold0" / "results.json",
        "路由分组16": ARCHIVE / "exp023_final" / "results.json",
    }
    return [load_ablation(name, path) for name, path in paths.items()]


def format_mean_std(summary: dict[str, float]) -> str:
    return f"{summary['mean']:.4f}±{summary['population_std']:.4f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# FADE-Net 五折实验核验",
        "",
        "> 本文件由 `scripts/summarize_fivefold_results.py` 从本地回传 JSON 自动生成。",
        "> 所有均值和标准差均按五折总体标准差（`ddof=0`）重算；正式 TTA 视图数仅在各折验证集的对称候选 2×/4×/6× 中选择。",
        "",
        f"- 数据划分指纹：`{report['split_fingerprint']}`",
        f"- 生成时间：{report['generated_at_note']}",
        "- 训练期 Test 字段：全部为空；Test 只来自训练结束后的独立 TTA JSON。",
        "- 五折测试样本数：" + "、".join(str(value) for value in EXPECTED_TEST_SIZES),
        "",
        "## 五折主结果",
        "",
        "| 方案 | Fold0 | Fold1 | Fold2 | Fold3 | Fold4 | 五折均值±标准差 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("small", "Small，EMA 1×"), ("medium", "Medium，EMA 1×"), ("ensemble", "Small+Medium 等权融合，1×")):
        item = report["results"][key]
        values = [fold["test_1x_mae"] for fold in item["folds"]]
        lines.append("| " + label + " | " + " | ".join(f"{value:.4f}" for value in values) + f" | {format_mean_std(item['test_1x'])} |")
    for key, label in (("small", "Small，Val预选TTA"), ("medium", "Medium，Val预选TTA"), ("ensemble", "Small+Medium 等权融合，Val预选TTA")):
        item = report["results"][key]
        values = [fold["selected_test_mae"] for fold in item["folds"]]
        lines.append("| " + label + " | " + " | ".join(f"{value:.4f}" for value in values) + f" | {format_mean_std(item['val_selected_tta'])} |")

    lines.extend(
        [
            "",
            "## TTA 预选结果",
            "",
            "| 方案 | Fold0 | Fold1 | Fold2 | Fold3 | Fold4 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in (("small", "Small"), ("medium", "Medium"), ("ensemble", "等权融合")):
        folds = report["results"][key]["folds"]
        lines.append("| " + label + " | " + " | ".join(f"{fold['selected_tta_views']}×" for fold in folds) + " |")

    lines.extend(
        [
            "",
            "## Fold0 验证集消融证据",
            "",
            "> 下表均为官方 Fold0、seed=42 的 EMA 1× 验证 MAE，仅用于局部结构和超参数判断，不替代五折主结果。",
            "",
            "| 配置 | 最佳验证 MAE | 最佳轮次 |",
            "|---|---:|---:|",
        ]
    )
    for row in report["ablations"]:
        lines.append(f"| {row['name']} | {row['best_val_mae']:.4f} | {row['best_epoch']} |")

    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- Small 与 Medium 均完成五折；融合结果由同折两个学生模型的主分布等权平均得到，不能表述为单模型性能。",
            "- CGBR 有直接开关消融；DCSR 尚无完全关闭的同配置实验，路由分组数扫描只能支持 8 组的局部选择，不能替代 DCSR-on/off 因果消融。",
            "- 参数量、MACs 和本机时延不由本脚本重算，论文中分别标注其测量环境与工具。",
            "",
            "## 机器可读产物",
            "",
            "完整源文件相对路径、SHA-256、checkpoint 大小和逐折数值见同目录 `fivefold_summary.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "paper" / "evidence",
        help="Directory for fivefold_summary.json and fivefold_summary.md",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_note": "deterministic local audit; rerun date intentionally omitted",
        "split_fingerprint": EXPECTED_FINGERPRINT,
        "standard_deviation": "population (ddof=0)",
        "tta_selection": {
            "candidate_views": list(SYMMETRIC_TTA_VIEWS),
            "selection_subset": "validation",
            "rule": "minimum validation MAE independently within each fold",
        },
        "results": {
            "small": audit_model("small", MODEL_RUNS["small"]),
            "medium": audit_model("medium", MODEL_RUNS["medium"]),
            "ensemble": audit_ensemble(),
        },
        "teachers": audit_teachers(),
        "ablations": audit_ablations(),
    }

    expected_views = {
        "small": [2, 4, 4, 2, 4],
        "medium": [2, 4, 2, 4, 6],
        "ensemble": [2, 2, 2, 2, 2],
    }
    for key, expected in expected_views.items():
        actual = [fold["selected_tta_views"] for fold in report["results"][key]["folds"]]
        require(actual == expected, f"unexpected {key} TTA selection: {actual}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "fivefold_summary.json"
    md_path = args.output_dir / "fivefold_summary.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    for key in ("small", "medium", "ensemble"):
        item = report["results"][key]
        print(f"{key}: 1x={format_mean_std(item['test_1x'])}; TTA={format_mean_std(item['val_selected_tta'])}")


if __name__ == "__main__":
    main()
