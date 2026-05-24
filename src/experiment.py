import os
import re
from typing import Any

import torch


def populate_runtime_model_metadata(config) -> None:
    """Fill model-derived metadata fields without downloading pretrained weights."""
    from contextlib import redirect_stdout
    import io

    from model import LightweightAgeEstimator

    original_pretrained = getattr(config, "backbone_pretrained", True)
    config.backbone_pretrained = False
    try:
        with torch.no_grad(), redirect_stdout(io.StringIO()):
            model = LightweightAgeEstimator(config)
        del model
    finally:
        config.backbone_pretrained = original_pretrained


def sanitize_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    return token.strip("._-") or "unset"


def optional_sanitize_token(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return sanitize_token(value)


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def backbone_signature(config) -> dict[str, Any]:
    return {
        "source": getattr(config, "backbone_source", "torchvision"),
        "name": getattr(config, "backbone_name", "mobilenet_v3_large"),
        "pretrained": bool(getattr(config, "backbone_pretrained", True)),
        "effective_pretrained": bool(getattr(config, "backbone_pretrained", True)),
        "msff_feature_indices": _list_value(getattr(config, "msff_feature_indices", (6, 12))),
        "effective_msff_feature_indices": _list_value(
            getattr(config, "effective_msff_feature_indices", getattr(config, "msff_feature_indices", (6, 12)))
        ),
        "effective_msff_channels": _list_value(getattr(config, "effective_msff_channels", [])),
        "effective_msff_spatial": _list_value(getattr(config, "effective_msff_spatial", [])),
        "effective_deep_channels": getattr(config, "effective_deep_channels", None),
        "head_version": getattr(config, "head_version", "fade-head-v1"),
    }


def ablation_signature(config) -> dict[str, bool]:
    replaced_blocks = _list_value(getattr(config, "hybrid_attention_replaced_blocks", []))
    effective_ha = bool(replaced_blocks)
    if not hasattr(config, "hybrid_attention_replaced_blocks"):
        effective_ha = bool(getattr(config, "use_hybrid_attention", False)) and getattr(config, "backbone_source", "torchvision") != "timm"
    return {
        "use_hybrid_attention": bool(getattr(config, "use_hybrid_attention", False)),
        "effective_hybrid_attention": effective_ha,
        "hybrid_attention_replaced_blocks": replaced_blocks,
        "use_dldl_v2": bool(getattr(config, "use_dldl_v2", False)),
        "use_multi_scale": bool(getattr(config, "use_multi_scale", False)),
        "use_spp": bool(getattr(config, "use_spp", False)),
        "use_mv_loss": bool(getattr(config, "use_mv_loss", False)),
        "use_texture_branch": bool(getattr(config, "use_texture_branch", False)),
        "use_freq_attention": bool(getattr(config, "use_freq_attention", False)),
        "use_moe": bool(getattr(config, "use_moe", False)),
        "use_adaptive_triplet": bool(getattr(config, "use_adaptive_triplet", False)),
        "use_asymmetric_ordinal": bool(getattr(config, "use_asymmetric_ordinal", False)),
    }


def loss_signature(config) -> dict[str, Any]:
    return {
        "lambda_l1": getattr(config, "lambda_l1", None),
        "lambda_rank": getattr(config, "lambda_rank", None),
        "lambda_mv": getattr(config, "lambda_mv", None),
        "lambda_triplet": getattr(config, "lambda_triplet", None),
        "triplet_base_margin": getattr(config, "triplet_base_margin", None),
        "triplet_alpha": getattr(config, "triplet_alpha", None),
        "triplet_age_threshold": getattr(config, "triplet_age_threshold", None),
        "lambda_asym": getattr(config, "lambda_asym", None),
        "asym_under_weight": getattr(config, "asym_under_weight", None),
        "asym_over_weight": getattr(config, "asym_over_weight", None),
        "asym_delta": getattr(config, "asym_delta", None),
        "moe_num_experts": getattr(config, "moe_num_experts", None),
        "moe_hidden_dim": getattr(config, "moe_hidden_dim", None),
        "lambda_moe_gate": getattr(config, "lambda_moe_gate", None),
        "use_reweighting": bool(getattr(config, "use_reweighting", False)),
        "lds_sigma": getattr(config, "lds_sigma", None),
    }


def augmentation_signature(config) -> dict[str, Any]:
    return {
        "use_mixup": bool(getattr(config, "use_mixup", False)),
        "mixup_alpha": getattr(config, "mixup_alpha", None),
        "mixup_prob": getattr(config, "mixup_prob", None),
        "use_sigma_jitter": bool(getattr(config, "use_sigma_jitter", False)),
        "sigma_jitter": getattr(config, "sigma_jitter", None),
        "use_random_erasing": bool(getattr(config, "use_random_erasing", False)),
        "re_prob": getattr(config, "re_prob", None),
    }


def build_experiment_id(config, seed: int) -> str:
    backbone = backbone_signature(config)
    weights_tag = "pretrained" if backbone["pretrained"] else "scratch"
    split_file_tag = optional_sanitize_token(getattr(config, "split_file_tag", None))
    parts = [
        getattr(config, "project_name", "FADE-Net"),
        backbone["source"],
        backbone["name"],
        weights_tag,
        getattr(config, "split_protocol", "split"),
    ]
    if split_file_tag:
        parts.append(f"splitfile-{split_file_tag}")
    parts.append(f"seed{seed}")
    experiment_tag = getattr(config, "experiment_tag", None)
    if experiment_tag:
        parts.append(experiment_tag)
    return "_".join(sanitize_token(part) for part in parts)


def build_training_metadata(config, seed: int, split_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    split_metadata = split_metadata or getattr(config, "split_metadata", {}) or {}
    return {
        "experiment_id": build_experiment_id(config, seed),
        "experiment_tag": getattr(config, "experiment_tag", None),
        "split_file_tag": optional_sanitize_token(getattr(config, "split_file_tag", None)),
        "seed": seed,
        "project_name": getattr(config, "project_name", None),
        "split_protocol": getattr(config, "split_protocol", None),
        "split_file": split_metadata.get("split_file"),
        "split_fingerprint": split_metadata.get("fingerprint"),
        "dataset_fingerprint": split_metadata.get("dataset_fingerprint"),
        "legacy_split_upgraded": bool(split_metadata.get("legacy_upgraded", False)),
        "img_size": getattr(config, "img_size", None),
        "num_classes": getattr(config, "num_classes", None),
        "min_age": getattr(config, "min_age", None),
        "max_age": getattr(config, "max_age", None),
        "backbone": backbone_signature(config),
        "ablations": ablation_signature(config),
        "loss": loss_signature(config),
        "augmentation": augmentation_signature(config),
        "reported_tta_modes": ["raw", "flip", "multi"],
        "selection_metric": {
            "split": "val",
            "metric": "MAE",
            "tta": "multi",
        },
        "validation_tta": "multi",
        "test_tta": "multi",
    }


def artifact_path(root_dir: str, kind: str, config, seed: int, extension: str) -> str:
    experiment_id = build_experiment_id(config, seed)
    return os.path.join(root_dir, f"{sanitize_token(kind)}_{experiment_id}{extension}")


def _intersection_dict_eq(a: Any, b: Any) -> bool:
    """Compare two dicts by shared keys only (tolerates new keys added in either direction)."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return a == b
    shared_keys = set(a) & set(b)
    return all(a[k] == b[k] for k in shared_keys)


def checkpoint_metadata_mismatches(
    checkpoint: dict[str, Any],
    expected_metadata: dict[str, Any],
    keys: tuple[str, ...] | None = None,
) -> list[tuple[str, Any, Any]]:
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    keys = keys or (
        "experiment_id",
        "experiment_tag",
        "split_file_tag",
        "project_name",
        "split_protocol",
        "split_file",
        "split_fingerprint",
        "dataset_fingerprint",
        "legacy_split_upgraded",
        "img_size",
        "num_classes",
        "min_age",
        "max_age",
        "backbone",
        "ablations",
        "loss",
        "reported_tta_modes",
        "selection_metric",
    )
    mismatches = []
    for key in keys:
        actual = metadata.get(key)
        expected = expected_metadata.get(key)
        if key in ("backbone", "ablations", "loss"):
            if not _intersection_dict_eq(actual, expected):
                mismatches.append((key, actual, expected))
        elif actual != expected:
            mismatches.append((key, actual, expected))
    return mismatches


def save_model_package(model, path: str, metadata: dict[str, Any]) -> None:
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)


def load_model_state_package(path: str, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint
    return checkpoint, {"metadata": {}}


def format_metadata_mismatches(mismatches) -> str:
    return ", ".join([f"{key}: checkpoint={old!r}, current={new!r}" for key, old, new in mismatches])


def is_compatible_checkpoint(path: str, config, seed: int, device="cpu"):
    try:
        _state_dict, checkpoint = load_model_state_package(path, device)
    except Exception as exc:
        return False, f"load failed: {exc}"
    expected_metadata = build_training_metadata(config, seed)
    mismatches = checkpoint_metadata_mismatches(checkpoint, expected_metadata)
    if mismatches:
        return False, format_metadata_mismatches(mismatches)
    return True, ""


def compatible_best_model_paths(root_dir: str, config, seed: int = 42, device="cpu"):
    exact_path = artifact_path(root_dir, "best_model", config, seed, ".pth")
    candidates = []
    if os.path.exists(exact_path):
        candidates.append(exact_path)
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if path == exact_path:
            continue
        if name.startswith("best_model_") and name.endswith(".pth") and os.path.isfile(path):
            candidates.append(path)

    compatible = []
    incompatible = []
    for path in candidates:
        ok, reason = is_compatible_checkpoint(path, config, seed, device=device)
        if ok:
            compatible.append(path)
        else:
            incompatible.append((path, reason))
    return compatible, incompatible
