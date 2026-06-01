"""
ONNX Export Script for FADE-Net

Usage:
    python scripts/export_onnx.py --model_path best_model.pth --output model.onnx

Exports the trained model to ONNX format for deployment without PyTorch dependency.
"""

import os
import sys
import argparse
from pathlib import Path
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from config import Config
from experiment import (
    build_model_for_checkpoint_load,
    checkpoint_metadata_mismatches,
    load_model_state_package,
    build_training_metadata,
    format_metadata_mismatches,
    populate_runtime_model_metadata,
)
from ablation_profiles import apply_ablation_profile, parse_ablation_ids
from utils import remap_state_dict_keys


def apply_common_overrides(cfg, args):
    apply_ablation_profile(cfg, getattr(args, "ablation_id", None))
    if args.backbone_source is not None:
        cfg.backbone_source = args.backbone_source
    if args.backbone_name is not None:
        cfg.backbone_name = args.backbone_name
    if args.no_pretrained:
        cfg.backbone_pretrained = False
    if args.split_file_tag is not None:
        cfg.split_file_tag = args.split_file_tag


def export_onnx(model_path, output_path, cfg, dynamic_batch=True):
    """Export FADE-Net to ONNX format."""
    device = torch.device("cpu")
    state_dict, checkpoint = load_model_state_package(model_path, device)
    populate_runtime_model_metadata(cfg)
    expected_metadata = build_training_metadata(cfg, checkpoint.get("metadata", {}).get("seed", 42) if isinstance(checkpoint, dict) else 42)
    mismatches = checkpoint_metadata_mismatches(checkpoint, expected_metadata)
    if mismatches:
        raise RuntimeError(f"Checkpoint metadata mismatch; refusing to export. {format_metadata_mismatches(mismatches)}")
    model = build_model_for_checkpoint_load(cfg)
    model.load_state_dict(remap_state_dict_keys(state_dict))
    model.eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size)

    # Validate forward pass before export
    with torch.no_grad():
        ref_output = model(dummy)

    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    # Validate exported model
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(output_path))
        onnx_output = session.run(None, {"input": dummy.numpy()})
        max_diff = abs(ref_output.numpy() - onnx_output[0]).max()
        print(f"✅ ONNX export validated. Max output diff: {max_diff:.6f}")
        if max_diff > 1e-4:
            print(f"⚠️ Output diff {max_diff:.6f} > 1e-4 — check numeric precision")
    except ImportError:
        print("ℹ️ onnxruntime not installed; skipping validation. Install with: pip install onnxruntime")
    except Exception as exc:
        print(f"⚠️ ONNX runtime validation skipped after export: {type(exc).__name__}: {exc}")

    print(f"✅ Model exported to: {output_path}")
    print(f"   Input shape: [B, 3, {cfg.img_size}, {cfg.img_size}]")
    print(f"   Output shape: [B, {cfg.num_classes}]")


def main():
    parser = argparse.ArgumentParser(description="Export FADE-Net to ONNX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model .pth")
    parser.add_argument("--output", type=str, default=None, help="Output .onnx path (default: same dir)")
    parser.add_argument("--split_file_tag", type=str, default="formal_v1", help="Split file tag")
    parser.add_argument('--backbone_source', type=str, choices=['torchvision', 'timm'], help='Backbone provider')
    parser.add_argument('--backbone_name', type=str, help='Backbone model name')
    parser.add_argument('--no_pretrained', action='store_true', help='Disable pretrained backbone weights')
    parser.add_argument('--ablation_id', type=str, choices=[item for item in parse_ablation_ids("A0,A1,A2,A3,A4,A5,A6,A7,A8,A9")], help='Apply an A0-A9 ablation profile')
    parser.add_argument("--static_batch", action="store_true", help="Disable dynamic batch axis")
    args = parser.parse_args()

    cfg = Config()
    apply_common_overrides(cfg, args)

    output_path = args.output or os.path.splitext(args.model_path)[0] + ".onnx"
    export_onnx(args.model_path, output_path, cfg, dynamic_batch=not args.static_batch)


if __name__ == "__main__":
    main()
