"""Profile FADE-Net Small/Medium parameters, MACs, feature shapes and latency."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path

import torch
from thop import profile


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.fade_net import FADENet  # noqa: E402


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def build_model(backbone_name: str, device: torch.device) -> FADENet:
    config = Config()
    config.backbone_source = "timm"
    config.backbone_name = backbone_name
    config.backbone_pretrained = False
    config.backbone_weights = ""
    config.img_size = 256
    config.fusion_channels = 96
    config.route_groups = 8
    config.residual_bound = 3.0
    config.gate_error_scale = 3.0
    config.use_cgbr = True
    return FADENet(config).to(device).eval()


@torch.inference_mode()
def feature_shapes(model: FADENet, dummy: torch.Tensor) -> list[list[int]]:
    return [list(tensor.shape[1:]) for tensor in model.extract_features(dummy)]


@torch.inference_mode()
def cuda_latency(
    model: FADENet,
    dummy: torch.Tensor,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        model(dummy)
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(dummy)
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def profile_model(
    label: str,
    backbone_name: str,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    model = build_model(backbone_name, device)
    dummy = torch.randn(1, 3, 256, 256, device=device)
    shapes = feature_shapes(model, dummy)
    macs, _ = profile(model, inputs=(dummy,), verbose=False)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    result: dict[str, object] = {
        "label": label,
        "backbone": backbone_name,
        "input_shape": [1, 3, 256, 256],
        "feature_shapes_chw": shapes,
        "parameters": total,
        "trainable_parameters": trainable,
        "macs": int(macs),
    }
    if device.type == "cuda":
        result["latency_batch1_fp32"] = cuda_latency(model, dummy, warmup, iterations)
    del dummy, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "paper" / "evidence" / "efficiency_profile.json",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--cpu", action="store_true", help="Skip CUDA even if available")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    report = {
        "schema_version": 1,
        "measurement": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "batch_size": 1,
            "precision": "FP32",
            "input_size": 256,
            "warmup_iterations": args.warmup if device.type == "cuda" else None,
            "timed_iterations": args.iterations if device.type == "cuda" else None,
            "latency_timer": "torch.cuda.Event with synchronization" if device.type == "cuda" else None,
            "macs_tool": "thop.profile",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "models": {
            "small": profile_model(
                "FADE-Net-Small", "mobilenetv4_conv_small", device, args.warmup, args.iterations
            ),
            "medium": profile_model(
                "FADE-Net-Medium", "mobilenetv4_conv_medium", device, args.warmup, args.iterations
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    for key, item in report["models"].items():
        print(f"{key}: params={item['parameters']}, macs={item['macs']}, latency={item.get('latency_batch1_fp32')}")


if __name__ == "__main__":
    main()
