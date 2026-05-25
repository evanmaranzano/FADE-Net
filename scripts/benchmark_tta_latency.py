import argparse
import os
import sys
import time

import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from ablation_profiles import apply_ablation_profile, parse_ablation_ids
from config import Config
from evaluation import predict_probs
from model import LightweightAgeEstimator


def apply_common_overrides(cfg, args):
    apply_ablation_profile(cfg, getattr(args, "ablation_id", None))
    if args.backbone_source is not None:
        cfg.backbone_source = args.backbone_source
    if args.backbone_name is not None:
        cfg.backbone_name = args.backbone_name
    if args.no_pretrained:
        cfg.backbone_pretrained = False


def _sync_if_needed(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def measure_latency(
    *,
    model,
    images,
    base_size,
    max_augmented_batch_size,
    warmup,
    iters,
    device,
):
    with torch.no_grad():
        for _ in range(warmup):
            predict_probs(
                model,
                images,
                mode="multi",
                base_size=base_size,
                max_augmented_batch_size=max_augmented_batch_size,
            )
        _sync_if_needed(device)
        start = time.perf_counter()
        for _ in range(iters):
            predict_probs(
                model,
                images,
                mode="multi",
                base_size=base_size,
                max_augmented_batch_size=max_augmented_batch_size,
            )
        _sync_if_needed(device)
    return (time.perf_counter() - start) / iters


def compare_tta_latency(model, images, base_size, warmup, iters, device=None):
    device = device or images.device
    chunk1_seconds = measure_latency(
        model=model,
        images=images,
        base_size=base_size,
        max_augmented_batch_size=1,
        warmup=warmup,
        iters=iters,
        device=device,
    )
    batched_seconds = measure_latency(
        model=model,
        images=images,
        base_size=base_size,
        max_augmented_batch_size=None,
        warmup=warmup,
        iters=iters,
        device=device,
    )
    chunk1_ms = chunk1_seconds * 1000
    batched_ms = batched_seconds * 1000
    return {
        "chunk1_latency_ms": chunk1_ms,
        "batched_latency_ms": batched_ms,
        "latency_improvement_percent": (chunk1_ms - batched_ms) / chunk1_ms * 100,
        "speedup_x": chunk1_ms / batched_ms,
    }


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def main():
    parser = argparse.ArgumentParser(description="Benchmark batched multi-TTA latency against chunk=1 sequential views")
    parser.add_argument("--iters", type=int, default=40, help="Measured iterations")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup iterations")
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Benchmark device")
    parser.add_argument("--backbone_source", type=str, choices=["torchvision", "timm"], help="Backbone provider")
    parser.add_argument("--backbone_name", type=str, help="Backbone model name")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable pretrained backbone weights")
    parser.add_argument(
        "--ablation_id",
        type=str,
        choices=[item for item in parse_ablation_ids("A0,A1,A2,A3,A4,A5,A6,A7,A8,A9")],
        help="Apply an A0-A9 ablation profile",
    )
    args = parser.parse_args()
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    cfg = Config()
    apply_common_overrides(cfg, args)
    device = resolve_device(args.device)
    model = LightweightAgeEstimator(cfg).to(device).eval()
    images = torch.randn(args.batch_size, 3, cfg.img_size, cfg.img_size, device=device)

    result = compare_tta_latency(
        model=model,
        images=images,
        base_size=cfg.img_size,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    print(f"device={device.type}")
    print("mode=multi")
    print(f"chunk1_latency_ms={result['chunk1_latency_ms']:.3f}")
    print(f"batched_latency_ms={result['batched_latency_ms']:.3f}")
    print(f"latency_improvement_percent={result['latency_improvement_percent']:.3f}")
    print(f"speedup_x={result['speedup_x']:.3f}")


if __name__ == "__main__":
    main()
