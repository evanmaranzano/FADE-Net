"""Evaluate a FADE-Net checkpoint with reproducible cumulative 1x-6x TTA."""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import train_fade_net as train  # noqa: E402
from config import Config  # noqa: E402


TTA_VIEW_ORDER = (
    {"scale": 1.0, "flip": False},
    {"scale": 1.0, "flip": True},
    {"scale": 0.9, "flip": False},
    {"scale": 1.1, "flip": False},
    {"scale": 0.9, "flip": True},
    {"scale": 1.1, "flip": True},
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--afad_dir", required=True)
    parser.add_argument("--official_db", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--subset", choices=("val", "test"), default="test")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def scaled_view(images, scale, image_size):
    if scale == 1.0:
        return images
    new_size = int(image_size * scale)
    resized = F.interpolate(
        images, size=(new_size, new_size), mode="bilinear", align_corners=False
    )
    if new_size > image_size:
        start = (new_size - image_size) // 2
        return resized[:, :, start:start + image_size, start:start + image_size]
    pad = (image_size - new_size) // 2
    return F.pad(
        resized,
        (pad, image_size - new_size - pad, pad, image_size - new_size - pad),
        mode="reflect",
    )


def make_ordered_views(images, image_size):
    views = []
    for spec in TTA_VIEW_ORDER:
        view = scaled_view(images, spec["scale"], image_size)
        if spec["flip"]:
            view = torch.flip(view, dims=[3])
        views.append(view)
    return views


def evaluate_view_counts(model, loader, view_counts, image_size, device):
    model.eval()
    view_counts = tuple(sorted(set(view_counts)))
    if not view_counts or view_counts[0] < 1 or view_counts[-1] > len(TTA_VIEW_ORDER):
        raise ValueError("view_counts must be between 1 and 6")
    total_mae = {count: 0.0 for count in view_counts}
    total_base_mae = {count: 0.0 for count in view_counts}
    total_samples = 0
    started = time.time()
    with torch.inference_mode():
        for batch_idx, (images, ages, _) in enumerate(loader):
            if images.numel() == 0:
                continue
            images = images.to(device, non_blocking=True)
            ages = ages.to(device, non_blocking=True)
            pred_sum = torch.zeros_like(ages)
            base_sum = torch.zeros_like(ages)
            views = make_ordered_views(images, image_size)
            for view_index, view in enumerate(views[:view_counts[-1]], start=1):
                outputs = model(view)
                pred_sum += outputs["age"]
                base_sum += outputs["base_age"]
                if view_index in total_mae:
                    total_mae[view_index] += torch.abs(
                        pred_sum / view_index - ages
                    ).sum().item()
                    total_base_mae[view_index] += torch.abs(
                        base_sum / view_index - ages
                    ).sum().item()
            total_samples += ages.numel()
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"up_to_{view_counts[-1]}x: batch {batch_idx + 1}/{len(loader)}",
                    flush=True,
                )
    elapsed = time.time() - started
    return {
        count: {
            "mae": total_mae[count] / total_samples,
            "base_mae": total_base_mae[count] / total_samples,
            "samples": total_samples,
            "views": count,
        }
        for count in view_counts
    }, elapsed


def checkpoint_value(saved_args, key, default):
    value = saved_args.get(key, default)
    return default if value is None else value


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    saved_args = checkpoint.get("config", {})
    image_size = checkpoint_value(saved_args, "input_size", 256)
    data_min_age = checkpoint_value(saved_args, "data_min_age", 15)
    data_max_age = checkpoint_value(saved_args, "data_max_age", 72)
    output_min_age = checkpoint_value(saved_args, "output_min_age", 0)
    output_max_age = checkpoint_value(saved_args, "output_max_age", 80)

    samples, _, val_idx, test_idx, metadata = train.load_official_split(
        args.official_db,
        args.afad_dir,
        data_min_age,
        data_max_age,
        args.split_id,
        strict=True,
    )
    eval_idx = val_idx if args.subset == "val" else test_idx
    eval_base = train.AFADDataset(
        args.afad_dir,
        train.get_transforms(image_size, is_train=False),
        data_min_age,
        data_max_age,
        samples=samples,
    )
    loader = DataLoader(
        Subset(eval_base, eval_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=train.collate_fn,
        pin_memory=device.type == "cuda",
    )

    config = Config()
    config.min_age = output_min_age
    config.max_age = output_max_age
    config.num_classes = output_max_age - output_min_age + 1
    config.data_min_age = data_min_age
    config.data_max_age = data_max_age
    config.img_size = image_size
    config.backbone_pretrained = False
    config.use_dcsr = True
    config.use_cgbr = checkpoint_value(saved_args, "use_cgbr", True)
    config.fusion_channels = checkpoint_value(saved_args, "fusion_channels", 96)
    config.route_groups = checkpoint_value(saved_args, "route_groups", 8)
    config.residual_bound = checkpoint_value(saved_args, "residual_bound", 3.0)
    config.gate_error_scale = checkpoint_value(saved_args, "gate_error_scale", 3.0)
    config.label_sigma = checkpoint_value(saved_args, "label_sigma", 2.0)
    config.backbone_source = checkpoint_value(saved_args, "backbone_source", "timm")
    config.backbone_name = checkpoint_value(saved_args, "backbone_name", "mobilenetv4_conv_small")
    config.backbone_weights = checkpoint_value(saved_args, "backbone_weights", None)

    with contextlib.redirect_stdout(io.StringIO()):
        model = train.FADENet(config).to(device)

    results = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_val_mae": checkpoint.get("best_val_mae"),
        "split_id": args.split_id,
        "split_fingerprint": metadata["split_fingerprint"],
        "evaluation_subset": args.subset,
        "evaluation_size": len(eval_idx),
        "tta_aggregation": "equal mean of final age and base_age scalar outputs",
        "tta_view_order": list(TTA_VIEW_ORDER),
        "selection_policy": (
            "Choose a future official view count using Val 1x-6x only, freeze it, "
            "then report Test; Test 1x-6x remain diagnostic while 6x is official"
        ),
        "metrics": {},
    }

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    raw_metrics, raw_seconds = evaluate_view_counts(
        model, loader, (1,), image_size, device
    )
    results["metrics"]["raw_1x"] = raw_metrics[1]
    results["timing"] = {"raw_1x_seconds": raw_seconds}
    print(json.dumps({"raw_1x": raw_metrics[1]}, ensure_ascii=False), flush=True)

    if "ema_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing required state: ema_state_dict")
    model.load_state_dict(checkpoint["ema_state_dict"])
    model.to(device)
    ema_metrics, ema_seconds = evaluate_view_counts(
        model, loader, range(1, 7), image_size, device
    )
    results["timing"]["ema_1x_to_6x_joint_seconds"] = ema_seconds
    for count in range(1, 7):
        result_name = f"ema_{count}x"
        results["metrics"][result_name] = ema_metrics[count]
        print(
            json.dumps({result_name: ema_metrics[count]}, ensure_ascii=False),
            flush=True,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
