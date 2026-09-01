"""Ensemble TTA: probability-averaged fusion of two FADE-Net checkpoints.

Fuses main_prob distributions of two EMA checkpoints (e.g. Small + Medium),
then reports cumulative 1x-6x TTA MAE. Fusion weight is fixed up-front (Val),
never tuned on Test. This is an upper-bound/ensemble result, NOT a single-model
deployment number.
"""
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


def cv(saved, key, default):
    value = saved.get(key, default)
    return default if value is None else value


def build_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("config", {})
    cfg = Config()
    cfg.min_age = cv(saved, "output_min_age", 0)
    cfg.max_age = cv(saved, "output_max_age", 80)
    cfg.num_classes = cfg.max_age - cfg.min_age + 1
    cfg.data_min_age = cv(saved, "data_min_age", 15)
    cfg.data_max_age = cv(saved, "data_max_age", 72)
    cfg.img_size = cv(saved, "input_size", 256)
    cfg.backbone_pretrained = False
    cfg.backbone_source = cv(saved, "backbone_source", "timm")
    cfg.backbone_name = cv(saved, "backbone_name", "mobilenetv4_conv_small")
    cfg.backbone_weights = cv(saved, "backbone_weights", None)
    cfg.use_dcsr = True
    cfg.use_cgbr = cv(saved, "use_cgbr", True)
    cfg.fusion_channels = cv(saved, "fusion_channels", 96)
    cfg.route_groups = cv(saved, "route_groups", 8)
    cfg.residual_bound = cv(saved, "residual_bound", 3.0)
    cfg.gate_error_scale = cv(saved, "gate_error_scale", 3.0)
    cfg.label_sigma = cv(saved, "label_sigma", 2.0)
    with contextlib.redirect_stdout(io.StringIO()):
        model = train.FADENet(cfg).to(device)
    model.load_state_dict(ckpt["ema_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt, cfg.img_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afad_dir", required=True)
    ap.add_argument("--official_db", required=True)
    ap.add_argument("--checkpoint1", required=True)
    ap.add_argument("--checkpoint2", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--split_id", type=int, default=0)
    ap.add_argument("--subset", choices=("val", "test"), default="val")
    ap.add_argument("--weight1", type=float, default=0.5,
                    help="weight for checkpoint1; checkpoint2 gets 1-w1")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    m1, c1, sz1 = build_model_from_ckpt(args.checkpoint1, device)
    m2, c2, sz2 = build_model_from_ckpt(args.checkpoint2, device)
    assert sz1 == sz2, f"input size mismatch {sz1} vs {sz2}"
    image_size = sz1
    w1, w2 = args.weight1, 1.0 - args.weight1

    ages = m1.ages.to(device)
    saved1 = c1.get("config", {})
    data_min_age = cv(saved1, "data_min_age", 15)
    data_max_age = cv(saved1, "data_max_age", 72)

    samples, _, val_idx, test_idx, metadata = train.load_official_split(
        args.official_db, args.afad_dir, data_min_age, data_max_age,
        args.split_id, strict=True,
    )
    eval_idx = val_idx if args.subset == "val" else test_idx
    ds = train.AFADDataset(
        args.afad_dir, train.get_transforms(image_size, is_train=False),
        data_min_age, data_max_age, samples=samples,
    )
    loader = DataLoader(
        Subset(ds, eval_idx), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=train.collate_fn,
        pin_memory=device.type == "cuda",
    )

    total_mae = {k: 0.0 for k in range(1, 7)}
    total = 0
    t0 = time.time()
    with torch.inference_mode():
        for bi, (images, y, _) in enumerate(loader):
            if images.numel() == 0:
                continue
            images = images.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            views = make_ordered_views(images, image_size)
            pred_sum = torch.zeros_like(y)
            for vi, view in enumerate(views, start=1):
                p = w1 * m1(view)["main_prob"] + w2 * m2(view)["main_prob"]
                age = (p * ages).sum(dim=1)
                pred_sum = pred_sum + age
                if vi in total_mae:
                    total_mae[vi] += torch.abs(pred_sum / vi - y).sum().item()
            total += y.numel()
            if (bi + 1) % 100 == 0:
                print(f"batch {bi + 1}/{len(loader)}", flush=True)
    elapsed = time.time() - t0

    metrics = {
        f"ensemble_{k}x": {"mae": total_mae[k] / total, "samples": total, "views": k}
        for k in range(1, 7)
    }
    results = {
        "checkpoint1": str(Path(args.checkpoint1).resolve()),
        "ckpt1_epoch": c1.get("epoch"), "ckpt1_best_val": c1.get("best_val_mae"),
        "checkpoint2": str(Path(args.checkpoint2).resolve()),
        "ckpt2_epoch": c2.get("epoch"), "ckpt2_best_val": c2.get("best_val_mae"),
        "weight1": w1, "weight2": w2,
        "fusion": "main_prob weighted average -> expected age",
        "split_id": args.split_id,
        "split_fingerprint": metadata["split_fingerprint"],
        "evaluation_subset": args.subset,
        "evaluation_size": len(eval_idx),
        "image_size": image_size,
        "tta_view_order": list(TTA_VIEW_ORDER),
        "elapsed_seconds": elapsed,
        "metrics": metrics,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {k: round(v["mae"], 4) for k, v in metrics.items()}, ensure_ascii=False),
        flush=True)


if __name__ == "__main__":
    main()
