"""Evaluate the FaRL teacher checkpoint (EXP-030) with the same cumulative 1x-6x TTA protocol as FADE-Net."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_fade_net as train  # noqa: E402
from eval_fade_net_tta import (  # noqa: E402
    TTA_VIEW_ORDER,
    checkpoint_value,
    evaluate_view_counts,
)
from teacher_vit import build_teacher  # noqa: E402
from train_farl_teacher import get_teacher_transforms  # noqa: E402

TEACHER_IMAGE_SIZE = 224


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


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if "ema_state_dict" not in checkpoint:
        raise KeyError("Teacher checkpoint missing required state: ema_state_dict")
    saved_args = checkpoint.get("config", {})
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
        get_teacher_transforms(TEACHER_IMAGE_SIZE, is_train=False),
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

    model = build_teacher(
        weights_path=None,
        num_classes=output_max_age - output_min_age + 1,
        output_min_age=output_min_age,
    )
    model.load_state_dict(checkpoint["ema_state_dict"])
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

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

    ema_metrics, ema_seconds = evaluate_view_counts(
        model, loader, range(1, 7), TEACHER_IMAGE_SIZE, device
    )
    results["timing"] = {"ema_1x_to_6x_joint_seconds": ema_seconds}
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
