"""
SWA (Stochastic Weight Averaging) Model Generator

使用方法:
    python scripts/swa_average.py [--seed SEED] [--eval]

功能:
    1. 平均最后 10 个 epoch 的 checkpoint 生成 SWA 模型
    2. 可选：直接评估 SWA 模型性能
"""

import os
import sys
import argparse
import torch

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from config import Config
from ablation_profiles import apply_ablation_profile, parse_ablation_ids
from model import LightweightAgeEstimator
from dataset import get_dataloaders
from experiment import (
    artifact_path,
    build_training_metadata,
    checkpoint_metadata_mismatches,
    load_model_state_package,
    populate_runtime_model_metadata,
)
from evaluation import TTA_MODES, evaluate_mae
from utils import remap_state_dict_keys


DEFAULT_SWA_SEEDS = (42, 3407, 2026, 1337)


def average_checkpoints(checkpoint_paths, device='cpu'):
    """
    Average model weights from multiple checkpoints.
    """
    print(f"📊 Averaging {len(checkpoint_paths)} checkpoints...")
    
    avg_state = None
    metadata = None
    n = len(checkpoint_paths)
    
    for i, path in enumerate(checkpoint_paths):
        print(f"   Loading [{i+1}/{n}]: {os.path.basename(path)}")
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
            raise RuntimeError(f"Checkpoint is not a packaged training checkpoint: {path}")
        if metadata is None:
            metadata = checkpoint.get("metadata", {})
        elif checkpoint.get("metadata", {}) != metadata:
            raise RuntimeError(f"Checkpoint metadata mismatch; refusing to average: {path}")
        state_dict = checkpoint['model_state_dict']
        
        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in state_dict.items()}
        else:
            for k in avg_state.keys():
                avg_state[k] += state_dict[k].float()
    
    # Divide by number of checkpoints
    for k in avg_state.keys():
        avg_state[k] /= n
    
    return avg_state, metadata


def discover_checkpoint_seeds(root_dir, cfg, epoch=111, candidate_seeds=DEFAULT_SWA_SEEDS):
    seeds = []
    for seed in candidate_seeds:
        if os.path.exists(artifact_path(root_dir, f'checkpoint_epoch_{epoch}', cfg, seed, '.pth')):
            seeds.append(seed)
    return seeds


def apply_common_overrides(cfg, args):
    apply_ablation_profile(cfg, getattr(args, "ablation_id", None))
    if args.split is not None:
        cfg.split_protocol = args.split
    if args.backbone_source is not None:
        cfg.backbone_source = args.backbone_source
    if args.backbone_name is not None:
        cfg.backbone_name = args.backbone_name
    if args.no_pretrained:
        cfg.backbone_pretrained = False
    if args.afad_dir is not None:
        cfg.afad_dir = os.path.abspath(args.afad_dir)
    if args.experiment_tag is not None:
        cfg.experiment_tag = args.experiment_tag
    if getattr(args, "split_file_tag", None) is not None:
        cfg.split_file_tag = args.split_file_tag
    if args.allow_legacy_split_upgrade:
        cfg.allow_legacy_split_upgrade = True


def format_metadata_mismatches(mismatches):
    return ", ".join([f"{key}: checkpoint={old!r}, current={new!r}" for key, old, new in mismatches])


def main():
    parser = argparse.ArgumentParser(description="SWA Model Generator")
    parser.add_argument('--seed', type=int, default=None, help='Seed to process (default: all available)')
    parser.add_argument('--eval', action='store_true', help='Evaluate SWA model after generation')
    parser.add_argument('--epochs', type=str, default='111-120', help='Epoch range to average (default: 111-120)')
    parser.add_argument('--split', type=str, choices=['80-10-10', '90-5-5', '72-8-20'], help='Split protocol')
    parser.add_argument('--backbone_source', type=str, choices=['torchvision', 'timm'], help='Backbone provider')
    parser.add_argument('--backbone_name', type=str, help='Backbone model name')
    parser.add_argument('--no_pretrained', action='store_true', help='Disable pretrained backbone weights')
    parser.add_argument('--afad_dir', type=str, help='Override AFAD dataset directory')
    parser.add_argument('--experiment_tag', type=str, help='Append tag to experiment id for smoke or side runs')
    parser.add_argument('--split_file_tag', type=str, help='Use tagged split file/artifact identity')
    parser.add_argument('--allow_legacy_split_upgrade', action='store_true', help='Trust and stamp a legacy split after size/index validation')
    parser.add_argument('--ablation_id', type=str, choices=[item for item in parse_ablation_ids("A0,A1,A2,A3,A4,A5,A6,A7,A8,A9")], help='Apply an A0-A9 ablation profile')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite an existing SWA artifact')
    args = parser.parse_args()
    cfg = Config()
    apply_common_overrides(cfg, args)
    populate_runtime_model_metadata(cfg)
    _, _, test_loader, _ = get_dataloaders(cfg)
    
    # Parse epoch range
    start_epoch, end_epoch = map(int, args.epochs.split('-'))

    # Determine seeds to process
    if args.seed:
        seeds = [args.seed]
    else:
        # Auto-detect available seeds
        seeds = discover_checkpoint_seeds(ROOT_DIR, cfg, epoch=start_epoch)
        print(f"🔍 Auto-detected seeds: {seeds}")
    
    if not seeds:
        print("❌ No checkpoint files found!")
        return
    
    results = {}
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"🌱 Processing Seed {seed}")
        print(f"{'='*60}")
        
        # Collect checkpoint paths
        checkpoint_paths = []
        for epoch in range(start_epoch, end_epoch + 1):
            path = artifact_path(ROOT_DIR, f'checkpoint_epoch_{epoch}', cfg, seed, '.pth')
            if os.path.exists(path):
                checkpoint_paths.append(path)
            else:
                print(f"⚠️ Missing: {path}")
        
        if len(checkpoint_paths) < 5:
            print(f"⚠️ Not enough checkpoints for Seed {seed} (found {len(checkpoint_paths)})")
            continue
        
        # Average checkpoints
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        avg_state, metadata = average_checkpoints(checkpoint_paths, device)
        expected_metadata = build_training_metadata(cfg, seed)
        mismatches = checkpoint_metadata_mismatches({"metadata": metadata}, expected_metadata)
        if mismatches:
            raise RuntimeError(f"SWA checkpoint metadata mismatch; refusing to average current protocol. {format_metadata_mismatches(mismatches)}")
        metadata = dict(metadata)
        metadata["swa_requested_epochs"] = list(range(start_epoch, end_epoch + 1))
        metadata["swa_actual_epochs"] = [
            int(os.path.basename(path).split("checkpoint_epoch_", 1)[1].split("_", 1)[0])
            for path in checkpoint_paths
        ]
        metadata["swa_checkpoint_paths"] = checkpoint_paths
        metadata["swa_checkpoint_count"] = len(checkpoint_paths)
        
        # Save SWA model
        epoch_tag = f"{start_epoch}_to_{end_epoch}"
        swa_path = artifact_path(ROOT_DIR, f'swa_model_{epoch_tag}', cfg, seed, '.pth')
        if os.path.exists(swa_path) and not args.overwrite:
            raise RuntimeError(f"SWA artifact already exists: {swa_path}. Use --overwrite only after archiving or intentionally replacing it.")
        torch.save({"model_state_dict": avg_state, "metadata": metadata}, swa_path)
        print(f"✅ SWA model saved: {swa_path}")
        
        # Evaluate if requested
        if args.eval:
            print(f"\n📊 Evaluating SWA model...")
            
            model = LightweightAgeEstimator(cfg)
            model.load_state_dict(remap_state_dict_keys(avg_state))
            model.to(device)
            
            test_metrics = evaluate_mae(model, test_loader, cfg, device, modes=TTA_MODES)
            selected_tta = metadata.get("selection_metric", {}).get("tta", "multi")
            test_mae = test_metrics[selected_tta]
            print(f"🏆 SWA Test MAE (Seed {seed}, TTA={selected_tta}): {test_mae:.4f}")
            for mode in TTA_MODES:
                print(f"   SWA_MAE_{mode}: {test_metrics[mode]:.4f}")
            results[seed] = test_mae
            
            # Compare with original best model
            best_model_path = artifact_path(ROOT_DIR, 'best_model', cfg, seed, '.pth')
            if os.path.exists(best_model_path):
                best_state, checkpoint = load_model_state_package(best_model_path, device)
                expected_metadata = build_training_metadata(cfg, seed)
                mismatches = checkpoint_metadata_mismatches(checkpoint, expected_metadata)
                if mismatches:
                    raise RuntimeError(f"Best model metadata mismatch; refusing comparison. {format_metadata_mismatches(mismatches)}")
                model.load_state_dict(remap_state_dict_keys(best_state))
                orig_metrics = evaluate_mae(model, test_loader, cfg, device, modes=TTA_MODES)
                orig_mae = orig_metrics[selected_tta]
                print(f"📋 Original Best MAE: {orig_mae:.4f}")
                print(f"📈 Improvement: {orig_mae - test_mae:+.4f}")
    
    if results:
        print(f"\n{'='*60}")
        print("📊 Summary")
        print(f"{'='*60}")
        for s, mae in results.items():
            print(f"  Seed {s}: {mae:.4f}")


if __name__ == '__main__':
    main()
