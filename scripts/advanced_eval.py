"""
Advanced TTA and Multi-Seed Ensemble Evaluation

使用方法:
    python scripts/advanced_eval.py [--seed SEED] [--tta MODE] [--ensemble]

功能:
    1. 扩展 TTA (Multi-Scale 3尺度 + flip = 6x 平均)
    2. 多种子 Ensemble
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from config import Config
from model import LightweightAgeEstimator
from dataset import get_dataloaders
from evaluation import TTA_MODES, evaluate_mae, normalize_tta_mode, predict_probs, probs_to_ages
from experiment import (
    artifact_path,
    build_training_metadata,
    checkpoint_metadata_mismatches,
    load_model_state_package,
)


def apply_common_overrides(cfg, args):
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


def load_checked_model(model_path, cfg, seed, device):
    model = LightweightAgeEstimator(cfg)
    state_dict, checkpoint = load_model_state_package(model_path, device)
    expected_metadata = build_training_metadata(cfg, seed)
    mismatches = checkpoint_metadata_mismatches(checkpoint, expected_metadata)
    if mismatches:
        raise RuntimeError(f"Checkpoint metadata mismatch; refusing to evaluate. {format_metadata_mismatches(mismatches)}")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def selected_modes(tta_arg):
    if tta_arg == "all":
        return TTA_MODES
    return (normalize_tta_mode(tta_arg),)


def ensemble_predict(models, images, mode, base_size):
    """
    Ensemble prediction: average probability distributions from multiple models.
    """
    all_probs = [predict_probs(model, images, mode=mode, base_size=base_size) for model in models]
    return torch.stack(all_probs, dim=0).mean(dim=0)


def evaluate_ensemble(models, test_loader, config, device, modes):
    mae_sums = {mode: 0.0 for mode in modes}
    count = 0

    with torch.no_grad():
        for images, _labels, ages in tqdm(test_loader, desc="Ensemble Eval"):
            if images.numel() == 0:
                continue

            images = images.to(device)
            ages = ages.to(device)

            for mode in modes:
                probs = ensemble_predict(models, images, mode=mode, base_size=config.img_size)
                output_ages = probs_to_ages(probs, config.num_classes)
                mae_sums[mode] += torch.abs(output_ages - ages).sum().item()
            count += images.size(0)

    if count == 0:
        raise RuntimeError("No valid evaluation samples were loaded.")

    return {mode: mae_sums[mode] / count for mode in modes}


def main():
    parser = argparse.ArgumentParser(description="Advanced Evaluation")
    parser.add_argument('--seed', type=int, default=1337, help='Seed to evaluate (default: 1337)')
    parser.add_argument('--tta', type=str, default='all', choices=['all', 'none', 'raw', 'flip', 'multi'], help='TTA mode')
    parser.add_argument('--ensemble', action='store_true', help='Use multi-seed ensemble')
    parser.add_argument('--seeds', type=str, default='42,1337', help='Seeds for ensemble (comma-separated)')
    parser.add_argument('--split', type=str, choices=['80-10-10', '90-5-5', '72-8-20'], help='Split protocol')
    parser.add_argument('--backbone_source', type=str, choices=['torchvision', 'timm'], help='Backbone provider')
    parser.add_argument('--backbone_name', type=str, help='Backbone model name')
    parser.add_argument('--no_pretrained', action='store_true', help='Disable pretrained backbone weights')
    parser.add_argument('--afad_dir', type=str, help='Override AFAD dataset directory')
    parser.add_argument('--experiment_tag', type=str, help='Append tag to experiment id for smoke or side runs')
    parser.add_argument('--split_file_tag', type=str, help='Use tagged split file/artifact identity')
    parser.add_argument('--allow_legacy_split_upgrade', action='store_true', help='Trust and stamp a legacy split after size/index validation')
    parser.add_argument('--model_path', type=str, help='Explicit model checkpoint path')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = Config()
    apply_common_overrides(cfg, args)
    
    # Load data
    print("📊 Loading test data...")
    _, _, test_loader, _ = get_dataloaders(cfg)
    modes = selected_modes(args.tta)
    
    if args.ensemble:
        # Multi-seed ensemble
        seeds = [int(s) for s in args.seeds.split(',')]
        print(f"🎯 Ensemble Mode: Seeds {seeds}")
        
        models = []
        for seed in seeds:
            model_path = artifact_path(ROOT_DIR, 'best_model', cfg, seed, '.pth')
            if not os.path.exists(model_path):
                print(f"⚠️ Model not found: {model_path}")
                continue
            
            model = load_checked_model(model_path, cfg, seed, device)
            models.append(model)
            print(f"✅ Loaded: {os.path.basename(model_path)}")
        
        if len(models) < 2:
            print("❌ Need at least 2 models for ensemble!")
            return
        
        metrics = evaluate_ensemble(models, test_loader, cfg, device, modes)
        print(f"\n🏆 Ensemble Test MAE (Seeds {seeds})")
        for mode, value in metrics.items():
            print(f"   MAE_{mode}: {value:.4f}")
        
    else:
        # Single model evaluation with TTA
        model_path = args.model_path or artifact_path(ROOT_DIR, 'best_model', cfg, args.seed, '.pth')
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        
        model = load_checked_model(model_path, cfg, args.seed, device)
        print(f"✅ Loaded: {os.path.basename(model_path)}")
        
        print(f"\n📊 Evaluating TTA modes: {', '.join(modes)}")
        metrics = evaluate_mae(model, test_loader, cfg, device, modes=modes)
        print(f"\n🏆 Test MAE (Seed {args.seed})")
        for mode, value in metrics.items():
            print(f"   MAE_{mode}: {value:.4f}")


if __name__ == '__main__':
    main()
