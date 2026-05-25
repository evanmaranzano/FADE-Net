import torch
import sys
import os
import argparse

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.model import LightweightAgeEstimator
from src.ablation_profiles import apply_ablation_profile, parse_ablation_ids
from thop import profile, clever_format

def apply_common_overrides(cfg, args):
    apply_ablation_profile(cfg, getattr(args, "ablation_id", None))
    if args.backbone_source is not None:
        cfg.backbone_source = args.backbone_source
    if args.backbone_name is not None:
        cfg.backbone_name = args.backbone_name
    if args.no_pretrained:
        cfg.backbone_pretrained = False


def main():
    parser = argparse.ArgumentParser(description="Count FADE-Net parameters and FLOPs")
    parser.add_argument('--backbone_source', type=str, choices=['torchvision', 'timm'], help='Backbone provider')
    parser.add_argument('--backbone_name', type=str, help='Backbone model name')
    parser.add_argument('--no_pretrained', action='store_true', help='Disable pretrained backbone weights')
    parser.add_argument('--ablation_id', type=str, choices=[item for item in parse_ablation_ids("A0,A1,A2,A3,A4,A5,A6,A7,A8,A9")], help='Apply an A0-A9 ablation profile')
    args = parser.parse_args()

    cfg = Config()
    apply_common_overrides(cfg, args)
    model = LightweightAgeEstimator(cfg)
    model.eval()
    
    # 1. Basic Count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Total Params (M): {total_params / 1e6:.2f}M")

    # 2. THOP (FLOPs)
    try:
        input = torch.randn(1, 3, cfg.img_size, cfg.img_size)
        flops, params = profile(model, inputs=(input, ), verbose=False)
        flops_fmt, params_fmt = clever_format([flops, params], "%.2f")
        print(f"FLOPs: {flops_fmt}")
        print(f"Params (THOP): {params_fmt}")
    except Exception as e:
        print(f"FLOPs: unavailable")
        print(f"THOP calculation failed: {e}")

if __name__ == "__main__":
    main()
