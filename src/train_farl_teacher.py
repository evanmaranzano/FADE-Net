"""
FaRL ViT-B/16 Teacher Training Script (EXP-030).

Trains an independent FaRL teacher on the official AFAD Fold split for later
distillation into the FADE-Net student. Main-path loss only: Gaussian soft
label KL + expectation regression (same form/weights as the student).

Usage:
    python train_farl_teacher.py --farl_weights <path> --split_id 0 --seed 42
"""

import os
import sys
import json
import time
import argparse
import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dcsr_cgbr import FADELoss
from teacher_vit import build_teacher
from train_fade_net import (
    DEFAULT_OFFICIAL_DB,
    AFADDataset,
    collate_fn,
    load_official_split,
    evaluate,
    build_scheduler,
    build_checkpoint,
    update_ema,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_FARL_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'pretrained', 'FaRL-Base-Patch16-LAIONFace20M-ep16.pth',
)


def get_teacher_transforms(img_size=224, is_train=True, random_erasing_p=0.1,
                           train_crop_scale_min=0.7, aligned_preproc=False):
    """Teacher transforms: same augmentation policy as the student, but with
    CLIP normalization constants (CLAHE is not used for the teacher).

    aligned_preproc: inputs are 281x281 aligned patches; crop the 256 student
    view first, then resize to the teacher's native img_size, mirroring the
    on-GPU KD bridge input distribution."""
    if aligned_preproc:
        if is_train:
            return transforms.Compose([
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(img_size),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
                transforms.RandomErasing(p=random_erasing_p),
            ])
        return transforms.Compose([
            transforms.CenterCrop(256),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                img_size, scale=(train_crop_scale_min, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            transforms.RandomErasing(p=random_erasing_p),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


class TeacherLoss(nn.Module):
    """Student main-path loss only: Gaussian soft label KL + expectation
    smooth-L1 regression with the same weights (1.0 / 1.0)."""

    def __init__(self, min_age=0, max_age=80, label_sigma=2.0,
                 lambda_main_kl=1.0, lambda_main_reg=1.0):
        super().__init__()
        # Reuse the student loss for the Gaussian soft label construction.
        self._base = FADELoss(min_age=min_age, max_age=max_age,
                              label_sigma=label_sigma)
        self.lambda_main_kl = lambda_main_kl
        self.lambda_main_reg = lambda_main_reg

    def forward(self, outputs, true_ages):
        target_dist = self._base._gaussian_label(true_ages)
        main_kl = F.kl_div(
            F.log_softmax(outputs['main_logits'], dim=1),
            target_dist, reduction='batchmean',
        )
        main_reg = F.smooth_l1_loss(outputs['base_age'], true_ages)
        total = self.lambda_main_kl * main_kl + self.lambda_main_reg * main_reg
        return {'total': total, 'main_kl': main_kl, 'main_reg': main_reg}


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch,
                    gradient_clip=5.0, ema_model=None, ema_decay=0.999,
                    ema_updates=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (images, ages, _) in enumerate(dataloader):
        if images.numel() == 0:
            continue
        images = images.to(device)
        ages = ages.to(device)

        outputs = model(images)
        losses = criterion(outputs, ages)
        loss = losses['total']

        optimizer.zero_grad()
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        if ema_model is not None:
            ema_updates = update_ema(ema_model, model, ema_decay, ema_updates)

        total_loss += loss.item() * ages.size(0)
        total_samples += ages.size(0)

        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(dataloader)} | "
                       f"Loss: {loss.item():.4f} | "
                       f"Main KL: {losses['main_kl'].item():.4f} | "
                       f"Main Reg: {losses['main_reg'].item():.4f}")

    return total_loss / max(total_samples, 1), ema_updates


def run_fold(args, split_id, device):
    """Run teacher training for a single fold."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training FaRL Teacher Fold {split_id}")
    logger.info(f"{'='*60}")

    train_transform = get_teacher_transforms(
        args.input_size, is_train=True,
        random_erasing_p=args.random_erasing_p,
        train_crop_scale_min=args.train_crop_scale_min,
        aligned_preproc=args.aligned_preproc,
    )
    val_transform = get_teacher_transforms(args.input_size, is_train=False,
                                           aligned_preproc=args.aligned_preproc)

    official_db_path = args.official_db
    if not official_db_path or not os.path.isfile(official_db_path):
        raise FileNotFoundError(
            "Official AFAD-Full.json is required; legacy project-generated splits are disabled"
        )
    (
        official_samples,
        train_idx,
        val_idx,
        test_idx,
        split_metadata,
    ) = load_official_split(
        official_db_path,
        args.afad_dir,
        args.data_min_age,
        args.data_max_age,
        split_id,
        strict=args.strict_official_data,
    )
    split_metadata["output_range"] = [args.output_min_age, args.output_max_age]
    logger.info(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    train_base = AFADDataset(
        args.afad_dir, train_transform, args.data_min_age, args.data_max_age,
        samples=official_samples,
    )
    val_base = AFADDataset(
        args.afad_dir, val_transform, args.data_min_age, args.data_max_age,
        samples=official_samples,
    )
    train_dataset = Subset(train_base, train_idx)
    val_dataset = Subset(val_base, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Model
    model = build_teacher(
        args.farl_weights,
        num_classes=args.output_max_age - args.output_min_age + 1,
        output_min_age=args.output_min_age,
    ).to(device)

    # Loss
    criterion = TeacherLoss(
        min_age=args.output_min_age, max_age=args.output_max_age,
        label_sigma=args.label_sigma,
    ).to(device)

    # Optimizer with differential learning rate (backbone / head)
    param_groups = [
        {'params': model.visual.parameters(), 'lr': args.backbone_lr},
        {'params': model.head.parameters(), 'lr': args.head_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # EMA
    ema_model = None
    ema_decay = args.ema_decay
    if args.use_ema:
        ema_model = deepcopy(model)
        ema_model.eval()
        ema_model.requires_grad_(False)

    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)

    output_dir = os.path.join(args.output_dir, f"fold{split_id}")
    os.makedirs(output_dir, exist_ok=True)

    best_val_mae = float('inf')
    best_epoch = 0
    patience_counter = 0
    ema_updates = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss, ema_updates = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            args.gradient_clip, ema_model, ema_decay, ema_updates,
        )

        eval_model = ema_model if ema_model else model
        val_mae, val_base_mae = evaluate(eval_model, val_loader, criterion, device,
                                         args.output_min_age, args.output_max_age)

        scheduler.step()
        epoch_time = time.time() - epoch_start

        logger.info(f"Epoch [{epoch+1}/{args.epochs}] | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val MAE: {val_mae:.4f} | "
                    f"Val Base MAE: {val_base_mae:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"Time: {epoch_time:.1f}s")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch + 1
            patience_counter = 0
            checkpoint = build_checkpoint(
                args, epoch + 1, model, ema_model, optimizer, scheduler,
                best_val_mae, ema_updates,
            )
            torch.save(checkpoint, os.path.join(output_dir, 'best_checkpoint.pth'))
            logger.info(f"  New best! MAE: {best_val_mae:.4f}")
        else:
            patience_counter += 1
            if (args.early_stopping_patience > 0 and
                    patience_counter >= args.early_stopping_patience):
                logger.info(
                    f"Early stopping at epoch {epoch + 1}: "
                    f"no Val MAE improvement for {patience_counter} epochs"
                )
                break

    # Teacher selection is Val-only: never touch Test.
    logger.info("Final Test evaluation skipped; teacher is a Val-selected artifact")
    results = {
        'fold': split_id,
        'completed_epochs': epoch + 1,
        'start_epoch': 0,
        'trained_epochs_this_run': epoch + 1,
        'resumed_from': None,
        'best_epoch': best_epoch,
        'best_val_mae': best_val_mae,
        'test_mae': None,
        'test_base_mae': None,
        'evaluation_model': None,
        'raw_test_mae': None,
        'raw_test_base_mae': None,
        'ema_test_mae': None,
        'ema_test_base_mae': None,
        'config': vars(args),
        'split': split_metadata,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def build_arg_parser():
    parser = argparse.ArgumentParser(description='FaRL ViT-B/16 Teacher Training (EXP-030)')

    # Data
    parser.add_argument('--afad_dir', type=str, default='/data/AFAD',
                       help='Path to AFAD dataset')
    parser.add_argument('--official_db', type=str, default=str(DEFAULT_OFFICIAL_DB),
                       help='Path to the authors\' official AFAD-Full.json')
    parser.add_argument('--strict_official_data', action='store_true',
                       help='Fail if official metadata references missing local AFAD images')
    parser.add_argument('--split_id', type=int, nargs='+', default=[0],
                       help='Split ID(s) to train (0-4)')
    parser.add_argument('--output_dir', type=str, default='outputs/farl_teacher',
                       help='Output directory')

    # Model
    parser.add_argument('--farl_weights', type=str, default=DEFAULT_FARL_WEIGHTS,
                       help='Path to FaRL-Base-Patch16 checkpoint')
    parser.add_argument('--data_min_age', type=int, default=15,
                       help='Minimum age present in the training/evaluation data')
    parser.add_argument('--data_max_age', type=int, default=72,
                       help='Maximum age present in the training/evaluation data')
    parser.add_argument('--output_min_age', type=int, default=0,
                       help='Minimum age represented by the model output space')
    parser.add_argument('--output_max_age', type=int, default=80,
                       help='Maximum age represented by the model output space')
    parser.add_argument('--input_size', type=int, default=224,
                       help='FaRL native input resolution')
    parser.add_argument('--label_sigma', type=float, default=2.0)

    # Training
    parser.add_argument('--epochs', type=int, default=55)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--head_lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Stop after this many non-improving validation epochs; 0 disables it')
    parser.add_argument('--gradient_clip', type=float, default=5.0)
    parser.add_argument('--random_erasing_p', type=float, default=0.1,
                       help='Training RandomErasing probability; set 0 to disable')
    parser.add_argument('--train_crop_scale_min', type=float, default=0.7,
                       help='Minimum RandomResizedCrop scale used for training')
    parser.add_argument('--aligned_preproc', action='store_true',
                       help='Inputs are official 281x281 aligned patches; crop the 256 student view then resize to input size (CVPR2024 aligned protocol)')

    # EMA
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--ema_decay', type=float, default=0.999)

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    return parser


def validate_args(args, parser):
    if args.data_min_age > args.data_max_age:
        parser.error('--data_min_age must be <= --data_max_age')
    if args.output_min_age > args.output_max_age:
        parser.error('--output_min_age must be <= --output_max_age')
    if args.label_sigma <= 0:
        parser.error('--label_sigma must be positive')
    if not 0.0 <= args.random_erasing_p <= 1.0:
        parser.error('--random_erasing_p must be in [0, 1]')
    if not 0.0 < args.train_crop_scale_min <= 1.0:
        parser.error('--train_crop_scale_min must be in (0, 1]')
    if args.output_min_age > args.data_min_age or args.output_max_age < args.data_max_age:
        parser.error('model output range must contain the data age range')
    if not os.path.isfile(args.farl_weights):
        parser.error(f'--farl_weights not found: {args.farl_weights}')


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    for split_id in args.split_id:
        run_fold(args, split_id, device)


if __name__ == '__main__':
    main()
