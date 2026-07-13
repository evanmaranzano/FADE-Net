"""
FADE-Net Training Script

Usage:
    python train_fade_net.py --split_id 0 --seed 42
    python train_fade_net.py --split_id 0 1 2 3 4 --seed 42  # Run all 5 folds
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from fade_net import FADENet
from dcsr_cgbr import FADELoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@torch.no_grad()
def update_ema(ema_model, model, decay, num_updates):
    """Update EMA parameters per optimizer step and synchronize model buffers."""
    decay = min(decay, (1.0 + num_updates) / (10.0 + num_updates))

    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    if ema_params.keys() != model_params.keys():
        raise ValueError("EMA and model parameter names do not match")

    for name, ema_param in ema_params.items():
        ema_param.mul_(decay).add_(model_params[name].detach(), alpha=1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    if ema_buffers.keys() != model_buffers.keys():
        raise ValueError("EMA and model buffer names do not match")
    for name, ema_buffer in ema_buffers.items():
        ema_buffer.copy_(model_buffers[name].detach())

    return num_updates + 1


class AFADDataset(Dataset):
    """AFAD Dataset for age 15-40."""

    def __init__(self, root_dir, transform=None, min_age=15, max_age=40):
        self.root_dir = root_dir
        self.transform = transform
        self.min_age = min_age
        self.max_age = max_age

        # Discover all images
        self.image_paths = []
        self.ages = []
        self.identity_ids = []

        for age_dir in sorted(os.listdir(root_dir)):
            age_path = os.path.join(root_dir, age_dir)
            if not os.path.isdir(age_path) or not age_dir.isdigit():
                continue
            age = int(age_dir)
            if age < min_age or age > max_age:
                continue
            for gender_dir in os.listdir(age_path):
                gender_path = os.path.join(age_path, gender_dir)
                if not os.path.isdir(gender_path):
                    continue
                for fname in os.listdir(gender_path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(age_dir, gender_dir, fname))
                        self.ages.append(age)
                        # Extract identity ID from filename
                        self.identity_ids.append(fname.split('-')[0])

        logger.info(f"Loaded {len(self.image_paths)} images (age {min_age}-{max_age})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        try:
            image = Image.open(img_path).convert('RGB')
            age = self.ages[idx]
            if self.transform:
                image = self.transform(image)
            return image, age, idx
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            return None


def collate_fn(batch):
    """Custom collate function that filters None samples."""
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    images, ages, indices = zip(*batch)
    return torch.stack(images), torch.tensor(ages, dtype=torch.float32), torch.tensor(indices)


def get_transforms(img_size=256, is_train=True):
    """Get data transforms."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def load_split(split_path):
    """Load split file and return indices."""
    with open(split_path, 'r') as f:
        data = json.load(f)
    return data['train'], data['val'], data['test']


def evaluate(model, dataloader, criterion, device, min_age, max_age):
    """Evaluate model on validation/test set."""
    model.eval()
    total_mae = 0.0
    total_samples = 0
    total_base_mae = 0.0

    with torch.no_grad():
        for images, ages, _ in dataloader:
            if images.numel() == 0:
                continue
            images = images.to(device)
            ages = ages.to(device)

            outputs = model(images)
            pred_ages = outputs['age']
            base_ages = outputs['base_age']

            # MAE
            mae = torch.abs(pred_ages - ages).sum().item()
            base_mae = torch.abs(base_ages - ages).sum().item()
            total_mae += mae
            total_base_mae += base_mae
            total_samples += ages.size(0)

    avg_mae = total_mae / max(total_samples, 1)
    avg_base_mae = total_base_mae / max(total_samples, 1)
    return avg_mae, avg_base_mae


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch,
                    cgbr_start_epoch=16, cgbr_full_epoch=26, gradient_clip=5.0,
                    ema_model=None, ema_decay=0.999, ema_updates=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (images, ages, _) in enumerate(dataloader):
        if images.numel() == 0:
            continue
        images = images.to(device)
        ages = ages.to(device)

        # Forward
        outputs = model(images)

        # Loss
        losses = criterion(outputs, ages, epoch, cgbr_start_epoch, cgbr_full_epoch)
        loss = losses['total']

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        if ema_model is not None:
            ema_updates = update_ema(ema_model, model, ema_decay, ema_updates)

        total_loss += loss.item() * ages.size(0)
        total_samples += ages.size(0)

        # Logging
        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(dataloader)} | "
                       f"Loss: {loss.item():.4f} | "
                       f"Main KL: {losses['main_kl'].item():.4f} | "
                       f"Coarse KL: {losses['coarse_kl'].item():.4f} | "
                       f"Gate: {losses['gate'].item():.4f} | "
                       f"Refine: {losses['refine'].item():.4f}")

    return total_loss / max(total_samples, 1), ema_updates


def run_single_fold(args, split_id, device):
    """Run training for a single fold."""
    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 Training Fold {split_id}")
    logger.info(f"{'='*60}")

    # Config
    config = Config()
    config.min_age = args.min_age
    config.max_age = args.max_age
    config.num_classes = args.max_age - args.min_age + 1
    config.img_size = args.input_size
    config.use_dcsr = True
    config.use_cgbr = True
    config.fusion_channels = args.fusion_channels
    config.route_groups = args.route_groups
    config.residual_bound = args.residual_bound
    config.gate_error_scale = args.gate_error_scale
    config.label_sigma = args.label_sigma
    config.backbone_lr = args.backbone_lr
    config.head_lr = args.head_lr
    config.weight_decay = args.weight_decay
    config.epochs = args.epochs
    config.warmup_epochs = args.warmup_epochs
    config.cgbr_start_epoch = args.cgbr_start_epoch
    config.cgbr_full_epoch = args.cgbr_full_epoch
    config.gradient_clip = args.gradient_clip

    # Dataset
    train_transform = get_transforms(args.input_size, is_train=True)
    val_transform = get_transforms(args.input_size, is_train=False)

    full_dataset = AFADDataset(args.afad_dir, min_age=args.min_age, max_age=args.max_age)

    # Load split
    split_path = os.path.join(args.split_dir, f"dataset_split_AFAD_15_40_iddisjoint_fold{split_id}.json")
    train_idx, val_idx, test_idx = load_split(split_path)
    logger.info(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Create subsets
    train_dataset = Subset(AFADDataset(args.afad_dir, train_transform, args.min_age, args.max_age), train_idx)
    val_dataset = Subset(AFADDataset(args.afad_dir, val_transform, args.min_age, args.max_age), val_idx)
    test_dataset = Subset(AFADDataset(args.afad_dir, val_transform, args.min_age, args.max_age), test_idx)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Model
    model = FADENet(config).to(device)

    # Loss
    criterion = FADELoss(
        min_age=args.min_age, max_age=args.max_age,
        label_sigma=args.label_sigma,
        lambda_main_kl=1.0, lambda_main_reg=1.0,
        lambda_coarse=args.lambda_coarse,
        lambda_refine=args.lambda_refine,
        lambda_gate=args.lambda_gate,
    )

    # Optimizer with differential learning rate
    param_groups = model.get_params_groups(args.backbone_lr, args.head_lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # EMA
    ema_model = None
    if args.use_ema:
        from copy import deepcopy
        ema_model = deepcopy(model)
        ema_model.eval()
        ema_model.requires_grad_(False)
        ema_decay = args.ema_decay

    # Output directory
    output_dir = os.path.join(args.output_dir, f"fold{split_id}")
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    best_val_mae = float('inf')
    best_epoch = 0
    patience_counter = 0
    ema_updates = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, ema_updates = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            args.cgbr_start_epoch, args.cgbr_full_epoch, args.gradient_clip,
            ema_model, ema_decay, ema_updates
        )

        # Evaluate
        eval_model = ema_model if ema_model else model
        val_mae, val_base_mae = evaluate(eval_model, val_loader, criterion, device,
                                         args.min_age, args.max_age)

        # Learning rate scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        logger.info(f"Epoch [{epoch+1}/{args.epochs}] | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val MAE: {val_mae:.4f} | "
                    f"Val Base MAE: {val_base_mae:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch + 1
            patience_counter = 0

            # Save checkpoint (save CPU state dicts to avoid CUDA context issues)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_mae': best_val_mae,
                'ema_updates': ema_updates,
                'config': vars(args),
            }
            if ema_model:
                checkpoint['ema_state_dict'] = {k: v.cpu() for k, v in ema_model.state_dict().items()}

            torch.save(checkpoint, os.path.join(output_dir, 'best_checkpoint.pth'))
            logger.info(f"  🏆 New best! MAE: {best_val_mae:.4f}")
        else:
            patience_counter += 1

    # Final test evaluation
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Fold {split_id} Final Results")
    logger.info(f"{'='*60}")
    logger.info(f"Best Val MAE: {best_val_mae:.4f} (Epoch {best_epoch})")

    # Load best checkpoint (load to CPU first to avoid CUDA context issues)
    checkpoint = torch.load(os.path.join(output_dir, 'best_checkpoint.pth'), map_location='cpu')
    if ema_model:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        ema_model = ema_model.to(device)
        test_mae, test_base_mae = evaluate(ema_model, test_loader, criterion, device,
                                          args.min_age, args.max_age)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        test_mae, test_base_mae = evaluate(model, test_loader, criterion, device,
                                          args.min_age, args.max_age)

    logger.info(f"Test MAE: {test_mae:.4f}")
    logger.info(f"Test Base MAE: {test_base_mae:.4f}")

    # Save results
    results = {
        'fold': split_id,
        'best_epoch': best_epoch,
        'best_val_mae': best_val_mae,
        'test_mae': test_mae,
        'test_base_mae': test_base_mae,
        'config': vars(args),
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description='FADE-Net Training')

    # Data
    parser.add_argument('--afad_dir', type=str, default='/data/AFAD',
                       help='Path to AFAD dataset')
    parser.add_argument('--split_dir', type=str, default='/data/FADE-Net',
                       help='Path to split files')
    parser.add_argument('--split_id', type=int, nargs='+', default=[0],
                       help='Split ID(s) to train (0-4)')
    parser.add_argument('--output_dir', type=str, default='outputs/fade_net',
                       help='Output directory')

    # Model
    parser.add_argument('--min_age', type=int, default=15)
    parser.add_argument('--max_age', type=int, default=40)
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--fusion_channels', type=int, default=96)
    parser.add_argument('--route_groups', type=int, default=8)
    parser.add_argument('--residual_bound', type=float, default=3.0)
    parser.add_argument('--gate_error_scale', type=float, default=3.0)
    parser.add_argument('--label_sigma', type=float, default=2.0)

    # Training
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--backbone_lr', type=float, default=3e-5)
    parser.add_argument('--head_lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--gradient_clip', type=float, default=5.0)
    parser.add_argument('--cgbr_start_epoch', type=int, default=16)
    parser.add_argument('--cgbr_full_epoch', type=int, default=26)
    parser.add_argument('--lambda_coarse', type=float, default=0.3)
    parser.add_argument('--lambda_refine', type=float, default=0.5)
    parser.add_argument('--lambda_gate', type=float, default=0.1)

    # EMA
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--ema_decay', type=float, default=0.999)

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Run folds
    all_results = []
    for split_id in args.split_id:
        results = run_single_fold(args, split_id, device)
        all_results.append(results)

    # Summary
    if len(all_results) > 1:
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Cross-Fold Summary")
        logger.info(f"{'='*60}")

        val_maes = [r['best_val_mae'] for r in all_results]
        test_maes = [r['test_mae'] for r in all_results]

        logger.info(f"Val MAE:  {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")
        logger.info(f"Test MAE: {np.mean(test_maes):.4f} ± {np.std(test_maes):.4f}")

        for r in all_results:
            logger.info(f"  Fold {r['fold']}: Val={r['best_val_mae']:.4f}, Test={r['test_mae']:.4f}")


if __name__ == '__main__':
    main()
