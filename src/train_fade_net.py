"""
FADE-Net Training Script

Usage:
    python train_fade_net.py --split_id 0 --seed 42
    python train_fade_net.py --split_id 0 1 2 3 4 --seed 42  # Run all 5 folds
"""

import os
import sys
import json
import hashlib
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

# Knowledge distillation input bridge (EXP-031): the student consumes
# ImageNet-normalized inputs at input_size while the frozen FaRL teacher was
# trained on CLIP-normalized 224px inputs. Converting on GPU (denormalize ->
# resize -> renormalize) avoids a second dataloader with its own augmentation
# stream and guarantees teacher and student see the same augmented pixels.
IMAGENET_NORM_MEAN = (0.485, 0.456, 0.406)
IMAGENET_NORM_STD = (0.229, 0.224, 0.225)
CLIP_NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_NORM_STD = (0.26862954, 0.26130258, 0.27577711)
TEACHER_INPUT_SIZE = 224


def student_to_teacher_input(images):
    """Convert ImageNet-normalized student inputs to CLIP-normalized 224px teacher inputs."""
    imagenet_mean = torch.tensor(
        IMAGENET_NORM_MEAN, dtype=images.dtype, device=images.device
    ).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(
        IMAGENET_NORM_STD, dtype=images.dtype, device=images.device
    ).view(1, 3, 1, 1)
    clip_mean = torch.tensor(
        CLIP_NORM_MEAN, dtype=images.dtype, device=images.device
    ).view(1, 3, 1, 1)
    clip_std = torch.tensor(
        CLIP_NORM_STD, dtype=images.dtype, device=images.device
    ).view(1, 3, 1, 1)
    pixels = (images * imagenet_std + imagenet_mean).clamp(0.0, 1.0)
    pixels = F.interpolate(
        pixels, size=(TEACHER_INPUT_SIZE, TEACHER_INPUT_SIZE),
        mode='bilinear', align_corners=False,
    )
    return (pixels - clip_mean) / clip_std


def compute_kd_loss(student_main_logits, teacher_probs):
    """KL(p_teacher || p_student) on the main distribution head only."""
    student_log_probs = F.log_softmax(student_main_logits, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_OFFICIAL_DB = Path(__file__).resolve().parents[1] / "data" / "official" / "AFAD-Full.json"
OFFICIAL_FOLDS = {
    0: {"train": [0, 1, 2, 3, 4, 5], "val": [6, 7], "test": [8, 9]},
    1: {"train": [2, 3, 4, 5, 6, 7], "val": [8, 9], "test": [0, 1]},
    2: {"train": [4, 5, 6, 7, 8, 9], "val": [0, 1], "test": [2, 3]},
    3: {"train": [5, 6, 7, 8, 9, 0], "val": [1, 2], "test": [3, 4]},
    4: {"train": [6, 7, 8, 9, 0, 1], "val": [2, 3], "test": [4, 5]},
}


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
    """AFAD dataset for the official 15-72 age range."""

    def __init__(self, root_dir, transform=None, min_age=15, max_age=72, samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.min_age = min_age
        self.max_age = max_age

        if samples is not None:
            self.image_paths = [sample["image_path"] for sample in samples]
            self.ages = [int(sample["age"]) for sample in samples]
            self.identity_ids = [str(sample["identity_id"]) for sample in samples]
            logger.info(f"Loaded {len(self.image_paths)} official AFAD samples (age {min_age}-{max_age})")
            return

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


def get_transforms(
    img_size=256, is_train=True, random_erasing_p=0.1,
    train_crop_scale_min=0.8, aligned_preproc=False,
):
    """Get data transforms.

    aligned_preproc: inputs are official 281x281 aligned patches (CVPR2024
    protocol); train random-crops (jitter) and val center-crops to img_size
    without rescaling, preserving the alignment.
    """
    if aligned_preproc:
        if is_train:
            return transforms.Compose([
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=random_erasing_p),
            ])
        return transforms.Compose([
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                img_size, scale=(train_crop_scale_min, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=random_erasing_p),
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


def _official_relative_path(image_path):
    """Convert the official AFAD path to a path relative to the local AFAD root."""
    parts = str(image_path).replace('\\', '/').strip('/').split('/')
    if len(parts) < 3:
        raise ValueError(f"Unexpected official AFAD image path: {image_path}")
    return os.path.join(*parts[-3:])


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_split(official_db_path, afad_dir, min_age, max_age, split_id, strict=False):
    """Build a fold from the authors' AFAD-Full.json folder annotations."""
    with open(official_db_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Official AFAD database must be a non-empty JSON list: {official_db_path}")

    samples = []
    missing_paths = []
    expected_entries = 0
    identity_folders = {}

    for record in records:
        age = int(record['age'])
        if age < min_age or age > max_age:
            continue

        expected_entries += 1
        identity_id = str(record['id_num'])
        folder = int(record['folder'])
        previous_folder = identity_folders.setdefault(identity_id, folder)
        if previous_folder != folder:
            raise ValueError(f"Official AFAD identity {identity_id} occurs in multiple folders")

        image_path = _official_relative_path(record['img_path'])
        if not os.path.isfile(os.path.join(afad_dir, image_path)):
            missing_paths.append(image_path)
            continue

        samples.append({
            "image_path": image_path,
            "age": age,
            "identity_id": identity_id,
            "folder": folder,
        })

    if not samples:
        raise FileNotFoundError(f"No official AFAD samples found under {afad_dir}")
    if missing_paths and strict:
        raise FileNotFoundError(
            f"{len(missing_paths)} official AFAD samples are missing under {afad_dir}; "
            "remove --strict_official_data to run on the available subset"
        )
    if missing_paths:
        logger.warning(
            f"Official AFAD metadata has {len(missing_paths)} missing local samples; "
            "using the available subset and recording the coverage in results.json"
        )

    split_def = OFFICIAL_FOLDS[int(split_id)]
    train_folders = set(split_def['train'])
    val_folders = set(split_def['val'])
    test_folders = set(split_def['test'])
    train_idx, val_idx, test_idx = [], [], []
    for idx, sample in enumerate(samples):
        folder = sample['folder']
        if folder in train_folders:
            train_idx.append(idx)
        elif folder in val_folders:
            val_idx.append(idx)
        elif folder in test_folders:
            test_idx.append(idx)
        else:
            raise ValueError(f"Official AFAD sample has invalid folder: {folder}")

    metadata = {
        "source": "CVPR2024 Paplham & Franc official AFAD-Full.json",
        "split_file": os.path.abspath(official_db_path),
        "split_fingerprint": _sha256_file(official_db_path),
        "split_id": int(split_id),
        "train_folders": split_def['train'],
        "val_folders": split_def['val'],
        "test_folders": split_def['test'],
        "age_range": [min_age, max_age],
        "official_entries_in_age_range": expected_entries,
        "available_entries_in_age_range": len(samples),
        "missing_entries_in_age_range": len(missing_paths),
        "missing_policy": "error" if strict else "filter_missing",
        "available_identities_in_age_range": len({sample['identity_id'] for sample in samples}),
        "split_ratios_approx": [
            round(len(train_idx) / len(samples), 4),
            round(len(val_idx) / len(samples), 4),
            round(len(test_idx) / len(samples), 4),
        ],
    }
    return samples, train_idx, val_idx, test_idx, metadata


def build_adaptive_sigma_table(
    samples, train_idx, output_min_age, output_max_age, base_sigma, max_sigma
):
    """Build a train-only, frequency-adaptive label sigma table."""
    counts = np.zeros(output_max_age - output_min_age + 1, dtype=np.int64)
    for index in train_idx:
        age = int(samples[index]["age"])
        counts[age - output_min_age] += 1

    observed = counts[counts > 0]
    if observed.size == 0:
        raise ValueError("Cannot build adaptive sigma table from an empty training split")
    max_count = int(observed.max())
    min_count = int(observed.min())
    sigmas = np.full(counts.shape, float(base_sigma), dtype=np.float32)
    if max_count > min_count:
        denominator = np.log(max_count / min_count)
        for index, count in enumerate(counts):
            if count > 0:
                rarity = np.log(max_count / int(count)) / denominator
                sigmas[index] = base_sigma + (max_sigma - base_sigma) * rarity

    details = {
        "method": "train_frequency_log_rarity",
        "base_sigma": float(base_sigma),
        "max_sigma": float(max_sigma),
        "min_train_age_count": min_count,
        "max_train_age_count": max_count,
        "sigma_by_age": {
            str(age): float(sigmas[age - output_min_age])
            for age in range(output_min_age, output_max_age + 1)
            if counts[age - output_min_age] > 0
        },
        "train_count_by_age": {
            str(age): int(counts[age - output_min_age])
            for age in range(output_min_age, output_max_age + 1)
            if counts[age - output_min_age] > 0
        },
    }
    return sigmas, details


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
                    ema_model=None, ema_decay=0.999, ema_updates=0,
                    teacher_model=None, lambda_kd=0.0):
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

        # Optional knowledge distillation on the main distribution head
        kd_loss = None
        if teacher_model is not None and lambda_kd > 0:
            with torch.no_grad():
                teacher_probs = teacher_model(
                    student_to_teacher_input(images)
                )['main_prob']
            kd_loss = compute_kd_loss(outputs['main_logits'], teacher_probs)
            loss = loss + lambda_kd * kd_loss

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
            message = (f"  Batch {batch_idx+1}/{len(dataloader)} | "
                       f"Loss: {loss.item():.4f} | "
                       f"Main KL: {losses['main_kl'].item():.4f} | "
                       f"CDF: {losses['main_cdf'].item():.4f} | "
                       f"Coarse KL: {losses['coarse_kl'].item():.4f} | "
                       f"Gate: {losses['gate'].item():.4f} | "
                       f"Refine: {losses['refine'].item():.4f}")
            if kd_loss is not None:
                message += f" | KD: {kd_loss.item():.4f}"
            logger.info(message)

    return total_loss / max(total_samples, 1), ema_updates


def build_scheduler(optimizer, epochs, warmup_epochs, min_lr=1e-6):
    """Build a per-group warmup + cosine schedule with an absolute LR floor."""
    warmup_epochs = max(0, min(int(warmup_epochs), int(epochs)))
    total_epochs = max(1, int(epochs))
    base_lrs = [group['lr'] for group in optimizer.param_groups]

    def make_lambda(base_lr):
        min_factor = min(1.0, min_lr / base_lr) if base_lr > 0 else 1.0

        def lr_lambda(epoch):
            if warmup_epochs and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            cosine_epochs = max(1, total_epochs - warmup_epochs)
            progress = min(1.0, max(0.0, (epoch - warmup_epochs) / cosine_epochs))
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return lr_lambda

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[make_lambda(base_lr) for base_lr in base_lrs]
    )


def restore_training_state(
    checkpoint_path, model, ema_model, optimizer, backbone_lr, head_lr
):
    """Restore a trusted training checkpoint and apply the requested resume LRs."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    required_keys = {
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'best_val_mae', 'ema_updates',
    }
    missing_keys = sorted(required_keys - checkpoint.keys())
    if missing_keys:
        raise KeyError(f"Resume checkpoint is missing keys: {missing_keys}")
    if ema_model is not None and 'ema_state_dict' not in checkpoint:
        raise KeyError("Resume checkpoint does not contain ema_state_dict")

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if len(optimizer.param_groups) != 2:
        raise ValueError(
            f"Expected 2 optimizer parameter groups, got {len(optimizer.param_groups)}"
        )
    optimizer.param_groups[0]['lr'] = backbone_lr
    optimizer.param_groups[1]['lr'] = head_lr

    if ema_model is not None:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])

    return checkpoint


def build_checkpoint(
    args, epoch, model, ema_model, optimizer, scheduler,
    best_val_mae, ema_updates, resumed_from=None,
):
    """Build a CPU checkpoint shared by normal training and fixed-LR resume."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_val_mae': best_val_mae,
        'ema_updates': ema_updates,
        'config': vars(args),
    }
    if resumed_from is not None:
        checkpoint['resumed_from'] = os.path.abspath(resumed_from)
    if ema_model is not None:
        checkpoint['ema_state_dict'] = {
            k: v.cpu() for k, v in ema_model.state_dict().items()
        }
    return checkpoint


def run_single_fold(args, split_id, device):
    """Run training for a single fold."""
    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 Training Fold {split_id}")
    logger.info(f"{'='*60}")

    # Config
    config = Config()
    config.min_age = args.output_min_age
    config.max_age = args.output_max_age
    config.num_classes = args.output_max_age - args.output_min_age + 1
    config.data_min_age = args.data_min_age
    config.data_max_age = args.data_max_age
    config.img_size = args.input_size
    config.backbone_source = args.backbone_source
    config.backbone_name = args.backbone_name
    config.backbone_pretrained = args.backbone_pretrained
    config.backbone_weights = args.backbone_weights or None
    config.use_dcsr = True
    config.use_cgbr = args.use_cgbr
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
    config.early_stopping_patience = args.early_stopping_patience
    config.cgbr_start_epoch = args.cgbr_start_epoch
    config.cgbr_full_epoch = args.cgbr_full_epoch
    config.gradient_clip = args.gradient_clip
    config.validate()

    # Dataset
    train_transform = get_transforms(
        args.input_size,
        is_train=not args.clean_resume_training,
        random_erasing_p=args.random_erasing_p,
        train_crop_scale_min=args.train_crop_scale_min,
        aligned_preproc=args.aligned_preproc,
    )
    val_transform = get_transforms(args.input_size, is_train=False,
                                   aligned_preproc=args.aligned_preproc)
    if args.clean_resume_training:
        logger.info("Resume training transform: Resize + CenterCrop (no random augmentation)")
    else:
        logger.info(
            f"Training crop scale: {args.train_crop_scale_min:.3f}-1.000"
        )

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
    adaptive_sigmas = None
    if args.adaptive_label_sigma:
        adaptive_sigmas, sigma_details = build_adaptive_sigma_table(
            official_samples,
            train_idx,
            args.output_min_age,
            args.output_max_age,
            args.label_sigma,
            args.adaptive_sigma_max,
        )
        split_metadata["adaptive_label_sigma"] = sigma_details
        logger.info(
            "Adaptive label sigma: train-frequency log rarity, "
            f"range={adaptive_sigmas[adaptive_sigmas > args.label_sigma - 1e-8].min():.4f}"
            f"-{adaptive_sigmas.max():.4f}"
        )
    logger.info(
        f"Data age range: {args.data_min_age}-{args.data_max_age}; "
        f"model output range: {args.output_min_age}-{args.output_max_age} "
        f"({config.num_classes} classes)"
    )
    logger.info(f"Split source: official AFAD-Full.json ({official_db_path})")
    logger.info(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Create subsets
    if official_samples is not None:
        train_base = AFADDataset(
            args.afad_dir, train_transform, args.data_min_age, args.data_max_age, samples=official_samples
        )
        val_base = AFADDataset(
            args.afad_dir, val_transform, args.data_min_age, args.data_max_age, samples=official_samples
        )
        test_base = AFADDataset(
            args.afad_dir, val_transform, args.data_min_age, args.data_max_age, samples=official_samples
        )
    else:
        train_base = AFADDataset(args.afad_dir, train_transform, args.data_min_age, args.data_max_age)
        val_base = AFADDataset(args.afad_dir, val_transform, args.data_min_age, args.data_max_age)
        test_base = AFADDataset(args.afad_dir, val_transform, args.data_min_age, args.data_max_age)
    train_dataset = Subset(train_base, train_idx)
    val_dataset = Subset(val_base, val_idx)
    test_dataset = Subset(test_base, test_idx)

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
        min_age=args.output_min_age, max_age=args.output_max_age,
        label_sigma=args.label_sigma,
        lambda_main_kl=1.0, lambda_main_reg=1.0,
        lambda_coarse=args.lambda_coarse,
        lambda_refine=args.lambda_refine,
        lambda_gate=args.lambda_gate,
        lambda_cdf=args.lambda_cdf,
        label_sigma_by_age=adaptive_sigmas,
    ).to(device)

    # Optimizer with differential learning rate
    param_groups = model.get_params_groups(args.backbone_lr, args.head_lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # EMA
    ema_model = None
    if args.use_ema:
        from copy import deepcopy
        ema_model = deepcopy(model)
        ema_model.eval()
        ema_model.requires_grad_(False)
        ema_decay = args.ema_decay

    # Optional frozen FaRL teacher for knowledge distillation
    teacher_model = None
    if args.teacher_checkpoint and args.lambda_kd > 0:
        from teacher_vit import build_teacher
        teacher_model = build_teacher(
            weights_path=None,
            num_classes=config.num_classes,
            output_min_age=args.output_min_age,
        )
        teacher_checkpoint = torch.load(
            args.teacher_checkpoint, map_location='cpu', weights_only=False
        )
        if 'ema_state_dict' not in teacher_checkpoint:
            raise KeyError('Teacher checkpoint does not contain ema_state_dict')
        teacher_model.load_state_dict(teacher_checkpoint['ema_state_dict'])
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        logger.info(
            f"KD enabled: teacher={args.teacher_checkpoint} "
            f"(ema_state_dict, epoch={teacher_checkpoint.get('epoch')}) | "
            f"lambda_kd={args.lambda_kd}"
        )

    # Output directory
    output_dir = os.path.join(args.output_dir, f"fold{split_id}")
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    best_val_mae = float('inf')
    best_epoch = 0
    patience_counter = 0
    ema_updates = 0
    start_epoch = 0
    end_epoch = args.epochs
    resumed_from = None

    if args.resume_checkpoint:
        resumed_from = os.path.abspath(args.resume_checkpoint)
        output_checkpoint = os.path.abspath(
            os.path.join(output_dir, 'best_checkpoint.pth')
        )
        if resumed_from == output_checkpoint:
            raise ValueError("Resume checkpoint and output checkpoint must be different")
        resume_checkpoint = restore_training_state(
            resumed_from, model, ema_model, optimizer,
            args.backbone_lr, args.head_lr,
        )
        start_epoch = int(resume_checkpoint['epoch'])
        end_epoch = start_epoch + args.resume_extra_epochs
        best_val_mae = float(resume_checkpoint['best_val_mae'])
        best_epoch = start_epoch
        ema_updates = int(resume_checkpoint['ema_updates'])
        logger.info(
            f"Resumed checkpoint: {resumed_from} | epoch={start_epoch} | "
            f"best_val_mae={best_val_mae:.4f} | ema_updates={ema_updates}"
        )
        logger.info(
            f"Resume LR: backbone={optimizer.param_groups[0]['lr']:.2e}, "
            f"head={optimizer.param_groups[1]['lr']:.2e} | "
            f"extra_epochs={args.resume_extra_epochs}"
        )

    scheduler = None if args.fixed_lr_resume else build_scheduler(
        optimizer, args.epochs, args.warmup_epochs
    )

    if args.resume_checkpoint:
        initial_checkpoint = build_checkpoint(
            args, start_epoch, model, ema_model, optimizer, scheduler,
            best_val_mae, ema_updates, resumed_from=resumed_from,
        )
        torch.save(initial_checkpoint, os.path.join(output_dir, 'best_checkpoint.pth'))

    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()

        # Train
        train_loss, ema_updates = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            args.cgbr_start_epoch, args.cgbr_full_epoch, args.gradient_clip,
            ema_model, ema_decay, ema_updates,
            teacher_model=teacher_model, lambda_kd=args.lambda_kd,
        )

        # Evaluate
        eval_model = ema_model if ema_model else model
        val_mae, val_base_mae = evaluate(eval_model, val_loader, criterion, device,
                                         args.output_min_age, args.output_max_age)

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        logger.info(f"Epoch [{epoch+1}/{end_epoch}] | "
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

            checkpoint = build_checkpoint(
                args, epoch + 1, model, ema_model, optimizer, scheduler,
                best_val_mae, ema_updates, resumed_from=resumed_from,
            )

            torch.save(checkpoint, os.path.join(output_dir, 'best_checkpoint.pth'))
            logger.info(f"  🏆 New best! MAE: {best_val_mae:.4f}")
        else:
            patience_counter += 1
            if (args.early_stopping_patience > 0 and
                    patience_counter >= args.early_stopping_patience):
                logger.info(
                    f"Early stopping at epoch {epoch + 1}: "
                    f"no Val MAE improvement for {patience_counter} epochs"
                )
                break

    # Final test evaluation
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Fold {split_id} Final Results")
    logger.info(f"{'='*60}")
    logger.info(f"Best Val MAE: {best_val_mae:.4f} (Epoch {best_epoch})")

    test_mae = None
    test_base_mae = None
    raw_test_mae = None
    raw_test_base_mae = None
    ema_test_mae = None
    ema_test_base_mae = None
    evaluation_model = None
    if args.skip_final_test:
        logger.info("Final Test evaluation skipped; select ablations using Val only")
    else:
        checkpoint = torch.load(
            os.path.join(output_dir, 'best_checkpoint.pth'), map_location='cpu'
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        raw_test_mae, raw_test_base_mae = evaluate(
            model, test_loader, criterion, device,
            args.output_min_age, args.output_max_age
        )

        if ema_model:
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
            ema_model = ema_model.to(device)
            ema_test_mae, ema_test_base_mae = evaluate(
                ema_model, test_loader, criterion, device,
                args.output_min_age, args.output_max_age
            )
            test_mae, test_base_mae = ema_test_mae, ema_test_base_mae
            evaluation_model = 'ema'
        else:
            test_mae, test_base_mae = raw_test_mae, raw_test_base_mae
            evaluation_model = 'raw'

        logger.info(f"Test Raw MAE: {raw_test_mae:.4f}")
        logger.info(f"Test Raw Base MAE: {raw_test_base_mae:.4f}")
        if ema_test_mae is not None:
            logger.info(f"Test EMA MAE: {ema_test_mae:.4f}")
            logger.info(f"Test EMA Base MAE: {ema_test_base_mae:.4f}")
        logger.info(f"Test MAE ({evaluation_model}): {test_mae:.4f}")
        logger.info(f"Test Base MAE ({evaluation_model}): {test_base_mae:.4f}")

    # Save results
    results = {
        'fold': split_id,
        'completed_epochs': epoch + 1,
        'start_epoch': start_epoch,
        'trained_epochs_this_run': epoch + 1 - start_epoch,
        'resumed_from': resumed_from,
        'best_epoch': best_epoch,
        'best_val_mae': best_val_mae,
        'test_mae': test_mae,
        'test_base_mae': test_base_mae,
        'evaluation_model': evaluation_model,
        'raw_test_mae': raw_test_mae,
        'raw_test_base_mae': raw_test_base_mae,
        'ema_test_mae': ema_test_mae,
        'ema_test_base_mae': ema_test_base_mae,
        'config': vars(args),
        'split': split_metadata,
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
    parser.add_argument('--official_db', type=str, default=str(DEFAULT_OFFICIAL_DB),
                       help='Path to the authors\' official AFAD-Full.json')
    parser.add_argument('--strict_official_data', action='store_true',
                       help='Fail if official metadata references missing local AFAD images')
    parser.add_argument('--split_id', type=int, nargs='+', default=[0],
                       help='Split ID(s) to train (0-4)')
    parser.add_argument('--output_dir', type=str, default='outputs/fade_net',
                       help='Output directory')

    # Model
    parser.add_argument('--data_min_age', '--min_age', dest='data_min_age', type=int, default=15,
                        help='Minimum age present in the training/evaluation data')
    parser.add_argument('--data_max_age', '--max_age', dest='data_max_age', type=int, default=72,
                        help='Maximum age present in the training/evaluation data')
    parser.add_argument('--output_min_age', type=int, default=0,
                        help='Minimum age represented by the model output space')
    parser.add_argument('--output_max_age', type=int, default=80,
                        help='Maximum age represented by the model output space')
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--backbone_source', choices=('timm', 'torchvision'),
                        default='timm')
    parser.add_argument('--backbone_name', type=str,
                        default='mobilenetv4_conv_small')
    parser.add_argument('--backbone_weights', type=str, default='',
                        help='Local timm pretrained weights file')
    parser.add_argument('--no_pretrained', dest='backbone_pretrained',
                        action='store_false', default=True)
    parser.add_argument('--fusion_channels', type=int, default=96)
    parser.add_argument('--route_groups', type=int, default=8)
    parser.add_argument('--residual_bound', type=float, default=3.0)
    parser.add_argument('--gate_error_scale', type=float, default=3.0)
    parser.add_argument('--label_sigma', type=float, default=2.0)
    parser.add_argument('--adaptive_label_sigma', action='store_true',
                        help='Adapt label sigma using training-split age frequency only')
    parser.add_argument('--adaptive_sigma_max', type=float, default=3.0)
    parser.add_argument('--disable_cgbr', dest='use_cgbr', action='store_false', default=True,
                        help='Disable the CGBR branch and its scheduled losses for ablation')

    # Training
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--backbone_lr', type=float, default=3e-5)
    parser.add_argument('--head_lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='Stop after this many non-improving validation epochs; 0 disables it')
    parser.add_argument('--skip_final_test', action='store_true',
                        help='Save Val-selected checkpoint without evaluating Test')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Trusted checkpoint used to resume raw/EMA/optimizer state')
    parser.add_argument('--resume_extra_epochs', type=int, default=0,
                        help='Number of epochs to train after the resumed checkpoint epoch')
    parser.add_argument('--fixed_lr_resume', action='store_true',
                        help='Keep backbone/head learning rates fixed during resumed training')
    parser.add_argument('--clean_resume_training', action='store_true',
                        help='Use validation Resize+CenterCrop transforms during resume')
    parser.add_argument('--random_erasing_p', type=float, default=0.1,
                        help='Training RandomErasing probability; set 0 to disable')
    parser.add_argument('--train_crop_scale_min', type=float, default=0.8,
                        help='Minimum RandomResizedCrop scale used for training')
    parser.add_argument('--aligned_preproc', action='store_true',
                        help='Inputs are official 281x281 aligned patches; crop to input size without rescaling (CVPR2024 aligned protocol)')
    parser.add_argument('--gradient_clip', type=float, default=5.0)
    parser.add_argument('--cgbr_start_epoch', type=int, default=16)
    parser.add_argument('--cgbr_full_epoch', type=int, default=26)
    parser.add_argument('--lambda_coarse', type=float, default=0.3)
    parser.add_argument('--lambda_refine', type=float, default=0.5)
    parser.add_argument('--lambda_gate', type=float, default=0.1)
    parser.add_argument('--lambda_cdf', type=float, default=0.0)

    # Knowledge distillation (disabled by default)
    parser.add_argument('--teacher_checkpoint', type=str, default='',
                       help='FaRL teacher checkpoint (ema_state_dict) for KD; empty disables KD')
    parser.add_argument('--lambda_kd', type=float, default=0.0,
                       help='Weight of KL(p_teacher || p_student) on the main head; 0 disables KD')

    # EMA
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--ema_decay', type=float, default=0.999)

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    if args.data_min_age > args.data_max_age:
        parser.error('--data_min_age must be <= --data_max_age')
    if args.output_min_age > args.output_max_age:
        parser.error('--output_min_age must be <= --output_max_age')
    if args.lambda_cdf < 0:
        parser.error('--lambda_cdf must be non-negative')
    if args.lambda_kd < 0:
        parser.error('--lambda_kd must be non-negative')
    if args.teacher_checkpoint and args.lambda_kd == 0:
        parser.error('--lambda_kd must be positive when --teacher_checkpoint is given')
    if args.lambda_kd > 0 and not args.teacher_checkpoint:
        parser.error('--teacher_checkpoint is required when --lambda_kd is positive')
    if args.label_sigma <= 0:
        parser.error('--label_sigma must be positive')
    if args.adaptive_sigma_max < args.label_sigma:
        parser.error('--adaptive_sigma_max must be >= --label_sigma')
    if not 0.0 <= args.random_erasing_p <= 1.0:
        parser.error('--random_erasing_p must be in [0, 1]')
    if not 0.0 < args.train_crop_scale_min <= 1.0:
        parser.error('--train_crop_scale_min must be in (0, 1]')
    if args.output_min_age > args.data_min_age or args.output_max_age < args.data_max_age:
        parser.error('model output range must contain the data age range')
    if args.resume_checkpoint:
        if len(args.split_id) != 1:
            parser.error('--resume_checkpoint requires exactly one --split_id')
        if args.resume_extra_epochs <= 0:
            parser.error('--resume_extra_epochs must be positive when resuming')
        if not args.fixed_lr_resume:
            parser.error('--resume_checkpoint currently requires --fixed_lr_resume')
    elif args.resume_extra_epochs or args.fixed_lr_resume or args.clean_resume_training:
        parser.error(
            '--resume_extra_epochs/--fixed_lr_resume/--clean_resume_training '
            'require --resume_checkpoint'
        )
    if not args.use_cgbr:
        args.cgbr_start_epoch = args.epochs + 1
        args.cgbr_full_epoch = args.epochs + 2

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
