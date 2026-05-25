import os
import argparse
import re
import random
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from config import Config, ROOT_DIR
from dataset import get_dataloaders
from model import LightweightAgeEstimator
from utils import DLDLProcessor, EMAModel, CombinedLoss, seed_everything
import csv
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from experiment import (
    artifact_path,
    build_training_metadata,
    checkpoint_metadata_mismatches,
    hard_distillation_schedule_metadata as _metadata_hard_distillation_schedule_metadata,
    hard_distillation_start_epoch as _metadata_hard_distillation_start_epoch,
    load_model_state_package,
    save_model_package,
)
from evaluation import TTA_MODES, evaluate_mae, predict_probs, probs_to_ages

# ==========================================
# Reproducibility
# ==========================================



# ==========================================
# MixUp 数据增强函数
# ==========================================
def mixup_data(x, y_dist, y_age, alpha=0.4):
    """
    MixUp 数据增强: 混合两个样本的图像和标签
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y_dist = lam * y_dist + (1 - lam) * y_dist[index]
    # 对真实年龄也做 mixup，用于 Aux Loss
    mixed_y_age = lam * y_age + (1 - lam) * y_age[index]
    
    return mixed_x, mixed_y_dist, mixed_y_age


def make_amp_context(device_type):
    if device_type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            try:
                return torch.amp.autocast("cuda")
            except TypeError:
                pass
        return torch.cuda.amp.autocast()
    return nullcontext()


def make_grad_scaler(device_type):
    enabled = device_type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def amp_step_was_skipped(scale_before, scale_after):
    return scale_after < scale_before


def _pack_numpy_random_state(state):
    name, keys, pos, has_gauss, cached_gaussian = state
    return (name, keys.tolist(), pos, has_gauss, cached_gaussian)


def _unpack_numpy_random_state(state):
    name, keys, pos, has_gauss, cached_gaussian = state
    return (name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached_gaussian)


def checkpoint_extra_state(scheduler_controller, scaler=None):
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": _pack_numpy_random_state(np.random.get_state()),
        "torch_rng_state": torch.get_rng_state(),
        "scheduler_controller_pending_steps": int(getattr(scheduler_controller, "pending_steps", 0)),
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_checkpoint_extra_state(checkpoint, scheduler_controller, scaler=None):
    if not isinstance(checkpoint, dict):
        return
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(_unpack_numpy_random_state(checkpoint["numpy_random_state"]))
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    scheduler_controller.pending_steps = int(checkpoint.get("scheduler_controller_pending_steps", 0))


def backbone_learning_rate(base_lr, effective_pretrained):
    return base_lr * 0.1 if effective_pretrained else base_lr


def model_needs_aux_outputs(cfg):
    return bool(getattr(cfg, "use_adaptive_triplet", False) or getattr(cfg, "use_moe", False))


def model_forward_for_loss(model, cfg, images):
    if model_needs_aux_outputs(cfg):
        return model(images, return_features=True)
    return model(images), None, None


def hard_distillation_start_epoch(total_epochs, default_start=105, tail_epochs=15):
    return _metadata_hard_distillation_start_epoch(total_epochs, default_start, tail_epochs)


def hard_distillation_schedule_metadata(total_epochs):
    return _metadata_hard_distillation_schedule_metadata(total_epochs)


class SchedulerStepController:
    def __init__(self, scheduler, max_epochs):
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.pending_steps = 0

    def step_epoch(self, epoch, optimizer_stepped):
        if epoch < self.max_epochs:
            self.pending_steps += 1
        if not optimizer_stepped or self.pending_steps == 0:
            return 0
        steps = self.pending_steps
        for _ in range(steps):
            self.scheduler.step()
        self.pending_steps = 0
        return steps


# ==========================================
# CSV Logger
# ==========================================
class CSVLogger:
    def __init__(self, filepath, headers, resume=False):
        self.filepath = filepath
        self.headers = headers
        if not resume or not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
    def log(self, row_data):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

# ==========================================
# Checkpoint 保存
# ==========================================
def save_checkpoint(state, filename="last_checkpoint.pth"):
    torch.save(state, filename)


def _format_metadata_mismatches(mismatches):
    return ", ".join([f"{key}: checkpoint={old!r}, current={new!r}" for key, old, new in mismatches])


def _guard_fresh_artifact_overwrite(paths, expected_metadata, device, allow_overwrite=False):
    for path in paths:
        if not os.path.exists(path):
            continue

        detail = "non-checkpoint artifact"
        if str(path).endswith(".pth"):
            try:
                try:
                    checkpoint = torch.load(path, map_location=device, weights_only=True)
                except TypeError:
                    checkpoint = torch.load(path, map_location=device)
                mismatches = checkpoint_metadata_mismatches(checkpoint, expected_metadata)
                detail = (
                    "checkpoint metadata matches current run"
                    if not mismatches
                    else _format_metadata_mismatches(mismatches)
                )
            except Exception as exc:
                detail = f"could not inspect checkpoint metadata: {exc}"

        if allow_overwrite:
            print(f"⚠️ Overwriting existing artifact by explicit request: {path}")
            print(f"   {detail}")
            continue
        raise RuntimeError(
            "Existing artifact already exists; refusing to overwrite with a fresh run. "
            f"Path: {path}. {detail}. "
            "Use --overwrite_artifacts only after archiving or intentionally replacing the old run."
        )


def _average_loss_sums(loss_sums, sample_count):
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    return {name: value / sample_count for name, value in loss_sums.items()}


def _accumulate_loss_components(loss_sums, batch_size, total_loss, kl_loss, l1_loss=0.0, rank_loss=0.0, mv_loss=0.0, triplet_loss=0.0, asym_loss=0.0, moe_gate_loss=0.0):
    for name, value in (
        ("total", total_loss),
        ("kl", kl_loss),
        ("l1", l1_loss),
        ("rank", rank_loss),
        ("mv", mv_loss),
        ("triplet", triplet_loss),
        ("asym", asym_loss),
        ("moe_gate", moe_gate_loss),
    ):
        if name in loss_sums:
            loss_sums[name] += value * batch_size


def apply_hard_distillation_mode(cfg, train_loader, val_loader):
    cfg.use_mixup = False
    cfg.use_sigma_jitter = False

    train_loader.dataset.augment_label = False
    train_loader.dataset.transform = val_loader.dataset.transform

    return DataLoader(
        train_loader.dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.type == 'cuda',
        collate_fn=train_loader.collate_fn,
        persistent_workers=cfg.num_workers > 0,
    )


def parse_selected_test_mae(result_text):
    patterns = (
        r"Selected_Test_MAE:\s*([\d.]+)",
        r"Final Test MAE(?:\s*\([^)]+\))?:\s*([\d.]+)",
        r"Test MAE:\s*([\d.]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, result_text)
        if match:
            return float(match.group(1))
    return None

# ==========================================
# 主训练函数
# ==========================================
def train(args):
    # Set seed first
    seed = args.seed
    seed_everything(seed)
    
    cfg = Config()
    
    # 🌟 CLI Overrides (Selection Space)
    if args.epochs is not None:
        cfg.epochs = args.epochs
        print(f"🔧 CLI Override: Epochs -> {cfg.epochs}")
        
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
        print(f"🔧 CLI Override: Batch Size -> {cfg.batch_size}")
        
    if args.split is not None:
        cfg.split_protocol = args.split
        print(f"🔧 CLI Override: Split Protocol -> {cfg.split_protocol}")
        
    if args.freeze is not None:
        cfg.freeze_backbone_epochs = args.freeze
        print(f"🔧 CLI Override: Freeze Epochs -> {cfg.freeze_backbone_epochs}")

    if getattr(args, "backbone_source", None) is not None:
        cfg.backbone_source = args.backbone_source
        print(f"🔧 CLI Override: Backbone Source -> {cfg.backbone_source}")

    if getattr(args, "backbone_name", None) is not None:
        cfg.backbone_name = args.backbone_name
        print(f"🔧 CLI Override: Backbone Name -> {cfg.backbone_name}")

    if getattr(args, "no_pretrained", False):
        cfg.backbone_pretrained = False
        print("🔧 CLI Override: Backbone Pretrained -> False")

    if getattr(args, "afad_dir", None) is not None:
        cfg.afad_dir = os.path.abspath(args.afad_dir)
        print(f"🔧 CLI Override: AFAD Dir -> {cfg.afad_dir}")

    if getattr(args, "experiment_tag", None) is not None:
        cfg.experiment_tag = args.experiment_tag
        print(f"🔧 CLI Override: Experiment Tag -> {cfg.experiment_tag}")

    if getattr(args, "split_file_tag", None) is not None:
        cfg.split_file_tag = args.split_file_tag
        print(f"🔧 CLI Override: Split File Tag -> {cfg.split_file_tag}")

    if getattr(args, "allow_legacy_split_upgrade", False):
        cfg.allow_legacy_split_upgrade = True
        print("🔧 CLI Override: Allow legacy split metadata upgrade -> True")

    # Ablation switches: map CLI args to config flags
    if getattr(args, 'use_ha', None) is not None:
        cfg.use_hybrid_attention = args.use_ha
        print(f"🔧 CLI Override: Hybrid Attention -> {cfg.use_hybrid_attention}")

    if getattr(args, 'use_dldl', None) is not None:
        cfg.use_dldl_v2 = args.use_dldl
        print(f"🔧 CLI Override: DLDL-v2 -> {cfg.use_dldl_v2}")

    if getattr(args, 'use_msff', None) is not None:
        cfg.use_multi_scale = args.use_msff
        print(f"🔧 CLI Override: MSFF -> {cfg.use_multi_scale}")

    if getattr(args, 'use_spp', None) is not None:
        cfg.use_spp = args.use_spp
        print(f"🔧 CLI Override: SPP -> {cfg.use_spp}")

    if getattr(args, 'use_mv', None) is not None:
        cfg.use_mv_loss = args.use_mv
        print(f"🔧 CLI Override: Mean-Variance Loss -> {cfg.use_mv_loss}")

    if getattr(args, 'use_texture', None) is not None:
        cfg.use_texture_branch = args.use_texture
        print(f"🔧 CLI Override: Texture Branch -> {cfg.use_texture_branch}")

    if getattr(args, 'use_triplet', None) is not None:
        cfg.use_adaptive_triplet = args.use_triplet
        print(f"🔧 CLI Override: Adaptive Triplet -> {cfg.use_adaptive_triplet}")

    if getattr(args, 'use_asym', None) is not None:
        cfg.use_asymmetric_ordinal = args.use_asym
        print(f"🔧 CLI Override: Asymmetric Ordinal -> {cfg.use_asymmetric_ordinal}")

    if getattr(args, 'use_freq', None) is not None:
        cfg.use_freq_attention = args.use_freq
        print(f"🔧 CLI Override: Freq Attention -> {cfg.use_freq_attention}")

    if getattr(args, 'use_moe', None) is not None:
        cfg.use_moe = args.use_moe
        print(f"🔧 CLI Override: MoE Head -> {cfg.use_moe}")

    # Backbone source notice
    if cfg.backbone_source == 'timm':
        print(f"Using timm backbone: {cfg.backbone_name} (not torchvision)")

    # 🌱 Easter Egg: Print Seed Meaning
    if seed in cfg.ACADEMIC_SEEDS:
        print(f"✨ Seed {seed}: {cfg.ACADEMIC_SEEDS[seed]}")

    max_train_batches = getattr(args, "max_train_batches", None)
    max_val_batches = getattr(args, "max_val_batches", None)
    max_test_batches = getattr(args, "max_test_batches", None)
    if max_train_batches is not None or max_val_batches is not None or max_test_batches is not None:
        if not getattr(cfg, "experiment_tag", None):
            cfg.experiment_tag = "smoke"
            print("🔧 Auto Experiment Tag -> smoke")
        print(
            "🧪 Smoke batch limits: "
            f"train={max_train_batches}, val={max_val_batches}, test={max_test_batches}. "
            "Do not report these metrics as paper results."
        )

    dldl_tools = DLDLProcessor(cfg)
    cfg.regularization_schedule = {
        "hard_distillation": hard_distillation_schedule_metadata(cfg.epochs),
    }
    
    # ==========================================
    # 2. 准备数据 (Stratified SOTA)
    # ==========================================
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(cfg)

    # 打印分布信息
    print(f"Dataset Size: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    # 1. 定义模型
    # model = LightweightAgeEstimator(num_classes=cfg.num_classes, dropout=cfg.dropout)
    # Updated for Ablation Support: Pass entire config
    model = LightweightAgeEstimator(cfg)
    model.to(cfg.device)

    effective_pretrained = getattr(model.backbone, 'pretrained_loaded', False)
    if cfg.backbone_pretrained and not effective_pretrained:
        print("⚠️ CRITICAL: Pretrained weights FAILED to load. Training with random initialization!")
        print("   Experiment metadata will record effective_pretrained=False")
    cfg.effective_pretrained = effective_pretrained
    training_metadata = build_training_metadata(cfg, seed)
    
    # 3. 初始化 EMA
    ema = None
    if getattr(cfg, 'use_ema', False):
        print(f"🔄 初始化 EMA (decay={cfg.ema_decay})")
        ema = EMAModel(model, decay=cfg.ema_decay)
        
    # 4. 损失函数 (Combined)
    criterion = CombinedLoss(cfg, weights=class_weights).to(cfg.device)
    
    # 5. 优化器 (Layer-wise Learning Rate)
    # 2027 Strategy: Backbone gets smaller LR (1e-5 range), Head gets normal LR (3e-4)
    backbone_params = []
    head_params = []
    
    # Iterate main model parameters
    # Note: 'model' is the LightweightAgeEstimator. 
    # model.backbone is the mobilenet.
    
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    backbone_lr = backbone_learning_rate(cfg.learning_rate, effective_pretrained)
    optimizer = optim.AdamW(
        [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': cfg.learning_rate}
        ],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    # ⚡ AMP Scaler
    scaler = make_grad_scaler(cfg.device.type)

    # 6. 调度器
    # Accelerated Decay: Reach min_lr at Epoch 100, then stay low for 20 epochs (Stable Phase)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=cfg.learning_rate * 0.01)
    scheduler_controller = SchedulerStepController(scheduler, max_epochs=100)

    # --- 断点续训逻辑 ---
    start_epoch = 0
    best_mae = float('inf')
    checkpoint_path = artifact_path(ROOT_DIR, "last_checkpoint", cfg, seed, ".pth")
    best_model_path = artifact_path(ROOT_DIR, "best_model", cfg, seed, ".pth")
    training_log_path = artifact_path(ROOT_DIR, "training_log", cfg, seed, ".csv")
    batch_log_path = artifact_path(ROOT_DIR, "batch_log", cfg, seed, ".csv")
    final_result_path = artifact_path(ROOT_DIR, "final_result", cfg, seed, ".txt")
    print(f"🎯 Target Checkpoint Name: {best_model_path}")
    print(f"🧾 Experiment ID: {training_metadata['experiment_id']}")
    resume_training = False

    should_resume = bool(getattr(args, "resume", False))
    if getattr(args, "fresh", False):
        should_resume = False

    if should_resume and os.path.exists(checkpoint_path):
        print(f"🔄 发现存档 '{checkpoint_path}'，正在恢复...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=cfg.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        mismatches = checkpoint_metadata_mismatches(checkpoint, training_metadata)
        if mismatches:
            raise RuntimeError(f"Checkpoint metadata mismatch; refusing to resume. {_format_metadata_mismatches(mismatches)}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 
        best_mae = checkpoint.get('best_mae', float('inf'))
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        restore_checkpoint_extra_state(checkpoint, scheduler_controller, scaler)
        
        # 恢复 EMA
        if ema and 'ema_state_dict' in checkpoint:
            ema.shadow = checkpoint['ema_state_dict']
            print("✅ EMA 状态已恢复")
            
        print(f"✅ 恢复成功！从 Epoch {start_epoch+1} 开始。最佳 MAE: {best_mae:.2f}")
        resume_training = True
    else:
        _guard_fresh_artifact_overwrite(
            (checkpoint_path, best_model_path, training_log_path, batch_log_path, final_result_path),
            training_metadata,
            cfg.device,
            allow_overwrite=bool(getattr(args, "overwrite_artifacts", False)),
        )
        if os.path.exists(checkpoint_path):
            print(f"🆕 Matching checkpoint exists but --resume was not set; starting fresh: {checkpoint_path}")
        else:
            print("🚀 开始全新训练...")

    # 初始化 Logger (Specific to seed)
    epoch_logger = CSVLogger(
        training_log_path,
        [
            'Epoch',
            'Train_Loss', 'Train_KL_Loss', 'Train_L1_Loss', 'Train_Rank_Loss', 'Train_MV_Loss', 'Train_Triplet_Loss', 'Train_Asym_Loss', 'Train_MoE_Gate_Loss',
            'Train_Mixup_MAE',
            'Val_MAE', 'LR', 'Time', 'Is_Best',
        ],
        resume=resume_training,
    )
    batch_logger = CSVLogger(
        batch_log_path,
        ['Epoch', 'Batch', 'Total_Loss', 'KL_Loss', 'L1_Loss', 'Rank_Loss', 'MV_Loss', 'Triplet_Loss', 'Asym_Loss', 'MoE_Gate_Loss'],
        resume=resume_training,
    )

    # 初始化 TensorBoard Writer
    log_dir = os.path.join(ROOT_DIR, "runs", f"{training_metadata['experiment_id']}_{int(time.time())}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📈 TensorBoard 日志目录: {log_dir}")

    print(f"设备: {cfg.device}")
    
    start_time = time.time()

    try:
        # 🌟 [Innovation] Freeze Backbone Strategy
        # Only train CA modules and Head for the first few epochs
        freeze_epochs = getattr(cfg, 'freeze_backbone_epochs', 0)
        if freeze_epochs > 0:
            if start_epoch < freeze_epochs:
                print(f"❄️  Freeze Strategy Enabled: Backbone will be frozen for first {freeze_epochs} epochs.")
                # Freeze all
                # 1. Freeze Backbone only (Keep Head/Adapters trainable)
                for param in model.backbone.parameters():
                    param.requires_grad = False

                # 2. Unfreeze CoordAtt modules inside backbone
                count_unfrozen = 0
                for name, module in model.backbone.named_modules():
                    if "CoordAtt" in str(type(module)):
                        for param in module.parameters():
                            param.requires_grad = True
                            count_unfrozen += 1
                
                print(f"    -> Backbone Frozen. {count_unfrozen} CoordAtt modules inside backbone Unfrozen.")
                print(f"    -> Head, SPP, and Fusion layers remain trainable.")
            else:
                print(f"❄️  Freeze Strategy Skipped: Resume Epoch {start_epoch+1} >= Freeze Limit {freeze_epochs}. Backbone remains unfrozen.")

        # 🛡️ Double Check for Safety
        first_param = next(model.backbone.parameters())
        print(f"🔍 检查 Backbone 状态: {'可训练' if first_param.requires_grad else '已冻结'}")

        hard_distill_start = hard_distillation_start_epoch(cfg.epochs)
        hard_distillation_applied = False
        for epoch in range(start_epoch, cfg.epochs):
            # 🌟 Unfreeze check
            if freeze_epochs > 0 and epoch == freeze_epochs:
                print(f"🔥 Unfreezing Backbone at Epoch {epoch+1} (Fine-tuning begins)...")
                for param in model.parameters():
                    param.requires_grad = True
                
                # Optional: Lower LR slightly? Or let cosine scheduler handle it.
                # Cosine is already decaying, so it's fine.

            # 🌟 [Online Hard Distillation] Disable Regularization at later stages
            if epoch >= hard_distill_start:
                # We use a 'Re-Loader' strategy to ensure worker processes (persistent_workers=True)
                # strictly receive the updated config and clean transforms on Windows.
                if not hard_distillation_applied:
                    print(f"🔥 [Epoch {epoch+1}] Hard Distillation Mode: Rebuilding DataLoader to flush persistent workers...")
                    train_loader = apply_hard_distillation_mode(cfg, train_loader, val_loader)
                    hard_distillation_applied = True
                
                if cfg.use_sigma_jitter:
                    # Fallback for dynamic safety
                    cfg.use_sigma_jitter = False

            # --- 1. 训练 ---
            model.train()
            train_loss = 0.0
            train_mae_sum = 0.0
            train_samples = 0
            train_batches = 0
            optimizer_stepped = False
            train_loss_sums = {"total": 0.0, "kl": 0.0, "l1": 0.0, "rank": 0.0, "mv": 0.0, "triplet": 0.0, "asym": 0.0, "moe_gate": 0.0}
            
            print(f"\nEpoch [{epoch+1}/{cfg.epochs}] Training (LR: {optimizer.param_groups[0]['lr']:.1e})...")
            
            for batch_idx, (images, target_dists, true_ages) in enumerate(train_loader):
                if images.numel() == 0:
                    print(f"⚠️ Skipping empty training batch at index {batch_idx}")
                    continue

                images = images.to(cfg.device)
                target_dists = target_dists.to(cfg.device)
                true_ages = true_ages.to(cfg.device)
                
                # MixUp
                if cfg.use_mixup and np.random.random() < cfg.mixup_prob:
                    images, target_dists, true_ages = mixup_data(
                        images, target_dists, true_ages, alpha=cfg.mixup_alpha
                    )
                
                optimizer.zero_grad()
                
                # ⚡ AMP Forward
                with make_amp_context(cfg.device.type):
                    logits, embeddings, extras = model_forward_for_loss(model, cfg, images)
                    log_probs = F.log_softmax(logits, dim=1)
                    
                    # 计算 Combined Loss
                    loss, loss_kl, loss_l1, loss_rank, loss_mv, loss_triplet, loss_asym, loss_moe_gate = criterion(
                        log_probs, target_dists, true_ages, logits, embeddings=embeddings, extras=extras
                    )
                
                # ⚡ AMP Backward
                scaler.scale(loss).backward()
                
                # Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                step_skipped = amp_step_was_skipped(scale_before, scaler.get_scale())
                optimizer_stepped = optimizer_stepped or not step_skipped
                
                # 更新 EMA
                if ema and not step_skipped:
                    ema.update()
                    
                loss_value = loss.item()
                current_batch_size = true_ages.size(0)
                train_loss += loss_value * current_batch_size
                train_batches += 1
                _accumulate_loss_components(
                    train_loss_sums,
                    current_batch_size,
                    loss_value,
                    loss_kl,
                    loss_l1,
                    loss_rank,
                    loss_mv,
                    loss_triplet,
                    loss_asym,
                    loss_moe_gate,
                )
                
                # 计算 MAE (Monitor)
                with torch.no_grad():
                    probs = F.softmax(logits, dim=1)
                    pred_ages = dldl_tools.expectation_regression(probs)
                    train_mae_sum += torch.sum(torch.abs(pred_ages - true_ages)).item()
                    train_samples += current_batch_size
                
                if (batch_idx + 1) % 100 == 0:
                    print(
                        f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss_value:.4f} "
                        f"(KL={loss_kl:.3f}, L1={loss_l1:.3f}, Rank={loss_rank:.3f}, "
                        f"MV={loss_mv:.3f}, Triplet={loss_triplet:.3f}, Asym={loss_asym:.3f}, MoE={loss_moe_gate:.3f})"
                    )
                
                if batch_idx % 10 == 0:
                    batch_logger.log([epoch + 1, batch_idx, loss_value, loss_kl, loss_l1, loss_rank, loss_mv, loss_triplet, loss_asym, loss_moe_gate])
                    
                    # 📈 TensorBoard Logging (Step)
                    global_step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('Train/Loss_Total', loss_value, global_step)
                    writer.add_scalar('Train/Loss_KL', loss_kl, global_step)
                    writer.add_scalar('Train/Loss_L1', loss_l1, global_step)
                    writer.add_scalar('Train/Loss_Rank', loss_rank, global_step)
                    writer.add_scalar('Train/Loss_MV', loss_mv, global_step)
                    writer.add_scalar('Train/Loss_Triplet', loss_triplet, global_step)
                    writer.add_scalar('Train/Loss_Asym', loss_asym, global_step)
                    writer.add_scalar('Train/Loss_MoE_Gate', loss_moe_gate, global_step)
                    writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)

                if max_train_batches is not None and train_batches >= max_train_batches:
                    print(f"🧪 Reached max_train_batches={max_train_batches}; ending training loop for this epoch.")
                    break
                
            if train_samples == 0:
                raise RuntimeError("No valid training samples were loaded in this epoch.")

            avg_train_loss = train_loss / train_samples
            avg_train_components = _average_loss_sums(train_loss_sums, train_samples)
            avg_train_mae = train_mae_sum / train_samples
            
            # --- 2. 验证 (Validation) ---
            # 如果使用了 EMA，验证时应该使用 EMA 的权重
            if ema:
                ema.apply_shadow()
                print("🛡️切换到 EMA 权重进行验证...")
                
            model.eval()
            mae_sum = 0.0
            total_samples = 0
            val_batches = 0
            
            with torch.no_grad():
                for images, target_dists, true_ages in val_loader:
                    if images.numel() == 0:
                        print("⚠️ Skipping empty validation batch")
                        continue

                    images = images.to(cfg.device)
                    target_dists = target_dists.to(cfg.device)
                    true_ages = true_ages.to(cfg.device)
                    
                    # Selection metric: multi-scale TTA MAE only.
                    probs = predict_probs(model, images, mode="multi", base_size=cfg.img_size)
                    pred_ages = probs_to_ages(probs, cfg.num_classes)
                    mae_sum += torch.sum(torch.abs(pred_ages - true_ages)).item()
                    total_samples += true_ages.size(0)
                    val_batches += 1

                    if max_val_batches is not None and val_batches >= max_val_batches:
                        print(f"🧪 Reached max_val_batches={max_val_batches}; ending validation loop.")
                        break
            
            # 验证结束，如果用了 EMA，恢复原始权重以便继续训练
            if ema:
                ema.restore()
                print("🛡️恢复原始权重继续训练...")
                
            if total_samples == 0:
                raise RuntimeError("No valid validation samples were loaded.")

            val_mae = mae_sum / total_samples
            
            print(f"Epoch [{epoch+1}/{cfg.epochs}] | "
                  f"T_Loss: {avg_train_loss:.4f} | T_Mixup_MAE: {avg_train_mae:.2f} | "
                  f"V_MAE: {val_mae:.2f}")

            # --- 3. 保存最佳模型 ---
            is_best = False
            if val_mae < best_mae:
                print(f"🏆 新纪录！MAE {best_mae:.2f} -> {val_mae:.2f}")
                best_mae = val_mae
                is_best = True
                
                # 如果用了 EMA，保存 EMA 后的权重为 best_model.pth
                if ema:
                    ema.apply_shadow()
                    try:
                        save_model_package(model, best_model_path, training_metadata)
                    finally:
                        ema.restore()
                else:
                    save_model_package(model, best_model_path, training_metadata)
                    
            # Logging
            current_lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            epoch_logger.log([
                epoch + 1,
                avg_train_loss,
                avg_train_components["kl"],
                avg_train_components["l1"],
                avg_train_components["rank"],
                avg_train_components["mv"],
                avg_train_components["triplet"],
                avg_train_components["asym"],
                avg_train_components["moe_gate"],
                avg_train_mae,
                val_mae,
                current_lr,
                elapsed,
                int(is_best),
            ])
            
            # 📈 TensorBoard Logging (Epoch)
            writer.add_scalar('Epoch/Train_Loss', avg_train_loss, epoch + 1)
            writer.add_scalar('Epoch/Train_KL_Loss', avg_train_components["kl"], epoch + 1)
            writer.add_scalar('Epoch/Train_L1_Loss', avg_train_components["l1"], epoch + 1)
            writer.add_scalar('Epoch/Train_Rank_Loss', avg_train_components["rank"], epoch + 1)
            writer.add_scalar('Epoch/Train_MV_Loss', avg_train_components["mv"], epoch + 1)
            writer.add_scalar('Epoch/Train_Triplet_Loss', avg_train_components["triplet"], epoch + 1)
            writer.add_scalar('Epoch/Train_Asym_Loss', avg_train_components["asym"], epoch + 1)
            writer.add_scalar('Epoch/Train_MoE_Gate_Loss', avg_train_components["moe_gate"], epoch + 1)
            writer.add_scalar('Epoch/Train_Mixup_MAE', avg_train_mae, epoch + 1)
            writer.add_scalar('Epoch/Val_MAE', val_mae, epoch + 1)
            
            scheduler_steps = scheduler_controller.step_epoch(epoch, optimizer_stepped)
            if epoch < scheduler_controller.max_epochs and scheduler_steps == 0:
                print("⚠️ Scheduler step deferred because AMP skipped all optimizer steps this epoch.")
            
            # 保存断点
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 
                'best_mae': best_mae,
                'metadata': training_metadata,
                **checkpoint_extra_state(scheduler_controller, scaler),
            }
            if ema:
                checkpoint_dict['ema_state_dict'] = ema.shadow
                
            save_checkpoint(checkpoint_dict, filename=checkpoint_path)
            
            # --- Manual SWA Strategy ---
            # Save checkpoints for the last 10 epochs
            if epoch >= cfg.epochs - 10:
                swa_filename = artifact_path(ROOT_DIR, f"checkpoint_epoch_{epoch+1}", cfg, seed, ".pth")
                print(f"💾 Saving SWA Checkpoint: {swa_filename}")
                save_checkpoint(checkpoint_dict, filename=swa_filename)
        

        # ==========================================
        # 🏁 Final Test Set Evaluation
        # ==========================================
        print("\n" + "="*50)
        print("🏁 Final Evaluation on TEST SET")
        print("="*50)
        
        # Load Best Model
        if os.path.exists(best_model_path):
            state_dict, checkpoint = load_model_state_package(best_model_path, cfg.device)
            mismatches = checkpoint_metadata_mismatches(checkpoint, training_metadata)
            if mismatches:
                raise RuntimeError(f"Best model metadata mismatch; refusing final evaluation. {_format_metadata_mismatches(mismatches)}")
            model.load_state_dict(state_dict)
            print(f"📂 Loaded Best Model from {best_model_path}")
        
        test_metrics = evaluate_mae(model, test_loader, cfg, cfg.device, modes=TTA_MODES, max_batches=max_test_batches)
        selected_tta = training_metadata["selection_metric"]["tta"]
        final_test_mae = test_metrics[selected_tta]
        print(f"🏆 Final Test MAE ({selected_tta}): {final_test_mae:.4f}")
        for mode in TTA_MODES:
            print(f"   MAE_{mode}: {test_metrics[mode]:.4f}")
        
        # Save Final Result
        with open(final_result_path, "w") as f:
            f.write(f"MAE_raw: {test_metrics['raw']:.4f}\n")
            f.write(f"MAE_flip: {test_metrics['flip']:.4f}\n")
            f.write(f"MAE_multi: {test_metrics['multi']:.4f}\n")
            f.write(f"Selected_TTA: {selected_tta}\n")
            f.write(f"Selected_Test_MAE: {final_test_mae:.4f}\n")
            f.write(f"Experiment ID: {training_metadata['experiment_id']}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Split_Protocol: {training_metadata.get('split_protocol', 'N/A')}\n")
            f.write(f"Split_File: {training_metadata.get('split_file', 'N/A')}\n")
            f.write(f"Split_Fingerprint: {training_metadata.get('split_fingerprint', 'N/A')}\n")
            f.write(f"Dataset_Fingerprint: {training_metadata.get('dataset_fingerprint', 'N/A')}\n")
            f.write(f"Pretrained_Requested: {training_metadata['backbone']['pretrained']}\n")
            f.write(f"Pretrained_Loaded: {training_metadata['backbone'].get('effective_pretrained', None)}\n")
            
    finally:
        writer.close()

if __name__ == "__main__":
    import sys
    import subprocess

    # Helper function for Batch Mode (Run All)
    def run_training_subprocess(seed):
        print(f"\n🚀 Starting subprocess for seed {seed}...")
        # Use sys.executable to ensure we use the same python interpreter
        # Use sys.argv[0] to refer to this script (train.py)
        cmd = [sys.executable, sys.argv[0], "--seed", str(seed), "--split_file_tag", "formal_v1"]
        
        # Pass through other common args if needed, or enforce defaults for benchmarks
        # For 'Run All', we usually want standard settings, so we just pass seed.
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        
        mae = None
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                if "Final Test MAE" in output:
                    mae = parse_selected_test_mae(output)
        
        rc = process.poll()
        if rc != 0:
            print(f"❌ Training failed for seed {seed}")
            return None
            
        # Fallback check file
        if mae is None:
            fallback_cfg = Config()
            fallback_cfg.split_file_tag = "formal_v1"
            result_candidates = [
                artifact_path(ROOT_DIR, "final_result", fallback_cfg, seed, ".txt"),
                os.path.join(ROOT_DIR, f"final_result_seed{seed}.txt"),
            ]
            for result_file in result_candidates:
                if not os.path.exists(result_file):
                    continue
                with open(result_file, 'r') as f:
                    mae = parse_selected_test_mae(f.read())
                if mae is not None:
                    break
        return mae

    # --- CLI Handling ---
    # Case 1: Arguments provided -> Run Training Immediately
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="FADE-Net Training Launcher")
        parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
        parser.add_argument('--epochs', type=int, help='Override total training epochs')
        parser.add_argument('--batch_size', type=int, help='Override batch size')
        parser.add_argument('--split', type=str, choices=['80-10-10', '90-5-5', '72-8-20'], help="Select Split Protocol")
        parser.add_argument('--freeze', type=int, dest='freeze', help='Override backbone freeze epochs')
        parser.add_argument('--freeze_backbone_epochs', type=int, dest='freeze_alias', help='Alias for --freeze')
        parser.add_argument('--resume', dest='resume', action='store_true', help='Resume only if checkpoint metadata matches')
        parser.add_argument('--fresh', dest='fresh', action='store_true', help='Ignore existing matching checkpoint')
        parser.add_argument('--overwrite_artifacts', action='store_true', help='Allow a fresh run to overwrite existing artifacts for the same experiment id')
        parser.add_argument('--backbone_source', type=str, choices=['torchvision', 'timm'], help='Backbone provider')
        parser.add_argument('--backbone_name', type=str, help='Backbone model name')
        parser.add_argument('--no_pretrained', action='store_true', help='Disable pretrained backbone weights')
        parser.add_argument('--afad_dir', type=str, help='Override AFAD dataset directory')
        parser.add_argument('--allow_legacy_split_upgrade', action='store_true', help='Trust and stamp a legacy split after size/index validation')
        parser.add_argument('--max_train_batches', type=int, help='Limit valid train batches for pipeline smoke tests')
        parser.add_argument('--max_val_batches', type=int, help='Limit valid validation batches for pipeline smoke tests')
        parser.add_argument('--max_test_batches', type=int, help='Limit valid test batches for pipeline smoke tests')
        parser.add_argument('--experiment_tag', type=str, help='Append tag to experiment id for smoke or side runs')
        parser.add_argument('--split_file_tag', type=str, help='Append tag to split filename and experiment id for isolated formal reruns')

        # --- Ablation Switches (消融实验开关) ---
        parser.add_argument('--ha', action='store_true', help='Enable Hybrid Attention (CoordAtt)')
        parser.add_argument('--no-ha', dest='ha_false', action='store_true', help='Disable Hybrid Attention')
        parser.add_argument('--dldl', action='store_true', help='Enable DLDL-v2')
        parser.add_argument('--no-dldl', dest='dldl_false', action='store_true', help='Disable DLDL-v2')
        parser.add_argument('--msff', action='store_true', help='Enable Multi-Scale Feature Fusion')
        parser.add_argument('--no-msff', dest='msff_false', action='store_true', help='Disable MSFF')
        parser.add_argument('--spp', action='store_true', help='Enable Spatial Pyramid Pooling')
        parser.add_argument('--no-spp', dest='spp_false', action='store_true', help='Disable SPP')
        parser.add_argument('--mv', action='store_true', help='Enable Mean-Variance Loss')
        parser.add_argument('--no-mv', dest='mv_false', action='store_true', help='Disable MV Loss')
        parser.add_argument('--texture', action='store_true', help='Enable texture enhancement branch')
        parser.add_argument('--no-texture', dest='texture_false', action='store_true', help='Disable texture branch')
        parser.add_argument('--triplet', action='store_true', help='Enable adaptive triplet loss')
        parser.add_argument('--no-triplet', dest='triplet_false', action='store_true', help='Disable adaptive triplet loss')
        parser.add_argument('--asym', action='store_true', help='Enable asymmetric ordinal loss')
        parser.add_argument('--no-asym', dest='asym_false', action='store_true', help='Disable asymmetric ordinal loss')
        parser.add_argument('--freq', action='store_true', help='Enable frequency-domain attention')
        parser.add_argument('--no-freq', dest='freq_false', action='store_true', help='Disable frequency-domain attention')
        parser.add_argument('--moe', action='store_true', help='Enable Mixture of Experts head')
        parser.add_argument('--no-moe', dest='moe_false', action='store_true', help='Disable MoE head')

        args = parser.parse_args()

        # Handle alias
        if args.freeze_alias is not None:
            args.freeze = args.freeze_alias

        # Handle ablation flags (only override if explicitly set)
        if args.ha_false:
            args.use_ha = False
        elif args.ha:
            args.use_ha = True
        else:
            args.use_ha = None  # Use config default

        if args.dldl_false:
            args.use_dldl = False
        elif args.dldl:
            args.use_dldl = True
        else:
            args.use_dldl = None

        if args.msff_false:
            args.use_msff = False
        elif args.msff:
            args.use_msff = True
        else:
            args.use_msff = None

        if args.spp_false:
            args.use_spp = False
        elif args.spp:
            args.use_spp = True
        else:
            args.use_spp = None

        if args.mv_false:
            args.use_mv = False
        elif args.mv:
            args.use_mv = True
        else:
            args.use_mv = None

        if args.texture_false:
            args.use_texture = False
        elif args.texture:
            args.use_texture = True
        else:
            args.use_texture = None

        if args.triplet_false:
            args.use_triplet = False
        elif args.triplet:
            args.use_triplet = True
        else:
            args.use_triplet = None

        if args.asym_false:
            args.use_asym = False
        elif args.asym:
            args.use_asym = True
        else:
            args.use_asym = None

        if args.freq_false:
            args.use_freq = False
        elif args.freq:
            args.use_freq = True
        else:
            args.use_freq = None

        if args.moe_false:
            args.use_moe = False
        elif args.moe:
            args.use_moe = True
        else:
            args.use_moe = None

        # cudnn.benchmark omitted: seed_everything() sets it to False anyway
        train(args)
        sys.exit(0)

    # Case 2: No Arguments -> Interactive Menu
    print("="*60)
    print("🎮 FADE-Net Interactive Training Launcher")
    print("="*60)
    print("1. [Default]  Run Standard Benchmark (Seed 42, 72-8-20, formal_v1)")
    print("2. [Seed]     Run 2026 Academic Seed (Seed 2026, 72-8-20, formal_v1)")
    print("3. [Batch]    Run Academic Seeds (42, 3407, 2026, formal_v1)")
    print("4. [Custom]   Configure Manually")
    print("q. [Quit]     Exit")
    print("-" * 60)
    
    try:
        choice = input("👉 Select mode [1-4/q]: ").strip().lower()
        
        if choice == '1' or choice == '':
            print("\n🚀 Selected: Standard Benchmark (Seed 42)")
            # Simulate args
            class Args:
                seed = 42
                epochs = None
                batch_size = None
                split = None
                freeze = None
                resume = False
                fresh = False
                overwrite_artifacts = False
                backbone_source = None
                backbone_name = None
                no_pretrained = False
                afad_dir = None
                allow_legacy_split_upgrade = False
                max_train_batches = None
                max_val_batches = None
                max_test_batches = None
                experiment_tag = None
                split_file_tag = "formal_v1"
            train(Args())
            
        elif choice == '2':
            print("\n🚀 Selected: Academic Seed 2026")
            class Args:
                seed = 2026
                epochs = None
                batch_size = None
                split = None
                freeze = None
                resume = False
                fresh = False
                overwrite_artifacts = False
                backbone_source = None
                backbone_name = None
                no_pretrained = False
                afad_dir = None
                allow_legacy_split_upgrade = False
                max_train_batches = None
                max_val_batches = None
                max_test_batches = None
                experiment_tag = None
                split_file_tag = "formal_v1"
            train(Args())

        elif choice == '3':
            print("\n🚀 Selected: Run All Academic Seeds")
            seeds = [42, 3407, 2026]
            results = {}
            for s in seeds:
                mae = run_training_subprocess(s)
                if mae is not None:
                    results[s] = mae
            
            print("\n" + "=" * 60)
            print("📊 Final Batch Report")
            print("=" * 60)
            if results:
                maes = list(results.values())
                mean_mae = np.mean(maes)
                std_mae = np.std(maes)
                print(f"{'Seed':<10} | {'Test MAE':<10}")
                print("-" * 25)
                for s, m in results.items():
                    print(f"{s:<10} | {m:.4f}")
                print("-" * 25)
                print(f"\n🏆 Average Test MAE: {mean_mae:.4f} ± {std_mae:.4f}")
            else:
                print("No successful runs.")

        elif choice == '4':
            print("\n🔧 Custom Configuration Mode:")
            s = input("   - Seed [42]: ").strip() or '42'
            sp_choice = input("   - Split (1: 72-8-20, 2: 80-10-10, 3: 90-5-5) [1]: ").strip()
            if sp_choice == '2':
                split = '80-10-10'
            elif sp_choice == '3':
                split = '90-5-5'
            else:
                split = '72-8-20'
            ep = input("   - Epochs [Default]: ").strip()
            fz = input("   - Freeze Epochs [Default]: ").strip()
            
            train(argparse.Namespace(
                seed=int(s),
                split=split,
                epochs=int(ep) if ep else None,
                batch_size=None,
                freeze=int(fz) if fz else None,
                resume=False,
                fresh=False,
                overwrite_artifacts=False,
                backbone_source=None,
                backbone_name=None,
                no_pretrained=False,
                afad_dir=None,
                allow_legacy_split_upgrade=False,
                max_train_batches=None,
                max_val_batches=None,
                max_test_batches=None,
                experiment_tag=None,
                split_file_tag="formal_v1",
            ))
            
        elif choice == 'q':
            pass
            
    except KeyboardInterrupt:
        print("\n👋 Exiting.")
        sys.exit(0)


