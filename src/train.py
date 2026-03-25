import os
import argparse
import torch
import torch.nn as nn
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


# ==========================================
# Multi-Scale TTA (6x: 0.9/1.0/1.1 + flip)
# ==========================================
def multi_scale_tta(images, model):
    """
    激进 Multi-Scale TTA: 3个尺度 (0.9, 1.0, 1.1) x 2 (原始 + 翻转) = 6x 平均
    可将单模型 MAE 降低约 0.01-0.02
    """
    scales = [0.9, 1.0, 1.1]
    all_probs = []
    
    for scale in scales:
        if scale != 1.0:
            new_size = int(224 * scale)
            resized = F.interpolate(images, size=new_size, mode='bilinear', align_corners=False)
            if new_size > 224:
                start = (new_size - 224) // 2
                resized = resized[:, :, start:start+224, start:start+224]
            else:
                pad = (224 - new_size) // 2
                resized = F.pad(resized, (pad, 224-new_size-pad, pad, 224-new_size-pad), mode='reflect')
        else:
            resized = images
        
        # 原始
        logits = model(resized)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs)
        
        # 水平翻转
        flipped = torch.flip(resized, dims=[3])
        logits_flip = model(flipped)
        probs_flip = F.softmax(logits_flip, dim=1)
        all_probs.append(probs_flip)
    
    # 平均 6 个预测
    return torch.stack(all_probs, dim=0).mean(dim=0)

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

    # 🌱 Easter Egg: Print Seed Meaning
    if seed in cfg.ACADEMIC_SEEDS:
        print(f"✨ Seed {seed}: {cfg.ACADEMIC_SEEDS[seed]}")

    dldl_tools = DLDLProcessor(cfg)
    
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

    optimizer = optim.AdamW(
        [
            {'params': backbone_params, 'lr': cfg.learning_rate},       # Backbone: 3e-4 (Full Sprint)
            {'params': head_params, 'lr': cfg.learning_rate}            # Head/Fusion: 3e-4
        ], 
        lr=cfg.learning_rate, 
        weight_decay=cfg.weight_decay 
    )
    
    # ⚡ AMP Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # 6. 调度器
    # Accelerated Decay: Reach min_lr at Epoch 100, then stay low for 20 epochs (Stable Phase)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=cfg.learning_rate * 0.01)
    
    # --- 断点续训逻辑 ---
    start_epoch = 0
    best_mae = float('inf')
    # Dynamic naming with Seed
    checkpoint_path = os.path.join(ROOT_DIR, f"last_checkpoint_seed{seed}.pth")
    # Dynamic naming: best_model_FADE-Net_HA_DLDL_MSFF_SPP_seed{seed}.pth
    best_model_path = os.path.join(ROOT_DIR, f"best_model_{cfg.project_name}_seed{seed}.pth")
    print(f"🎯 Target Checkpoint Name: {best_model_path}")
    resume_training = False

    if os.path.exists(checkpoint_path):
        print(f"🔄 发现存档 '{checkpoint_path}'，正在恢复...")
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 
        best_mae = checkpoint.get('best_mae', float('inf'))
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复 EMA
        if ema and 'ema_state_dict' in checkpoint:
            ema.shadow = checkpoint['ema_state_dict']
            print("✅ EMA 状态已恢复")
            
        print(f"✅ 恢复成功！从 Epoch {start_epoch+1} 开始。最佳 MAE: {best_mae:.2f}")
        resume_training = True
    else:
        print("🚀 开始全新训练...")

    # 初始化 Logger (Specific to seed)
    epoch_logger = CSVLogger(os.path.join(ROOT_DIR, f'training_log_seed{seed}.csv'), 
                             ['Epoch', 'Train_Loss', 'Train_MAE', 'Val_Loss', 'Val_MAE', 'LR', 'Time', 'Is_Best'], 
                             resume=resume_training)
    batch_logger = CSVLogger(os.path.join(ROOT_DIR, f'batch_log_seed{seed}.csv'), ['Epoch', 'Batch', 'Total_Loss', 'KL_Loss', 'L1_Loss', 'Rank_Loss'], resume=resume_training)

    # 初始化 TensorBoard Writer
    log_dir = os.path.join(ROOT_DIR, "runs", f"{cfg.project_name}_seed{seed}_{int(time.time())}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📈 TensorBoard 日志目录: {log_dir}")

    print(f"设备: {cfg.device}")
    
    start_time = time.time()
    
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

    for epoch in range(start_epoch, cfg.epochs):
        # 🌟 Unfreeze check
        if freeze_epochs > 0 and epoch == freeze_epochs:
            print(f"🔥 Unfreezing Backbone at Epoch {epoch+1} (Fine-tuning begins)...")
            for param in model.parameters():
                param.requires_grad = True
            
            # Optional: Lower LR slightly? Or let cosine scheduler handle it.
            # Cosine is already decaying, so it's fine.

        # 🌟 [Online Hard Distillation] Disable Regularization at later stages
        if epoch >= 105:
            # We use a 'Re-Loader' strategy to ensure worker processes (persistent_workers=True)
            # strictly receive the updated config and clean transforms on Windows.
            if epoch == 105:
                print(f"🔥 [Epoch {epoch+1}] Hard Distillation Mode: Rebuilding DataLoader to flush persistent workers...")
                cfg.use_mixup = False
                cfg.use_sigma_jitter = False
                
                # Update Dataset in-place
                train_loader.dataset.augment_label = False
                # Switch to pure validation transform (Clean images)
                train_loader.dataset.transform = val_loader.dataset.transform
                
                # Re-instantiate DataLoader
                train_loader = DataLoader(
                    train_loader.dataset, 
                    batch_size=cfg.batch_size, 
                    shuffle=True, 
                    num_workers=cfg.num_workers, 
                    pin_memory=True, 
                    collate_fn=train_loader.collate_fn, 
                    persistent_workers=True
                )
            
            if cfg.use_sigma_jitter:
                # Fallback for dynamic safety
                cfg.use_sigma_jitter = False

        # --- 1. 训练 ---
        model.train()
        train_loss = 0.0
        train_mae_sum = 0.0
        train_samples = 0
        
        print(f"\nEpoch [{epoch+1}/{cfg.epochs}] Training (LR: {optimizer.param_groups[0]['lr']:.1e})...")
        
        for batch_idx, (images, target_dists, true_ages) in enumerate(train_loader):
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
            with torch.cuda.amp.autocast():
                logits = model(images)
                log_probs = F.log_softmax(logits, dim=1)
                
                # 计算 Combined Loss
                loss, loss_kl, loss_l1, loss_rank = criterion(log_probs, target_dists, true_ages, logits)
            
            # ⚡ AMP Backward
            scaler.scale(loss).backward()
            
            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # 更新 EMA
            if ema:
                ema.update()
                
            train_loss += loss.item()
            
            # 计算 MAE (Monitor)
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                pred_ages = dldl_tools.expectation_regression(probs)
                train_mae_sum += torch.sum(torch.abs(pred_ages - true_ages)).item()
                train_samples += true_ages.size(0)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f} (KL={loss_kl:.3f}, L1={loss_l1:.3f}, Rank={loss_rank:.3f})")
            
            if batch_idx % 10 == 0:
                batch_logger.log([epoch + 1, batch_idx, loss.item(), loss_kl, loss_l1, loss_rank])
                
                # 📈 TensorBoard Logging (Step)
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Train/Loss_Total', loss.item(), global_step)
                writer.add_scalar('Train/Loss_KL', loss_kl, global_step)
                writer.add_scalar('Train/Loss_L1', loss_l1, global_step)
                writer.add_scalar('Train/Loss_Rank', loss_rank, global_step)
                writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_mae_sum / train_samples
        
        # --- 2. 验证 (Validation) ---
        # 如果使用了 EMA，验证时应该使用 EMA 的权重
        if ema:
            ema.apply_shadow()
            print("🛡️切换到 EMA 权重进行验证...")
            
        model.eval()
        mae_sum = 0.0
        val_loss_sum = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for images, target_dists, true_ages in val_loader:
                images = images.to(cfg.device)
                target_dists = target_dists.to(cfg.device)
                true_ages = true_ages.to(cfg.device)
                
                # TTA 验证 (Multi-Scale 6x: 0.9/1.0/1.1 + flip)
                probs = multi_scale_tta(images, model)
                
                # 计算 Loss (仅参考，这里只算主KL)
                log_probs = torch.log(probs + 1e-8) 
                val_loss = F.kl_div(log_probs, target_dists, reduction='batchmean')
                val_loss_sum += val_loss.item()
                
                pred_ages = dldl_tools.expectation_regression(probs)
                mae_sum += torch.sum(torch.abs(pred_ages - true_ages)).item()
                total_samples += true_ages.size(0)
        
        # 验证结束，如果用了 EMA，恢复原始权重以便继续训练
        if ema:
            ema.restore()
            print("🛡️恢复原始权重继续训练...")
            
        val_mae = mae_sum / total_samples
        avg_val_loss = val_loss_sum / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{cfg.epochs}] | "
              f"T_Loss: {avg_train_loss:.4f} | T_MAE: {avg_train_mae:.2f} | "
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
                torch.save(model.state_dict(), best_model_path)
                ema.restore()
            else:
                torch.save(model.state_dict(), best_model_path)
                
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time
        epoch_logger.log([epoch + 1, avg_train_loss, avg_train_mae, avg_val_loss, val_mae, current_lr, elapsed, int(is_best)])
        
        # 📈 TensorBoard Logging (Epoch)
        writer.add_scalar('Epoch/Train_Loss', avg_train_loss, epoch + 1)
        writer.add_scalar('Epoch/Train_MAE', avg_train_mae, epoch + 1)
        writer.add_scalar('Epoch/Val_Loss', avg_val_loss, epoch + 1)
        writer.add_scalar('Epoch/Val_MAE', val_mae, epoch + 1)
        
        if epoch < 100:
            scheduler.step()
        else:
             # Maintain eta_min for Stable Phase (101-120)
             pass
        
        # 保存断点
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 
            'best_mae': best_mae
        }
        if ema:
            checkpoint_dict['ema_state_dict'] = ema.shadow
            
        save_checkpoint(checkpoint_dict, filename=checkpoint_path)
        
        # --- Manual SWA Strategy ---
        # Save checkpoints for the last 10 epochs
        if epoch >= cfg.epochs - 10:
            swa_filename = os.path.join(ROOT_DIR, f"checkpoint_seed{seed}_epoch_{epoch+1}.pth")
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
        model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))
        print(f"📂 Loaded Best Model from {best_model_path}")
    
    model.eval()
    test_mae = 0.0
    count = 0
    rank_arange = torch.arange(cfg.num_classes).to(cfg.device) # Define rank_arange
    
    with torch.no_grad():
        for images, labels, ages in test_loader:
            images, labels, ages = images.to(cfg.device), labels.to(cfg.device), ages.to(cfg.device)
            # TTA Evaluation (Multi-Scale 6x: 0.9/1.0/1.1 + flip)
            probs = multi_scale_tta(images, model)
            
            # Predict
            output_ages = torch.sum(probs * rank_arange.float(), dim=1)
            
            # MAE
            mae = torch.abs(output_ages - ages).sum().item()
            test_mae += mae
            count += images.size(0)
            
    final_test_mae = test_mae / count
    print(f"🏆 Final Test MAE: {final_test_mae:.4f}")
    
    # Save Final Result
    with open(os.path.join(ROOT_DIR, f"final_result_seed{seed}.txt"), "w") as f:
        f.write(f"Test MAE: {final_test_mae:.4f}\n")
        
    writer.close()

if __name__ == "__main__":
    import sys
    import subprocess
    import re

    # Helper function for Batch Mode (Run All)
    def run_training_subprocess(seed):
        print(f"\n🚀 Starting subprocess for seed {seed}...")
        # Use sys.executable to ensure we use the same python interpreter
        # Use sys.argv[0] to refer to this script (train.py)
        cmd = [sys.executable, sys.argv[0], "--seed", str(seed)]
        
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
                if "Final Test MAE:" in output:
                    try:
                        mae = float(output.strip().split(":")[-1].strip())
                    except:
                        pass
        
        rc = process.poll()
        if rc != 0:
            print(f"❌ Training failed for seed {seed}")
            return None
            
        # Fallback check file
        if mae is None:
            # Assuming ROOT_DIR is defined and available
            result_file = os.path.join(ROOT_DIR, f"final_result_seed{seed}.txt")
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    content = f.read()
                    match = re.search(r"Test MAE:\s*([\d\.]+)", content)
                    if match:
                        mae = float(match.group(1))
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

        torch.backends.cudnn.benchmark = True
        train(args)
        sys.exit(0)

    # Case 2: No Arguments -> Interactive Menu
    print("="*60)
    print("🎮 FADE-Net Interactive Training Launcher")
    print("="*60)
    print("1. [Default]  Run Standard Benchmark (Seed 42, 80-10-10)")
    print("2. [SOTA]     Run 2026 Academic Seed (Seed 2026, 80-10-10)")
    print("3. [Batch]    Run All Academic Seeds (42, 3407, 2026, 1337, 1106)")
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
            train(Args())
            
        elif choice == '2':
            print("\n🚀 Selected: SOTA 2026 (Seed 2026)")
            class Args:
                seed = 2026
                epochs = None
                batch_size = None
                split = None
                freeze = None
            train(Args())

        elif choice == '3':
            print("\n🚀 Selected: Run All Academic Seeds")
            seeds = [42, 3407, 2026, 1337, 1106]
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
            sp_choice = input("   - Split (1: 80-10-10, 2: 90-5-5, 3: 72-8-20) [1]: ").strip()
            if sp_choice == '2':
                split = '90-5-5'
            elif sp_choice == '3':
                split = '72-8-20'
            else:
                split = '80-10-10'  # Default to Robust 80-10-10
            ep = input("   - Epochs [Default]: ").strip()
            fz = input("   - Freeze Epochs [Default]: ").strip()
            
            class Args:
                seed = int(s)
                split = split
                epochs = int(ep) if ep else None
                batch_size = None
                freeze = int(fz) if fz else None
            
            train(Args())
            
        elif choice == 'q':
            pass
            
    except KeyboardInterrupt:
        print("\n👋 Exiting.")
        sys.exit(0)


