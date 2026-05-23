import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import argparse
import glob

# ================= 配置区域 =================
SAVE_DIR = 'plots'
plt.style.use('seaborn-v0_8-paper')

# 统一的论文级字体配置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 2.5 
})

def infer_batch_log_path(epoch_log):
    if os.path.basename(epoch_log).startswith('training_log_'):
        return epoch_log.replace('training_log_', 'batch_log_', 1)
    return epoch_log.replace('training_log', 'batch_log', 1)


def experiment_id_from_log_path(log_path):
    name = os.path.basename(log_path)
    if name.startswith('training_log_seed'):
        return None
    if name.startswith('training_log_') and name.endswith('.csv'):
        return name[len('training_log_'):-len('.csv')]
    return None


def load_real_data(seed=None, log_path=None):
    experiment_id = None
    # Determine filenames based on seed
    if log_path:
        file_epoch = log_path
        file_batch = infer_batch_log_path(file_epoch)
        experiment_id = experiment_id_from_log_path(file_epoch)
        seed = experiment_id or seed or 'Manual'
    elif seed:
        legacy_epoch = f'training_log_seed{seed}.csv'
        legacy_batch = f'batch_log_seed{seed}.csv'
        metadata_logs = glob.glob(f'training_log_*_seed{seed}.csv')
        if metadata_logs:
            file_epoch = max(metadata_logs, key=os.path.getmtime)
            file_batch = infer_batch_log_path(file_epoch)
            experiment_id = experiment_id_from_log_path(file_epoch)
        else:
            file_epoch = legacy_epoch
            file_batch = legacy_batch
    else:
        # Default mode: Try standard file first
        if os.path.exists('training_log.csv'):
            file_epoch = 'training_log.csv'
            file_batch = 'batch_log.csv'
            seed = 'Legacy'
        else:
            # Auto-detect latest seed log
            print("⚠️ No seed specified and 'training_log.csv' not found.")
            logs = glob.glob('training_log_*_seed*.csv') + glob.glob('training_log_seed*.csv')
            if not logs:
                print("❌ No training logs found in current directory.")
                return None, None, None, None
                
            # Sort by modification time (newest first)
            latest_log = max(logs, key=os.path.getmtime)
            # Extract seed from filename "training_log_seedXXXX.csv"
            try:
                detected_seed = latest_log.replace('.csv', '').split('seed')[-1]
                print(f"🕵️ Auto-detected latest run: Seed {detected_seed}")
                file_epoch = latest_log
                file_batch = infer_batch_log_path(latest_log)
                experiment_id = experiment_id_from_log_path(latest_log)
                seed = detected_seed
            except:
                print("❌ Failed to parse seed from filename.")
                return None, None, None, None

    print(f"🔍 Loading logs: {file_epoch}")

    if not (os.path.exists(file_epoch) and os.path.exists(file_batch)):
        print(f"❌ Error: Batch log not found for seed {seed}")
        print(f"   Expected: {file_batch}")
        return None, None, None, None
        
    print(f"📂 Reading data...")
    df_epoch = pd.read_csv(file_epoch)
    df_batch = pd.read_csv(file_batch)
    return df_epoch, df_batch, seed, experiment_id

def add_best_model_line(ax_plt, epoch, label_y_pos=None, color='#333333'):
    """辅助函数：添加最佳模型垂直线"""
    ax_plt.axvline(x=epoch, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    
    ymin, ymax = ax_plt.get_ylim()
    text_pos = ymax - (ymax - ymin) * 0.05 if label_y_pos is None else label_y_pos
    
    ax_plt.text(epoch, text_pos, ' Best Checkpoint', rotation=90, 
                verticalalignment='top', fontsize=10, color=color, alpha=0.8)


def train_mae_column(df_epoch):
    if 'Train_Mixup_MAE' in df_epoch.columns:
        return 'Train_Mixup_MAE', 'Train MixUp MAE'
    return 'Train_MAE', 'Train MAE'

def plot_thesis_suite(seed=None, log_path=None, experiment_id=None):
    df_epoch, df_batch, detected_seed, detected_experiment_id = load_real_data(seed, log_path=log_path)
    if df_epoch is None: return

    # Use detected seed if original seed was None
    final_seed = seed if seed else detected_seed
    final_experiment_id = experiment_id or detected_experiment_id

    # Dynamic Save Dir
    if final_experiment_id:
        current_save_dir = os.path.join(SAVE_DIR, final_experiment_id)
    elif final_seed and final_seed != 'Legacy':
        current_save_dir = os.path.join(SAVE_DIR, f"seed_{final_seed}")
    else:
        current_save_dir = SAVE_DIR
        
    if not os.path.exists(current_save_dir):
        os.makedirs(current_save_dir)

    # 计算全局最佳轮次 (基于 Val_MAE)
    best_idx = df_epoch['Val_MAE'].idxmin()
    best_epoch = df_epoch.loc[best_idx, 'Epoch']
    best_mae_val = df_epoch.loc[best_idx, 'Val_MAE']
    
    print(f"🚀 开始生成 8 张独立图表 -> {SAVE_DIR}/")
    print(f"💡 最佳模型出现在第 {best_epoch} 轮 (MAE={best_mae_val:.4f})")

    # ==========================================
    # 图 1: Loss 收敛曲线 (基础版)
    # ==========================================
    plt.figure(figsize=(8, 6))
    ax1 = plt.gca()
    plt.plot(df_epoch['Epoch'], df_epoch['Train_Loss'], label='Train Loss', color='#2878B5')
    if 'Val_Loss' in df_epoch.columns:
        plt.plot(df_epoch['Epoch'], df_epoch['Val_Loss'], label='Val Loss', color='#D76364', linestyle='--')
    add_best_model_line(ax1, best_epoch)
    plt.title('Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(frameon=True, fancybox=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{current_save_dir}/1_loss_curve.png')
    plt.close()

    # ==========================================
    # 图 2: MAE 性能曲线
    # ==========================================
    plt.figure(figsize=(8, 6))
    train_mae_col, train_mae_label = train_mae_column(df_epoch)
    plt.plot(df_epoch['Epoch'], df_epoch[train_mae_col], label=train_mae_label, color='#9AC9DB')
    plt.plot(df_epoch['Epoch'], df_epoch['Val_MAE'], label='Val MAE', color='#C82423', linestyle='--')
    plt.axvline(x=best_epoch, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    plt.scatter(best_epoch, best_mae_val, color='black', s=60, zorder=5)
    plt.annotate(f'Best MAE: {best_mae_val:.2f}\n(Epoch {best_epoch})', 
                 xy=(best_epoch, best_mae_val), 
                 xytext=(best_epoch + 5, best_mae_val + 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    plt.title('Model Performance (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend(frameon=True, fancybox=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{current_save_dir}/2_mae_curve.png')
    plt.close()

    # ==========================================
    # 图 3: 学习率调度
    # ==========================================
    plt.figure(figsize=(8, 4))
    plt.plot(df_epoch['Epoch'], df_epoch['LR'], color='#6D6D6D', alpha=0.8)
    plt.fill_between(df_epoch['Epoch'], df_epoch['LR'], color='#6D6D6D', alpha=0.1)
    plt.axvline(x=best_epoch, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate (log scale)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{current_save_dir}/3_lr_schedule.png')
    plt.close()

    # ==========================================
    # 图 4: 同口径 KL Loss 差距
    # ==========================================
    if {'Train_KL_Loss', 'Val_KL_Loss'}.issubset(df_epoch.columns):
        gap = df_epoch['Val_KL_Loss'] - df_epoch['Train_KL_Loss']
        plt.figure(figsize=(8, 5))
        ax4 = plt.gca()
        plt.plot(df_epoch['Epoch'], gap, color='#845EC2', label='KL Gap (Val - Train)')
        plt.fill_between(df_epoch['Epoch'], gap, 0, color='#845EC2', alpha=0.15)
        if len(df_epoch) >= 2:
            z = np.polyfit(df_epoch['Epoch'], gap, 1)
            p = np.poly1d(z)
            plt.plot(df_epoch['Epoch'], p(df_epoch['Epoch']), "k--", alpha=0.5, linewidth=1, label='Gap Trend')
        add_best_model_line(ax4, best_epoch)
        plt.title('Comparable KL Loss Gap')
        plt.xlabel('Epoch')
        plt.ylabel('KL Loss Difference (Val - Train)')
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{current_save_dir}/4_kl_loss_gap.png')
        plt.close()
    else:
        print("⚠️ Skipping KL loss gap: Train_KL_Loss/Val_KL_Loss columns are missing in this legacy log.")

    # ==========================================
    # 图 5: Batch 稳定性 (趋势图)
    # ==========================================
    plt.figure(figsize=(12, 4))
    global_steps = range(len(df_batch))
    plt.plot(global_steps, df_batch['Total_Loss'], color='#555555', alpha=0.3, linewidth=0.5, label='Raw Batch Loss')
    window = 100
    if len(df_batch) > window:
        trend = df_batch['Total_Loss'].rolling(window).mean()
        plt.plot(global_steps, trend, color='#C82423', linewidth=1.5, label=f'Trend (MA={window})')
    plt.title('Training Stability (Batch Level)')
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    limit = df_batch['Total_Loss'].iloc[int(len(df_batch)*0.01):].quantile(0.999) * 1.1
    plt.ylim(0, limit)
    plt.legend(loc='upper right', frameon=True)
    plt.margins(x=0)
    plt.tight_layout()
    plt.savefig(f'{current_save_dir}/5_batch_stability.png')
    plt.close()

    # ==========================================
    # [NEW] 图 6: 训练时间效率分析 (Time Efficiency)
    # ==========================================
    # 计算每个 Epoch 的耗时 (处理断点续训的情况)
    time_deltas = []
    prev_time = 0
    print("\n⏱️ 正在分析训练耗时 (检测断点)...")
    for idx, t in enumerate(df_epoch['Time']):
        epoch_num = df_epoch.loc[idx, 'Epoch']
        if t < prev_time: # 发生了重启
            delta = t
            print(f"  -> Epoch {epoch_num}: 检测到时间重置 (Time={t:.1f}s) -> 判定为重启后首轮")
        else:
            delta = t - prev_time
        
        time_deltas.append(delta)
        prev_time = t
    
    avg_time = np.mean(time_deltas)
    print(f"  -> 平均每轮耗时: {avg_time:.2f} 秒")

    plt.figure(figsize=(8, 5))
    plt.plot(df_epoch['Epoch'], time_deltas, marker='o', markersize=4, color='#2E8B57', alpha=0.8)
    plt.axhline(y=avg_time, color='#2E8B57', linestyle='--', alpha=0.5, label=f'Avg: {avg_time:.1f}s')
    plt.title('Training Time Cost per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Duration (seconds)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{current_save_dir}/6_time_efficiency.png')
    plt.close()

    # ==========================================
    # [NEW] 图 7: Batch Loss 分布 (Boxplot)
    # ==========================================
    # 展示每个 Epoch 的 Loss 分布，观察收敛的方差变化
    plt.figure(figsize=(10, 6))
    # 为了防止 Epoch 太多导致箱线图太挤，我们每隔几个 Epoch 采样一个，或者只画前N和后N
    # 这里选择：如果 Epoch < 20 全画，否则每隔 (Total/20) 画一个
    unique_epochs = df_batch['Epoch'].unique()
    if len(unique_epochs) > 20:
        step = len(unique_epochs) // 20
        selected_epochs = unique_epochs[::step]
    else:
        selected_epochs = unique_epochs
    
    filtered_batch = df_batch[df_batch['Epoch'].isin(selected_epochs)]
    
    # 修复：添加 hue 参数和 legend=False
    sns.boxplot(x='Epoch', y='Total_Loss', data=filtered_batch, hue='Epoch', palette="Blues", fliersize=1, linewidth=1)
    plt.title('Batch Loss Distribution per Epoch (Variance Analysis)')
    plt.xlabel('Epoch')
    plt.ylabel('Batch Loss')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{current_save_dir}/7_batch_loss_dist.png')
    plt.close()

    # ==========================================
    # [NEW] 图 8: Loss 与 LR 联合分析 (Dual Axis)
    # ==========================================
    fig, ax1_dual = plt.subplots(figsize=(9, 6))
    
    color_loss = '#D76364'
    ax1_dual.set_xlabel('Epoch')
    ax1_dual.set_ylabel('Val Loss', color=color_loss)
    if 'Val_Loss' in df_epoch.columns:
        ax1_dual.plot(df_epoch['Epoch'], df_epoch['Val_Loss'], color=color_loss, label='Val Loss', linewidth=2)
    else:
        ax1_dual.plot(df_epoch['Epoch'], df_epoch['Train_Loss'], color=color_loss, label='Train Loss', linewidth=2)
    ax1_dual.tick_params(axis='y', labelcolor=color_loss)
    
    ax2_dual = ax1_dual.twinx()  # 实例化第二个轴
    color_lr = '#6D6D6D'
    ax2_dual.set_ylabel('Learning Rate', color=color_lr)
    ax2_dual.plot(df_epoch['Epoch'], df_epoch['LR'], color=color_lr, linestyle='--', alpha=0.6, label='LR')
    ax2_dual.tick_params(axis='y', labelcolor=color_lr)
    ax2_dual.set_yscale('log')

    plt.title('Validation Loss vs Learning Rate')
    fig.tight_layout()
    plt.savefig(f'{current_save_dir}/8_loss_lr_combined.png')
    plt.close()

    print(f"\n🎉 Plots saved to: {current_save_dir}/")

if __name__ == '__main__':
    import sys
    
    # Interactive Menu if no arguments provided
    if len(sys.argv) == 1:
        print("="*60)
        print("📊 FADE-Net Plotting Wizard")
        print("="*60)
        
        # Scan for seeds
        logs = glob.glob('training_log_*_seed*.csv') + glob.glob('training_log_seed*.csv')
        found_logs = []
        for log in logs:
            try:
                found_logs.append((log, os.path.getmtime(log), experiment_id_from_log_path(log)))
            except:
                pass
        
        found_logs.sort(key=lambda item: item[1], reverse=True)
        
        print("🔍 Found the following experiments:")
        menu_map = {}
        
        # Option 1: Auto-Detect (Latest)
        print("   1. [Auto]      Latest Modified Experiment")
        menu_map['1'] = 'AUTO'
        
        # List found experiment logs
        idx = 2
        for log, _mtime, exp_id in found_logs:
            label = exp_id or log
            print(f"   {idx}. [Run]       {label}")
            menu_map[str(idx)] = log
            idx += 1
            
        # Legacy Option
        if os.path.exists('training_log.csv'):
             print(f"   {idx}. [Legacy]    Standard Log (No Seed)")
             menu_map[str(idx)] = 'LEGACY'
             idx += 1
             
        print(f"   m. [Manual]    Enter Seed ID Manually")
        print("   q. [Quit]      Exit")
        print("-" * 60)
        
        try:
            choice = input(f"👉 Select experiment to plot [1-{idx-1}]: ").strip().lower()
            
            if choice == 'q':
                print("👋 Exiting.")
                sys.exit(0)
            elif choice == 'm':
                 manual_seed = input("👉 Enter Seed ID: ").strip()
                 sys.argv.extend(['--seed', manual_seed])
            elif choice in menu_map:
                selection = menu_map[choice]
                if selection == 'AUTO':
                    # Explicitly find latest log to enforce "Latest" behavior
                    # even if legacy training_log.csv exists
                    if logs:
                        latest_log = max(logs, key=os.path.getmtime)
                        print(f"🚀 Auto-Selected Latest: {latest_log}")
                        sys.argv.extend(['--log', latest_log])
                    else:
                        pass # Fallback to load_real_data logic
                elif selection == 'LEGACY':
                    # Check if legacy file actually exists to avoid error
                    if os.path.exists('training_log.csv'):
                        pass # Pass nothing, load_real_data picks legacy
                    else:
                        print("❌ Legacy log not found.")
                        sys.exit(1)
                else:
                    sys.argv.extend(['--log', selection])
            else:
                 print("❌ Invalid choice. Using Auto-Detect.")
                 
        except KeyboardInterrupt:
             sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Specify seed to plot (e.g. 2026)')
    parser.add_argument('--log', type=str, help='Explicit training log CSV path')
    parser.add_argument('--experiment_id', type=str, help='Override output directory name under plots/')
    args = parser.parse_args()
    
    # Special handling for Legacy override:
    # If user explicitly wants Legacy but training_log.csv is missing, it will error.
    # If user wants Auto, args.seed is None.
    
    plot_thesis_suite(seed=args.seed, log_path=args.log, experiment_id=args.experiment_id)
