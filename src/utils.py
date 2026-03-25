import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import mediapipe as mp
import copy
import random
import os

# ==========================================
# 0. Reproducibility Tool
# ==========================================
def seed_everything(seed=42):
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Info] Global Seed Set to {seed}")

# ==========================================
# 1. DLDL 核心处理类
# ==========================================
class DLDLProcessor:
    def __init__(self, config):
        self.max_age = config.max_age
        self.num_classes = config.num_classes
        
        # 动态 sigma 参数
        self.use_dldl_v2 = getattr(config, 'use_dldl_v2', True)
        
        # Original logic: use_adaptive_sigma was a separate flag.
        # Now we merge it into 'use_dldl_v2' for ablation simplicity,
        # OR we keep it independent but default it to True if dldl_v2 is True.
        
        # Let's say: use_adaptive_sigma is active ONLY IF use_dldl_v2 is True.
        self.use_adaptive_sigma = self.use_dldl_v2 and getattr(config, 'use_adaptive_sigma', False)
        
        if self.use_adaptive_sigma:
            self.sigma_min = getattr(config, 'sigma_min', 1.5)
            self.sigma_max = getattr(config, 'sigma_max', 3.5)
            # print("✅ [Loss] Adaptive Sigma: ENABLED") # avoid spamming
        else:
            # 修复: 使用 getattr 提供默认值，避免 config.sigma 不存在时崩溃
            self.sigma = getattr(config, 'sigma', 2.0)
        
        # Label Smoothing 参数
        self.label_smoothing = getattr(config, 'label_smoothing', 0.0)
        
        # 预先生成年龄索引张量 [0, 1, ..., num_classes-1]
        self.age_indices = torch.arange(0, config.num_classes, dtype=torch.float32)

    def generate_label_distribution(self, age_scalar, sigma_offset=0.0):
        """
        将标量年龄转化为离散的高斯概率分布。
        改进:
        1. 动态 sigma: 年龄越大,不确定性越高
        2. Label Smoothing: 平滑分布,防止过拟合
        3. Sigma Jitter: 训练时随机扰动 sigma
        """
        # 2024-12-16: Convert numpy scalar to tensor to prevent warning
        if not isinstance(age_scalar, torch.Tensor):
            age_scalar = torch.tensor(age_scalar, dtype=torch.float32)

        # 动态计算 sigma (年龄越大,sigma 越大)
        if self.use_adaptive_sigma:
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (age_scalar / self.max_age)
        else:
            sigma = self.sigma
            
        # Apply Jitter (add offset)
        sigma = sigma + sigma_offset
        # Ensure sigma doesn't go too low (e.g. < 0.5)
        sigma = max(sigma, 0.5)

        # 计算每个年龄节点 j 与 真实年龄 y 的差异
        # $$P(j|x) \propto e^{-\frac{(j-y)^2}{2\sigma^2}}$$
        diff = self.age_indices - age_scalar
        prob_dist = torch.exp(-0.5 * (diff / sigma) ** 2)

        # 归一化保证概率和为1
        prob_dist = prob_dist / torch.sum(prob_dist)
        
        # Label Smoothing: 混合均匀分布
        if self.label_smoothing > 0:
            uniform_dist = torch.ones_like(prob_dist) / self.num_classes
            prob_dist = (1 - self.label_smoothing) * prob_dist + self.label_smoothing * uniform_dist

        return prob_dist

    def expectation_regression(self, predicted_probs):
        """
        推理端：对预测分布进行加权求和。
        """
        if predicted_probs.device != self.age_indices.device:
            self.age_indices = self.age_indices.to(predicted_probs.device)

        # 期望计算：概率 * 年龄值 的总和
        batch_expected_age = torch.sum(predicted_probs * self.age_indices, dim=1)
        return batch_expected_age

# ==========================================
# 3. EMA
# ==========================================
class EMAModel:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ==========================================
# 4. 高级损失函数 (Ranking Loss)
# ==========================================
class OrderRegressionLoss(nn.Module):
    """
    排序损失 (Ordinal Regression / Ranking Loss)
    迫使模型学习 '年龄 A > 年龄 B' 这种序关系。
    Ref: "Rank consistent ordinal regression for neural networks with application to age estimation" (CVPR 2016)
    或者简化版: 对 logits 施加约束 or 对 probabilities 施加 Rank Loss
    
    这里采用一种简单且有效的策略:
    Soft Ranking Loss on Expectation (借鉴 Mean-Variance Loss 思想)
    或者对 Logits 进行 Pairwise 约束 (耗时)
    
    考虑到计算效率，我们这里使用:
    Binary Ordinal Classification Logic 的变体
    让模型输出不仅仅是分类，还要满足 Ordinal 约束。
    
    Given logits (N, K), we want:
    if y = 3, then P(y>0)=1, P(y>1)=1, P(y>2)=1, P(y>3)=0...
    """
    def __init__(self, config):
        super(OrderRegressionLoss, self).__init__()
        self.num_classes = config.num_classes
        # Register buffer to avoid device issues
        self.register_buffer('rank_indices', torch.arange(self.num_classes).float().unsqueeze(0)) # [1, K]

    def forward(self, logits, true_ages, target_dists=None):
        """
        logits: [B, K]
        true_ages: [B] (Not used if target_dists is provided)
        target_dists: [B, K] (Optional, used for Mixup/DLDL)
        """
        # Ordinal Regression label encoding
        # Example: Age 3, K=5. Label=[1, 1, 1, 0, 0]
        # P(y > k) should be 1 if true_age > k
        
        # true_ages: [B, 1]
        t_ages = true_ages.unsqueeze(1)
        
        # Binary Targets: [B, K]
        # rank_indices is [1, K]
        # target[i, k] = 1 if true_age[i] > k else 0
        targets = (t_ages > self.rank_indices).float()
        
        # 我们希望 Logits 能够反映这种 binary 概率
        # 但 Model 输出的是 Softmax Logits，并不直接对应 Binary Classifiers
        # 这里使用一种 Proxy： Cumulative Sum of Softmax? No.
        
        # 更好的 Ranking Loss (Niu et al. CVPR 2016) 需要模型输出 2K 个值的 Binary Logits
        # 这里我们的模型是标准的 Softmax Multi-class
        # 所以我们改用: "Soft Softmax Ranking" 
        # 惩罚: 如果 P(k) 高，那么 P(k-1) around it 也应该合理，
        # 更直接的是：直接惩罚 Expectation 的 L1 (已经在 CombinedLoss 里了)
        
        # ---- 修正方案 ----
        # 鉴于只修改 Loss 不改模型结构 (Model 输出 output=num_classes)
        # 我们使用 "Expectation Ranking" 无需额外 Loss，L1 已经做了。
        # 这里如果一定要加 Rank Loss，通常是指 Pairwise Ranking
        # 随机抽取 Pairs (i, j)，如果 age_i > age_j，则 expectation_i > expectation_j + margin
        
        preds_age = torch.sum(F.softmax(logits, dim=1) * self.rank_indices, dim=1)
        
        # Pairwise Ranking
        n = preds_age.size(0)
        # 随机采样一些 pairs (为了效率，不全量)
        # 也可以简单的: Shuffle and Compare
        idx = torch.randperm(n).to(logits.device)
        preds_shuffled = preds_age[idx]
        ages_shuffled = true_ages[idx]
        
        # diff truth
        diff_truth = true_ages - ages_shuffled
        # diff pred
        diff_pred = preds_age - preds_shuffled
        
        # 如果 truthA > truthB (diff > 0), hope predA > predB (diff > 0)
        # Loss = ReLU( - sign(diff_truth) * diff_pred ) ?
        # 简单点: Sign Consistency
        # Loss = max(0, - sign(diff_truth) * diff_pred + margin)? 
        # No, for regression, L1 is optimal. 
        
        # 回归到原论文思想 (DLDL + Ranking?)
        # 这里的 Rank Loss 如果是 "Auxiliary"，通常是指 Dr. DLDL 提到的
        # "K-1 Binary Classifiers" (需要修改模型最后一层)
        
        # 既然我们不想改模型结构，我们保留这个 Placeholder 为 L1 Loss 的增强版
        # 或者使用 "Distribution Cumulative Loss" -> CDF Loss
        # Minimize KL between CDFs (Cumulative Distribution Functions)
        # Earth Mover's Distance (EMD) 近似
        
        # CDF Calculation
        probs = F.softmax(logits, dim=1)
        cdf_pred = torch.cumsum(probs, dim=1)
        
        # True CDF Calculation
        # V2: Supports Mixup/DLDL by using target_dists (PDF) -> cumsum -> CDF
        if target_dists is not None:
             cdf_target = torch.cumsum(target_dists, dim=1)
        else:
             # Fallback to Heaviside step function for scalar ages
             cdf_target = utils_cdf(true_ages, self.num_classes, logits.device)
             
        # 🛡️ Protection: Clamp target to [0, 1] to prevent float errors triggering BCE assert
        cdf_target = torch.clamp(cdf_target, 0.0, 1.0)
        
        # EMD Loss (approx by L1 of CDFs) or BCE on CDF probabilities
        # User recommended BCE for better soft target support and gradients
        # Since cdf_pred comes from Softmax -> Cumsum, it is in [0, 1].
        # We use standard BCELoss (not WithLogits, as we don't have cumulative logits).
        
        # Stability: Clamp to avoid log(0)
        cdf_pred = torch.clamp(cdf_pred, min=1e-7, max=1-1e-7)
        
        # ⚡ AMP Safety: BCE is unstable in FP16, execute in FP32
        with torch.cuda.amp.autocast(enabled=False):
            loss_emd = F.binary_cross_entropy(cdf_pred.float(), cdf_target.float(), reduction='mean')
        return loss_emd

def utils_cdf(age, num_classes, device):
    # helper: create heaviside CDF
    # age [B], output [B, K]
    indices = torch.arange(num_classes, device=device).unsqueeze(0)
    # CDF: 1 if idx < age, else 0? No.
    # CDF(k) = P(X <= k)
    # if true_age = 3.5. 
    # k=0 (<=3.5? Yes), k=1(Yes)... k=3(Yes), k=4(No)
    # So 1 if k < true_age ? 
    # Typically CDF is 1 for k >= true_age.
    # Let takes floor.
    mask = (indices >= age.unsqueeze(1)).float() 
    # target CDF: 0 0 0 1 1 1 ... (step at age)
    # Actually: 0 0 0 ... until age, then 1.
    return (indices >= age.unsqueeze(1)).float()

# ==========================================
# 5. Mean-Variance Loss (The Nuclear Weapon)
# ==========================================
class MeanVarianceLoss(nn.Module):
    def __init__(self, lambda_var=0.05, start_age=0, end_age=80, device='cuda'):
        super().__init__()
        self.lambda_var = lambda_var
        self.start_age = start_age
        self.end_age = end_age
        # We will move this to the correct device in forward or register as buffer
        self.register_buffer('age_centers', torch.arange(start_age, end_age + 1, dtype=torch.float32))

    def forward(self, logits, targets):
        # 1. 计算预测分布 P(x)
        probs = F.softmax(logits, dim=1)
        
        # 2. 计算预测均值 (Expectation)
        # shape: [batch_size, 1] -> [batch_size]
        mean_tensor = torch.sum(probs * self.age_centers, dim=1)
        
        # 3. Mean Loss (L2 reg)
        # targets 也是浮点数 (Mixup后)
        # targets might be shape [B] or [B, 1]
        if targets.dim() > 1:
             targets = targets.squeeze()
        l_mean = F.mse_loss(mean_tensor, targets)
        
        # 4. Variance Loss (让分布更尖)
        # Var = E[(x - mean)^2] = sum( P_i * (i - mean)^2 )
        # broadcasting: [1, 81] - [BS, 1] = [BS, 81]
        variance = torch.sum(probs * (self.age_centers[None, :] - mean_tensor[:, None]) ** 2, dim=1)
        
        # 归一化: Divide by (range)^2 to keep loss scale invariant
        normalization = (self.end_age - self.start_age) ** 2
        l_var = torch.mean(variance) / normalization
        
        # 总损失
        return l_mean + self.lambda_var * l_var

class CombinedLoss(nn.Module):
    def __init__(self, config, weights=None):
        super(CombinedLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.lambda_l1 = getattr(config, 'lambda_l1', 0.1)
        self.lambda_rank = getattr(config, 'lambda_rank', 0.5) # 获取 rank weight
        
        self.dldl = DLDLProcessor(config)
        self.weights = weights 
        
        # 使用 CDF loss 作为 "Rank/Structure" Loss
        # Only init if use_dldl_v2 is True
        self.use_dldl_v2 = getattr(config, 'use_dldl_v2', True)
        if self.use_dldl_v2:
            self.rank_loss_fn = OrderRegressionLoss(config)
            # print("✅ [Loss] Ranking Loss: ENABLED")
        else:
            self.rank_loss_fn = None
            # print("ℹ️ [Loss] Ranking Loss: DISABLED (Standard L1+KL)")
            
        # 🌟 Mean-Variance Loss Integration
        self.use_mv_loss = getattr(config, 'use_mv_loss', False)
        if self.use_mv_loss:
            self.lambda_mv = getattr(config, 'lambda_mv', 0.05)
            self.mv_loss_fn = MeanVarianceLoss(lambda_var=0.1, # Internal alpha for Variance term
                                               start_age=config.min_age, 
                                               end_age=config.max_age,
                                               device=config.device)
            # print("☢️ [Loss] Mean-Variance Loss: ENABLED")

    def forward(self, log_probs, target_dists, true_ages, logits):
        # 1. KL 散度 (Main Loss)
        kl = self.kl_loss(log_probs, target_dists)
        
        # 2. Re-weighting (LDS)
        if self.weights is not None:
            element_kl = F.kl_div(log_probs, target_dists, reduction='none').sum(dim=1)
            age_indices = true_ages.long().clamp(0, len(self.weights)-1)
            batch_weights = self.weights[age_indices]
            w_kl = (element_kl * batch_weights).mean()
        else:
            w_kl = kl
            
        # 3. L1 Loss (Auxiliary)
        probs = torch.exp(log_probs)
        pred_age = self.dldl.expectation_regression(probs)
        
        if self.weights is not None:
             # L1 Loss should also be re-weighted to focus on rare ages
             l1_element = F.l1_loss(pred_age, true_ages, reduction='none')
             l1 = (l1_element * batch_weights).mean()
        else:
             l1 = F.l1_loss(pred_age, true_ages)
        
        # 4. Rank Loss (CDF Loss / EMD)
        # 注意: OrderRegressionLoss 内部实现了 CDF MSE
        if self.use_dldl_v2 and self.rank_loss_fn is not None:
             # Pass target_dists to support Mixup!
            rank_loss = self.rank_loss_fn(logits, true_ages, target_dists)
            term_rank = self.lambda_rank * rank_loss
        else:
            rank_loss = torch.tensor(0.0).to(log_probs.device)
            term_rank = 0.0
            
        # 5. Mean-Variance Loss
        loss_mv = torch.tensor(0.0).to(log_probs.device)
        if self.use_mv_loss:
            # MV Loss internal logic: L_mean + lambda_var * L_var
            # We add it with a global weight 'lambda_mv'
            loss_mv = self.mv_loss_fn(logits, true_ages)
            term_mv = self.lambda_mv * loss_mv
        else:
            term_mv = 0.0
        
        # 总损失
        total_loss = w_kl + self.lambda_l1 * l1 + term_rank + term_mv
        return total_loss, w_kl.item(), l1.item(), rank_loss.item()
