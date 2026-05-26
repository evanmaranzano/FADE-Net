import os
import torch

# Define Project Root (src is one level deep)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    # --- 1. 🔬 Ablation Switch (消融实验核心开关) ---
    # NOTE: use_hybrid_attention has no effect on timm backbones (V4 has built-in attention).
    use_hybrid_attention = True  # HA: Coordinate Attention (torchvision backbone only)
    use_dldl_v2 = True           # DLDL: Adaptive Sigma + Rank Loss
    use_multi_scale = True       # MSFF: Texture-Semantics Dual-Stream
    use_spp = True               # SPP: Bottleneck SPP v2 (Global-Local Fusion)

    # --- 1.0.1 New Innovation Modules ---
    use_texture_branch = False   # M1: High-frequency texture enhancement branch
    use_adaptive_triplet = False # M2: Adaptive triplet loss
    use_asymmetric_ordinal = False  # M3: Asymmetric ordinal loss
    use_freq_attention = False   # M4: Frequency-domain attention (DCT)
    use_moe = False              # M5: Age-aware Mixture of Experts head

    # --- 1.0 Backbone Adapter ---
    # Primary backbone: MobileNetV4-Small via timm (2024 architecture, built-in attention).
    # HA (CoordAtt injection) is not applicable to timm backbones — V4 already has
    # efficient attention. Ablation focuses on MSFF / SPP / DLDL / MV modules.
    backbone_source = "timm"
    backbone_name = "mobilenetv4_conv_small"
    backbone_pretrained = True
    experiment_tag = None
    split_file_tag = None
    allow_legacy_split_upgrade = False
    head_version = "fade-head-v2"
    hybrid_attention_blocks = 4
    msff_feature_indices = (1, 3)
    fusion_dim = 64
    fusion_out_dim = 128
    spp_channels = 512
    semantic_dim = 1280
    
    # --- 1.1 📊 Split Protocol (New) ---
    # Options: '80-10-10' (Our Best), '90-5-5' (Legacy), or '72-8-20' (Standard 80-20 implementation)
    split_protocol = '72-8-20'

    # --- 1.2 🌱 Academic Seeds (with Meanings) ---
    ACADEMIC_SEEDS = {
        42:   "The Answer to Life, the Universe, and Everything",
        3407: '"Torch.manual_seed(3407) is all you need" (arXiv:2109.08203)',
        2026: "Current Year (Modernity Check)",
        1337: "Leet (Elite)",
        1106: "Special Dedication <3 (Randomly Sampled w.r.t our hearts)",
        2027: "The Robustness Overhaul (Correction of 2026's Hubris)"
    }

    # --- 2. 🚀 动态项目命名逻辑 (Robust & Dynamic) ---
    @property
    def project_name(self):
        base = "FADE-Net"
        tags = []
        replaced_blocks = getattr(self, "hybrid_attention_replaced_blocks", None)
        if replaced_blocks is None:
            effective_ha = self.use_hybrid_attention and self.backbone_source != "timm"
        else:
            effective_ha = bool(replaced_blocks)
        if effective_ha: tags.append("HA")
        if self.use_dldl_v2:          tags.append("DLDL")
        if self.use_multi_scale:      tags.append("MSFF")
        if self.use_spp:              tags.append("SPP")
        if getattr(self, 'use_mv_loss', False): tags.append("MV")
        if getattr(self, 'use_texture_branch', False): tags.append("TEX")
        if getattr(self, 'use_freq_attention', False): tags.append("FREQ")
        if getattr(self, 'use_moe', False): tags.append("MOE")
        if getattr(self, 'use_adaptive_triplet', False): tags.append("TRIPLET")
        if getattr(self, 'use_asymmetric_ordinal', False): tags.append("ASYM")
        
        suffix = "_".join(tags) if tags else "Baseline"
        return f"{base}_{suffix}"

    # --- 3. 🎯 核心超参数 (Based on Final Tuning) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 年龄区间
    min_age, max_age = 0, 80
    num_classes = 81 
    
    # DLDL-v2 动态微调参数
    use_adaptive_sigma = True
    sigma_min = 1.0              # 🛡️ Rescue: Sharpened back to 1.0 (Precision)
    sigma_max = 3.0              # 🛡️ Rescue: Tightened upper bound
    lambda_l1 = 0.1              # 📉 Oracle: 0.1
    lambda_rank = 0.5            # 👑 Standard: 0.5 for Ranking Loss weight

    # Mean-Variance Loss (Nuclear Weapon)
    use_mv_loss = True
    lambda_mv = 0.1

    # Adaptive Triplet Loss (M2)
    lambda_triplet = 0.1
    triplet_base_margin = 0.2
    triplet_alpha = 0.01
    triplet_max_margin = 0.5
    triplet_age_threshold = 3.0  # years: within this = positive pair

    # MoE Head (M5)
    moe_num_experts = 3
    moe_hidden_dim = 256
    lambda_moe_gate = 0.02
    lambda_moe_balance = 0.005

    # Asymmetric Ordinal Loss (M3)
    lambda_asym = 0.1
    asym_under_weight = 2.0
    asym_over_weight = 1.0
    asym_delta = 1.0
    
    # Label-Level Perturbation (Sigma Jitter)
    use_sigma_jitter = True
    sigma_jitter = 0.2
    
    # 训练/优化
    batch_size = 128             # 🚀 Increased for A10 (24GB VRAM) utilization
    learning_rate = 0.0003       #保持 3e-4
    weight_decay = 4e-4          # ⚖️ 2027c: Increased to 4e-4 (Standard AdamW)
    epochs = 120
    
    # 训练策略
    freeze_backbone_epochs = 10  # Keep 10 for safety
    
    # 数据增强与正则化
    dropout = 0.35               # ⚖️ 2027d: Adjusted to 0.35 (Rescue 1106: Stronger Regularization)
    use_mixup = True             # ✅ Re-enabled: Essential for Manifold Smoothing & Generalization
    
    # ✅ [Added] Random Erasing as Compensation
    use_random_erasing = True    # 🛡️ Balanced: Enabled (Light regularization)
    re_prob = 0.1                # 🛡️ Balanced: 0.1 (Conservative but robust)
    
    mixup_alpha = 0.5            # 🐸 Oracle: 0.5 (Standard Mixup)
    mixup_prob = 0.5
    
    use_ema = True
    ema_decay = 0.999            # 🛡️ 以 EMA 为准
    
    # 标签平滑 (Label Smoothing)
    label_smoothing = 0.0        # 禁用，避免污染 DLDL 分布
    
    # 数据集开关 (AFAD Only)
    use_afad = True

    # 数据集路径 relative to ROOT_DIR; can be overridden without editing code.
    afad_dir = os.environ.get("FADE_NET_AFAD_DIR", os.path.join(ROOT_DIR, "datasets", "AFAD"))
    
    # LDS (标签分布平滑)
    use_reweighting = True
    use_alignment = False
    
    lds_sigma = 4                # 📉 Oracle: 4 (Stronger LDS smoothing)
    
    # 图片参数
    img_size = 224
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    num_workers = 4              # 🏎️ Optimized for CPU usage (avoid 100% load)
    early_stopping_patience = 999 # 🛡️ 2027 Strategy: "Trust the Process". Let Cosine Annealing finish its full cycle.

    # TTA batch size: controls chunk size during augmented-view inference.
    # None = auto (min(total_views, max(batch_size*2, 16))).
    # Lower this if you hit OOM on small VRAM GPUs (e.g. 8GB).
    tta_batch_size = None

    def __init__(self):
        pass # Attributes are class-level or properties
        
    def __repr__(self):
        return f"🚀 Starting Project: {self.project_name} on {self.device}"
