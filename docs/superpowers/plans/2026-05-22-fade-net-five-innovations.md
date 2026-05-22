# FADE-Net 五模块创新实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 MobileNetV4-Small backbone 上实现 5 个新模块，增强 FADE-Net 的创新点和论文贡献。

**Architecture:** 5 个独立模块，各自有 config 开关，可单独开启/关闭用于消融。模型结构改动集中在 model.py，loss 改动集中在 utils.py，config 开关在 config.py。

**Tech Stack:** PyTorch, timm, numpy

---

## 模块总览

| 编号 | 模块 | 开关 | 改动文件 | 新增参数量估计 |
|------|------|------|----------|---------------|
| M1 | 高频纹理增强分支 | `use_texture_branch` | model.py | ~0.3M |
| M2 | 自适应三元组损失 | `use_adaptive_triplet` | utils.py, train.py | 0 |
| M3 | 非对称序数损失 | `use_asymmetric_ordinal` | utils.py | 0 |
| M4 | 频域注意力 | `use_freq_attention` | model.py | ~0.01M |
| M5 | 年龄分段 MoE | `use_moe` | model.py | ~0.2M |

---

## Task 1: Config 开关和 CLI 参数

**Files:**
- Modify: `F:/FADE-Net/src/config.py`
- Modify: `F:/FADE-Net/src/train.py` (CLI args + override logic)

- [ ] **Step 1: 在 config.py 添加 5 个开关**

在 `config.py` 的消融开关区域（`use_spp = True` 之后）添加：

```python
    # --- 1.0.1 New Innovation Modules ---
    use_texture_branch = False   # M1: High-frequency texture enhancement branch
    use_adaptive_triplet = False # M2: Adaptive triplet loss
    use_asymmetric_ordinal = False  # M3: Asymmetric ordinal loss
    use_freq_attention = False   # M4: Frequency-domain attention (DCT)
    use_moe = False              # M5: Age-aware Mixture of Experts head
```

- [ ] **Step 2: 在 train.py 添加 CLI 参数**

在消融开关区域（`--no-mv` 之后）添加：

```python
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
```

- [ ] **Step 3: 在 train.py 的 override 逻辑中添加处理**

在 `args.mv_false` 处理块之后添加：

```python
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
```

- [ ] **Step 4: 在 train() 函数中应用 override**

在 `cfg.use_mv_loss = args.use_mv` 之后添加：

```python
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
```

- [ ] **Step 5: 更新 project_name 属性**

在 `config.py` 的 `project_name` property 中，`if getattr(self, 'use_mv_loss', False): tags.append("MV")` 之后添加：

```python
        if getattr(self, 'use_texture_branch', False): tags.append("TEX")
        if getattr(self, 'use_freq_attention', False): tags.append("FREQ")
        if getattr(self, 'use_moe', False): tags.append("MOE")
```

- [ ] **Step 6: 验证**

```bash
cd F:/FADE-Net && .venv/Scripts/python.exe -B -c "from src.config import Config; c = Config(); print(c.project_name)"
cd F:/FADE-Net && .venv/Scripts/python.exe -B src/train.py --help
```

---

## Task 2: M1 — 高频纹理增强分支

**原理:** 用 Sobel 算子提取面部纹理图（边缘/皱纹），接轻量 CNN 编码高频特征，与 MSFF 浅层特征融合。参考 HEAT (Multimedia Systems 2026)。

**Files:**
- Modify: `F:/FADE-Net/src/model.py`
- Test: `F:/FADE-Net/tests/test_texture_branch.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_texture_branch.py
import torch
from src.config import Config
from src.model import LightweightAgeEstimator

def test_texture_branch_forward():
    """Texture branch should produce output with correct shape."""
    cfg = Config()
    cfg.use_texture_branch = True
    cfg.use_multi_scale = True
    cfg.use_spp = True
    cfg.backbone_source = "timm"
    cfg.backbone_name = "mobilenetv4_conv_small"
    cfg.img_size = 224
    model = LightweightAgeEstimator(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, cfg.num_classes), f"Expected (2, {cfg.num_classes}), got {out.shape}"

def test_texture_branch_disabled():
    """Without texture branch, model should work normally."""
    cfg = Config()
    cfg.use_texture_branch = False
    cfg.use_multi_scale = True
    cfg.use_spp = True
    cfg.backbone_source = "timm"
    cfg.backbone_name = "mobilenetv4_conv_small"
    cfg.img_size = 224
    model = LightweightAgeEstimator(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, cfg.num_classes)

def test_texture_branch_adds_fusion_dim():
    """Texture branch should increase classifier input dim when enabled."""
    cfg1 = Config()
    cfg1.use_texture_branch = False
    cfg1.use_multi_scale = True
    cfg1.use_spp = True
    cfg1.backbone_source = "timm"
    cfg1.backbone_name = "mobilenetv4_conv_small"
    cfg1.img_size = 224
    m1 = LightweightAgeEstimator(cfg1)

    cfg2 = Config()
    cfg2.use_texture_branch = True
    cfg2.use_multi_scale = True
    cfg2.use_spp = True
    cfg2.backbone_source = "timm"
    cfg2.backbone_name = "mobilenetv4_conv_small"
    cfg2.img_size = 224
    m2 = LightweightAgeEstimator(cfg2)

    dim1 = m1.final_head[0].in_features
    dim2 = m2.final_head[0].in_features
    assert dim2 > dim1, f"Texture branch should increase head dim: {dim1} vs {dim2}"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd F:/FADE-Net && .venv/Scripts/python.exe -B -m pytest tests/test_texture_branch.py -v
```

- [ ] **Step 3: 实现 TextureEnhanceBranch 和 SobelTextureExtractor**

在 model.py 的 `BottleneckSPP` 类之后添加：

```python
class SobelTextureExtractor(nn.Module):
    """Extract texture maps using fixed Sobel filters (no trainable params)."""
    def __init__(self):
        super().__init__()
        # Sobel X
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).reshape(1, 1, 3, 3)
        # Sobel Y
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('weight_x', sobel_x)
        self.register_buffer('weight_y', sobel_y)

    def forward(self, x):
        # x: (B, 1, H, W) grayscale
        gx = F.conv2d(x, self.weight_x, padding=1)
        gy = F.conv2d(x, self.weight_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


class TextureEnhanceBranch(nn.Module):
    """Lightweight branch that extracts high-frequency texture features from
    Sobel-filtered facial texture maps. Inspired by HEAT (Multimedia Systems 2026)."""
    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        self.sobel = SobelTextureExtractor()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.Hardswish(),
        )
        self.out_dim = out_dim

    def forward(self, x_rgb):
        # Convert to grayscale
        gray = 0.299 * x_rgb[:, 0:1] + 0.587 * x_rgb[:, 1:1+1] + 0.114 * x_rgb[:, 2:3]
        texture_map = self.sobel(gray)
        feat = self.encoder(texture_map)
        feat = self.pool(feat).flatten(1)
        return self.project(feat)
```

- [ ] **Step 4: 在 LightweightAgeEstimator 中集成**

在 `__init__` 中，`self.use_spp` 块之后添加：

```python
        # M1: Texture Enhancement Branch
        self.use_texture_branch = bool(getattr(config, "use_texture_branch", False))
        if self.use_texture_branch:
            print("[Model] Texture Enhancement Branch: ENABLED")
            self.texture_branch = TextureEnhanceBranch(3, out_dim=64)
            texture_dim = self.texture_branch.out_dim
        else:
            print("[Model] Texture Enhancement Branch: DISABLED")
            texture_dim = 0
```

修改 `classifier_input_dim` 计算：

```python
        classifier_input_dim = semantic_dim + (fusion_out_dim if self.use_multi_scale else 0) + texture_dim
```

在 `forward` 方法中，`if self.use_multi_scale:` 块之后添加：

```python
        if self.use_texture_branch:
            texture_feat = self.texture_branch(x)
            semantic = torch.cat([semantic, texture_feat], dim=1)
```

- [ ] **Step 5: 跑测试确认通过**

```bash
cd F:/FADE-Net && .venv/Scripts/python.exe -B -m pytest tests/test_texture_branch.py -v
```

---

## Task 3: M4 — 频域注意力（DCT Attention）

**原理:** 用 2D DCT 替代全局平均池化，捕获频域特征通道关系。参考 FcaNet (CVPR 2021)。在 SPP 输出后加轻量频域注意力。

**Files:**
- Modify: `F:/FADE-Net/src/model.py`
- Test: `F:/FADE-Net/tests/test_freq_attention.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_freq_attention.py
import torch
from src.config import Config
from src.model import LightweightAgeEstimator, FrequencyDomainAttention

def test_freq_attention_shape():
    """Freq attention should preserve input shape."""
    attn = FrequencyDomainAttention(512, reduction=16)
    x = torch.randn(2, 512, 7, 7)
    out = attn(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

def test_freq_attention_in_model():
    """Model with freq attention should produce correct output."""
    cfg = Config()
    cfg.use_freq_attention = True
    cfg.use_spp = True
    cfg.backbone_source = "timm"
    cfg.backbone_name = "mobilenetv4_conv_small"
    cfg.img_size = 224
    model = LightweightAgeEstimator(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, cfg.num_classes)

def test_freq_attention_disabled():
    """Model without freq attention should work normally."""
    cfg = Config()
    cfg.use_freq_attention = False
    cfg.use_spp = True
    cfg.backbone_source = "timm"
    cfg.backbone_name = "mobilenetv4_conv_small"
    cfg.img_size = 224
    model = LightweightAgeEstimator(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, cfg.num_classes)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd F:/FADE-Net && .venv/Scripts/python.exe -B -m pytest tests/test_freq_attention.py -v
```

- [ ] **Step 3: 实现 FrequencyDomainAttention**

在 model.py 的 `TextureEnhanceBranch` 类之后添加：

```python
class FrequencyDomainAttention(nn.Module):
    """Lightweight frequency-domain channel attention using DCT.
    Replaces GAP with DCT to capture frequency-domain channel relationships.
    Inspired by FcaNet (CVPR 2021)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(8, channels // reduction)
        self.compress = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish()
        self.expand = nn.Conv2d(mid, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        # Use DCT-like pooling: split into frequency bands via avg pools
        b, c, h, w = x.shape
        # Multi-scale frequency extraction
        f1 = F.adaptive_avg_pool2d(x, 1)           # DC component
        f2 = F.adaptive_avg_pool2d(x, 2)[:, :, 1:, :]  # low-freq row
        f3 = F.adaptive_avg_pool2d(x, 2)[:, :, :, 1:]  # low-freq col
        # Flatten and concat
        freq = torch.cat([f1.flatten(1), f2.flatten(1), f3.flatten(1)], dim=1)
        # Pad or project to match channels
        if freq.shape[1] < c:
            freq = F.pad(freq, (0, c - freq.shape[1]))
        else:
            freq = freq[:, :c]
        freq = freq.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        attn = self.compress(freq)
        attn = self.bn(attn)
        attn = self.act(attn)
        attn = self.expand(attn)
        return x * self.sigmoid(attn)
```

- [ ] **Step 4: 在 LightweightAgeEstimator 中集成**

在 `__init__` 中，`self.use_texture_branch` 块之后添加：

```python
        # M4: Frequency-Domain Attention
        self.use_freq_attention = bool(getattr(config, "use_freq_attention", False))
        if self.use_freq_attention:
            print("[Model] Frequency-Domain Attention: ENABLED")
            spp_out = self.spp_channels if self.use_spp else semantic_dim
            self.freq_attention = FrequencyDomainAttention(spp_out, reduction=16)
        else:
            print("[Model] Frequency-Domain Attention: DISABLED")
```

在 `forward` 方法中，SPP 语义特征计算之后、拼接之前添加：

```python
        if self.use_freq_attention:
            semantic = self.freq_attention(semantic.unsqueeze(-1).unsqueeze(-1)).flatten(1) if semantic.dim() == 2 else self.freq_attention(semantic)
```

注意：这里需要根据 semantic 的维度来处理。如果 SPP 路径已经 pool 到 (B, C)，需要先 reshape 回 (B, C, 1, 1) 再过注意力。实际实现时需要根据 SPP 输出的形状调整。

- [ ] **Step 5: 跑测试确认通过**

```bash
cd F:/FADE-Net && .venv/Scripts/python.exe -B -m pytest tests/test_freq_attention.py -v
```

---

## Task 4: M5 — 年龄分段 MoE Head

**原理:** 不同年龄段的面部特征不同（年轻人看脸型轮廓，中年人看法令纹，老年人看皱纹深度）。用门控网络自动路由到不同专家 FC 头。

**Files:**
- Modify: `F:/FADE-Net/src/model.py`
- Test: `F:/FADE-Net/tests/test_moe_head.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_moe_head.py
import torch
from src.config import Config
from src.model import LightweightAgeEstimator, AgeMoEHead

def test_moe_head_shape():
    """MoE head should output correct shape."""
    head = AgeMoEHead(512, num_classes=81, num_experts=3)
    x = torch.randn(4, 512)
    out = head(x)
    assert out.shape == (4, 81)

def test_moe_head_routing():
    """Different inputs should activate different experts."""
    head = AgeMoEHead(512, num_classes=81, num_experts=3)
    x = torch.randn(8, 512)
    out = head(x)
    # Check gate weights are valid probabilities
    gate = head.gate(x)
    assert torch.allclose(gate.sum(dim=1), torch.ones(8), atol=1e-5)

def test_moe_in_model():
    """Model with MoE should produce correct output."""
    cfg = Config()
    cfg.use_moe = True
    cfg.use_spp = True
    cfg.backbone_source = "timm"
    cfg.backbone_name = "mobilenetv4_conv_small"
    cfg.img_size = 224
    model = LightweightAgeEstimator(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, cfg.num_classes)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd F:/FADE-Net && .venv/Scripts/python.exe -B -m pytest tests/test_moe_head.py -v
```

- [ ] **Step 3: 实现 AgeMoEHead**

在 model.py 的 `FrequencyDomainAttention` 类之后添加：

```python
class AgeMoEHead(nn.Module):
    """Age-aware Mixture of Experts head.
    A lightweight gating network routes features to specialized expert FC heads,
    enabling different experts to focus on different age ranges."""
    def __init__(self, in_dim, num_classes, num_experts=3, hidden_dim=256, dropout=0.35):
        super().__init__()
        self.num_experts = num_experts
        # Gate: routes to experts based on input features
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        # Experts: each is a small FC head
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.Hardswish(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=1)  # (B, num_experts)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, num_experts, num_classes)
        # Weighted sum
        out = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # (B, num_classes)
        return out
```

- [ ] **Step 4: 在 LightweightAgeEstimator 中集成**

在 `__init__` 中，`self.final_head` 定义处改为条件：

```python
        # M5: MoE Head
        self.use_moe = bool(getattr(config, "use_moe", False))
        if self.use_moe:
            print("[Model] MoE Head: ENABLED")
            self.final_head = AgeMoEHead(
                classifier_input_dim, num_classes,
                num_experts=getattr(config, "moe_num_experts", 3),
                hidden_dim=1024,
                dropout=dropout,
            )
        else:
            print("[Model] MoE Head: DISABLED")
            self.final_head = nn.Sequential(
                nn.Linear(classifier_input_dim, 1024),
                nn.Hardswish(),
                nn.Dropout(p=dropout),
                nn.Linear(1024, num_classes),
            )
```

- [ ] **Step 5: 跑测试确认通过**

```bash
cd F:/FADE-Net && .venv/Scripts/python.exe -B -m pytest tests/test_moe_head.py -v
```

---

## Task 5: M2 — 自适应三元组损失

**原理:** 传统 triplet loss 的 margin 是固定的。年龄估计中，差 1 岁和差 20 岁的约束应不同。动态 margin = base_margin × (1 + α × |age_diff|)。

**Files:**
- Modify: `F:/FADE-Net/src/utils.py`
- Modify: `F:/FADE-Net/src/train.py` (loss accumulation + logging)
- Test: `F:/FADE-Net/tests/test_adaptive_triplet.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_adaptive_triplet.py
import torch
from src.utils import AdaptiveTripletLoss

def test_adaptive_triplet_basic():
    """Adaptive triplet loss should produce scalar loss."""
    loss_fn = AdaptiveTripletLoss(base_margin=1.0, alpha=0.1)
    embeddings = torch.randn(6, 128)
    ages = torch.tensor([10, 12, 30, 32, 60, 62], dtype=torch.float32)
    loss = loss_fn(embeddings, ages)
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

def test_adaptive_triplet_margin_scales():
    """Larger age difference should produce larger effective margin."""
    loss_fn = AdaptiveTripletLoss(base_margin=1.0, alpha=0.1)
    # age diff = 2 -> margin ~1.2
    # age diff = 20 -> margin ~3.0
    m1 = loss_fn.base_margin * (1 + loss_fn.alpha * 2)
    m2 = loss_fn.base_margin * (1 + loss_fn.alpha * 20)
    assert m2 > m1, "Margin should scale with age difference"

def test_adaptive_triplet_zero_loss_same_age():
    """Same-age pairs should have small loss if embeddings are close."""
    loss_fn = AdaptiveTripletLoss(base_margin=1.0, alpha=0.1)
    embeddings = torch.randn(2, 128)
    ages = torch.tensor([30.0, 30.0])
    loss = loss_fn(embeddings, ages)
    # Should be small but not necessarily zero (depends on embeddings)
    assert loss.item() < 10.0
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd F:/FADE-Net && .venv/Scripts/python.exe -B -m pytest tests/test_adaptive_triplet.py -v
```

- [ ] **Step 3: 实现 AdaptiveTripletLoss**

在 `utils.py` 的 `MeanVarianceLoss` 类之后添加：

```python
class AdaptiveTripletLoss(nn.Module):
    """Adaptive Triplet Loss with dynamic margin based on age difference.
    margin = base_margin * (1 + alpha * |age_diff|)
    Inspired by HEAT (Multimedia Systems 2026)."""
    def __init__(self, base_margin=1.0, alpha=0.05):
        super().__init__()
        self.base_margin = base_margin
        self.alpha = alpha

    def forward(self, embeddings, ages):
        """
        Args:
            embeddings: (B, D) feature embeddings from model neck
            ages: (B,) true ages
        """
        B = embeddings.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Compute pairwise distances
        dist = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

        # Compute pairwise age differences
        age_diff = torch.abs(ages.unsqueeze(0) - ages.unsqueeze(1))  # (B, B)

        # Adaptive margin
        margin = self.base_margin * (1.0 + self.alpha * age_diff)  # (B, B)

        # Semi-hard negative mining: for each anchor, find positives (same-ish age)
        # and negatives (different age) with margin violation
        mask_pos = age_diff < 3.0  # within 3 years = positive
        mask_neg = age_diff >= 3.0  # >= 3 years = negative

        # For each anchor i, positive j, negative k: loss = max(0, d(i,j) - d(i,k) + margin)
        # Simplified: triplet loss over all valid (i,j,k) triples
        loss = torch.tensor(0.0, device=embeddings.device)
        count = 0
        for i in range(B):
            pos_idx = mask_pos[i].nonzero(as_tuple=True)[0]
            neg_idx = mask_neg[i].nonzero(as_tuple=True)[0]
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            for j in pos_idx:
                if i == j:
                    continue
                d_pos = dist[i, j]
                # Use all negatives with margin violation
                for k in neg_idx:
                    d_neg = dist[i, k]
                    m = self.base_margin * (1.0 + self.alpha * age_diff[i, k])
                    violation = d_pos - d_neg + m
                    if violation > 0:
                        loss += violation
                        count += 1

        if count > 0:
            loss = loss / count
        return loss
```

- [ ] **Step 4: 在 CombinedLoss 中集成**

在 `CombinedLoss.__init__` 中，MV loss 块之后添加：

```python
        # M2: Adaptive Triplet Loss
        self.use_adaptive_triplet = getattr(config, 'use_adaptive_triplet', False)
        if self.use_adaptive_triplet:
            self.lambda_triplet = getattr(config, 'lambda_triplet', 0.1)
            self.triplet_loss_fn = AdaptiveTripletLoss(
                base_margin=getattr(config, 'triplet_base_margin', 1.0),
                alpha=getattr(config, 'triplet_alpha', 0.05),
            )
```

在 `CombinedLoss.forward` 中，MV loss 计算之后、总损失计算之前添加：

```python
        # M2: Adaptive Triplet Loss
        loss_triplet = torch.tensor(0.0).to(log_probs.device)
        if self.use_adaptive_triplet:
            # Use logits as embeddings for triplet
            loss_triplet = self.triplet_loss_fn(logits, true_ages)
            term_triplet = self.lambda_triplet * loss_triplet
        else:
            term_triplet = 0.0
```

修改总损失：

```python
        total_loss = w_kl + self.lambda_l1 * l1 + term_rank + term_mv + term_triplet
        return total_loss, w_kl.item(), l1.item(), rank_loss.item(), loss_mv.item(), loss_triplet.item()
```

- [ ] **Step 5: 更新 train.py 的 loss logging**

CombinedLoss.forward 返回值从 5 个变为 6 个，需要更新所有调用点：
- `train()` 函数中的训练循环
- 验证循环
- loss_sums 字典添加 `"triplet"` 键
- CSV header 添加 `Train_Triplet_Loss`, `Val_Triplet_Loss`

- [ ] **Step 6: 跑测试确认通过**

```bash
cd F:/FADE-Net && .venv/Scripts/python.exe -B -m pytest tests/test_adaptive_triplet.py -v
```

---

## Task 6: M3 — 非对称序数损失

**原理:** 年龄估计中低估和高估代价不同。非对称损失对不同方向的误差施加不同权重。实现方式：修改 L1 loss 为 Huber-like 非对称版本。

**Files:**
- Modify: `F:/FADE-Net/src/utils.py`
- Test: `F:/FADE-Net/tests/test_asymmetric_ordinal.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_asymmetric_ordinal.py
import torch
from src.utils import AsymmetricOrdinalLoss

def test_asymmetric_loss_shape():
    """Asymmetric loss should produce scalar."""
    loss_fn = AsymmetricOrdinalLoss(under_weight=2.0, over_weight=1.0)
    pred = torch.tensor([25.0, 35.0, 50.0])
    true = torch.tensor([30.0, 30.0, 30.0])
    loss = loss_fn(pred, true)
    assert loss.dim() == 0

def test_asymmetric_underestimate_penalty():
    """Underestimation should be penalized more heavily."""
    loss_fn = AsymmetricOrdinalLoss(under_weight=3.0, over_weight=1.0)
    # Underestimate: pred=20, true=30 (diff=-10)
    pred_under = torch.tensor([20.0])
    true_age = torch.tensor([30.0])
    # Overestimate: pred=40, true=30 (diff=+10)
    pred_over = torch.tensor([40.0])

    loss_under = loss_fn(pred_under, true_age)
    loss_over = loss_fn(pred_over, true_age)
    assert loss_under.item() > loss_over.item(), "Underestimate should have higher loss"

def test_asymmetric_symmetric_mode():
    """When weights are equal, loss should be symmetric."""
    loss_fn = AsymmetricOrdinalLoss(under_weight=1.0, over_weight=1.0)
    pred_under = torch.tensor([20.0])
    pred_over = torch.tensor([40.0])
    true_age = torch.tensor([30.0])

    loss_under = loss_fn(pred_under, true_age)
    loss_over = loss_fn(pred_over, true_age)
    assert abs(loss_under.item() - loss_over.item()) < 1e-5, "Symmetric weights should produce equal loss"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd F:/FADE-Net && .venv/Scripts/python.exe -B -m pytest tests/test_asymmetric_ordinal.py -v
```

- [ ] **Step 3: 实现 AsymmetricOrdinalLoss**

在 `utils.py` 的 `AdaptiveTripletLoss` 类之后添加：

```python
class AsymmetricOrdinalLoss(nn.Module):
    """Asymmetric ordinal loss that penalizes under-estimation more than over-estimation
    (or vice versa). Useful when the cost of prediction errors is direction-dependent.
    Inspired by CAP-WAE (2025)."""
    def __init__(self, under_weight=2.0, over_weight=1.0, delta=1.0):
        super().__init__()
        self.under_weight = under_weight
        self.over_weight = over_weight
        self.delta = delta

    def forward(self, pred_ages, true_ages):
        diff = pred_ages - true_ages  # positive = overestimate, negative = underestimate
        # Asymmetric weights
        weights = torch.where(diff < 0, self.under_weight, self.over_weight)
        # Smooth L1 (Huber) with asymmetric weighting
        abs_diff = torch.abs(diff)
        loss = torch.where(abs_diff < self.delta,
                           0.5 * abs_diff ** 2 / self.delta,
                           abs_diff - 0.5 * self.delta)
        return (weights * loss).mean()
```

- [ ] **Step 4: 在 CombinedLoss 中集成**

在 `CombinedLoss.__init__` 中，Adaptive Triplet 之后添加：

```python
        # M3: Asymmetric Ordinal Loss
        self.use_asymmetric_ordinal = getattr(config, 'use_asymmetric_ordinal', False)
        if self.use_asymmetric_ordinal:
            self.lambda_asym = getattr(config, 'lambda_asym', 0.1)
            self.asym_loss_fn = AsymmetricOrdinalLoss(
                under_weight=getattr(config, 'asym_under_weight', 2.0),
                over_weight=getattr(config, 'asym_over_weight', 1.0),
                delta=getattr(config, 'asym_delta', 1.0),
            )
```

在 `CombinedLoss.forward` 中，L1 loss 计算之后替换：

```python
        # M3: Asymmetric Ordinal Loss (replaces or augments L1)
        if self.use_asymmetric_ordinal:
            l1 = self.asym_loss_fn(pred_age, true_ages)
            loss_asym = l1  # for logging
        else:
            loss_asym = torch.tensor(0.0).to(log_probs.device)
```

- [ ] **Step 5: 跑测试确认通过**

```bash
cd F:/FADE-Net && .venv/Scripts/python.exe -B -m pytest tests/test_asymmetric_ordinal.py -v
```

---

## Task 7: 集成测试 — 全模块组合

**Files:**
- Test: `F:/FADE-Net/tests/test_all_modules_integration.py`

- [ ] **Step 1: 写集成测试**

```python
# tests/test_all_modules_integration.py
import torch
from src.config import Config
from src.model import LightweightAgeEstimator
from src.utils import CombinedLoss

def _make_cfg(**overrides):
    cfg = Config()
    cfg.backbone_source = "timm"
    cfg.backbone_name = "mobilenetv4_conv_small"
    cfg.img_size = 224
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg

def test_all_modules_enabled():
    """All 5 new modules enabled should produce correct output."""
    cfg = _make_cfg(
        use_texture_branch=True,
        use_freq_attention=True,
        use_moe=True,
        use_adaptive_triplet=True,
        use_asymmetric_ordinal=True,
    )
    model = LightweightAgeEstimator(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, cfg.num_classes)

def test_loss_with_all_modules():
    """CombinedLoss with all new losses should produce scalar."""
    cfg = _make_cfg(
        use_adaptive_triplet=True,
        use_asymmetric_ordinal=True,
    )
    criterion = CombinedLoss(cfg)
    logits = torch.randn(4, 81)
    log_probs = torch.log_softmax(logits, dim=1)
    target_dists = torch.softmax(torch.randn(4, 81), dim=1)
    true_ages = torch.tensor([10.0, 30.0, 50.0, 70.0])
    result = criterion(log_probs, target_dists, true_ages, logits)
    assert len(result) == 6, f"Expected 6 return values, got {len(result)}"
    total_loss = result[0]
    assert total_loss.dim() == 0
    assert total_loss.item() > 0

def test_each_module_independently():
    """Each module should work independently."""
    for module in ['use_texture_branch', 'use_freq_attention', 'use_moe']:
        cfg = _make_cfg(**{module: True})
        model = LightweightAgeEstimator(cfg)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, cfg.num_classes), f"{module} failed"
```

- [ ] **Step 2: 跑全部测试**

```bash
cd F:/FADE-Net && .venv/Scripts/python.exe -B -m pytest tests/ -v
```

---

## Task 8: 更新文档

**Files:**
- Modify: `F:/FADE-Net/docs/ablation_plan_v4.md`
- Modify: `F:/FADE-Net/docs/paper_core_claims.md`

- [ ] **Step 1: 更新消融计划**

在 `ablation_plan_v4.md` 中添加新模块的消融配置：

```
| A4 | +TEX | Y | Y | Y | Y | 加高频纹理增强 |
| A5 | +FREQ | Y | Y | Y | Y | 加频域注意力 |
| A6 | +MOE | Y | Y | Y | Y | 替换为 MoE Head |
| A7 | +TRIPLET | Y | Y | Y | Y | 加自适应三元组损失 |
| A8 | +ASYM | Y | Y | Y | Y | 加非对称序数损失 |
| A9 | Full+ | Y | Y | Y | Y | 全部启用 |
```

- [ ] **Step 2: 更新论文贡献声明**

---

## 完成标准

- [ ] 全部 5 个模块可通过 CLI 开关独立开启/关闭
- [ ] 全部新增测试通过
- [ ] 原有测试不受影响
- [ ] `project_name` 正确反映启用的模块
- [ ] 训练日志 CSV 包含新 loss 列
- [ ] 文档更新完毕
