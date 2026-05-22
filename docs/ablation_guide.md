# 消融实验指南 (Ablation Study Guide)

> **注意**：当前主 backbone 已切换为 MobileNetV4-Small（timm），HA 不适用。
> 适用于 V4-small 的消融计划见 [ablation_plan_v4.md](ablation_plan_v4.md)。
> 以下内容保留为 MobileNetV3-Large backbone 的参考。

## 概述

FADE-Net 有 5 个核心模块开关，消融实验旨在验证每个模块的独立贡献。

| 开关 | 模块名称 | 说明 |
|------|----------|------|
| `use_hybrid_attention` | HA | Coordinate Attention 替换 SE |
| `use_dldl_v2` | DLDL-v2 | 自适应 Sigma + CDF Ranking Loss |
| `use_multi_scale` | MSFF | 纹理-语义双流融合 |
| `use_spp` | SPP | Bottleneck Spatial Pyramid Pooling |
| `use_mv_loss` | MV | Mean-Variance Loss |

---

## 实验设计

### 方案 A：最小消融（推荐，共 6 组实验）

验证每个模块相对于 Baseline 的独立增益，最后展示 Full Model。

| 实验编号 | 配置名称 | HA | DLDL | MSFF | SPP | MV |
|:--------:|----------|:--:|:----:|:----:|:---:|:--:|
| 1 | Baseline | ❌ | ❌ | ❌ | ❌ | ❌ |
| 2 | +HA | ✅ | ❌ | ❌ | ❌ | ❌ |
| 3 | +DLDL | ❌ | ✅ | ❌ | ❌ | ❌ |
| 4 | +MSFF | ❌ | ❌ | ✅ | ❌ | ❌ |
| 5 | +SPP | ❌ | ❌ | ❌ | ✅ | ❌ |
| 6 | Full Model | ✅ | ✅ | ✅ | ✅ | ✅ |

### 方案 B：完整消融（共 10 组实验）

在方案 A 基础上，增加逐模块累积实验。

| 实验编号 | 配置名称 | 说明 |
|:--------:|----------|------|
| 7 | HA+DLDL | 验证两个核心模块组合 |
| 8 | HA+DLDL+MSFF | 逐步累积 |
| 9 | HA+DLDL+MSFF+SPP | 逐步累积 |
| 10 | -MV (消融MV) | Full Model 移除 MV |

---

## 实验执行步骤

### ⚠️ 重要提示

1. **固定数据划分**：所有实验使用相同的 split 文件，确保公平比较
2. **多 Seed 验证**：每个配置建议跑 3 个 seed（42, 3407, 2026），报告 mean±std
3. **训练周期**：120 epochs（与默认完整配置一致）
4. **统一评估口径**：当前主实验统一使用 `--split 72-8-20`，结果文件记录 `MAE_raw`、`MAE_flip`、`MAE_multi` 和 `Selected_Test_MAE`
5. **断点恢复**：默认不自动续训；只有确认 metadata 匹配时才加 `--resume`
6. **指标声明**：论文主表使用 `Selected_Test_MAE`；TTA 影响单独用 `MAE_raw`、`MAE_flip`、`MAE_multi` 展示，不能把不同 TTA 口径混在同一排名表里。

通用运行模板：

```powershell
.\.venv\Scripts\python.exe src/train.py --seed 42 --split 72-8-20 --split_file_tag formal_v1 --fresh
.\.venv\Scripts\python.exe src/train.py --seed 3407 --split 72-8-20 --split_file_tag formal_v1 --fresh
.\.venv\Scripts\python.exe src/train.py --seed 2026 --split 72-8-20 --split_file_tag formal_v1 --fresh
```

记录结果时优先使用最终结果文件中的 `Selected_Test_MAE`；同时保留 `MAE_raw`、`MAE_flip`、`MAE_multi`，用于补充 TTA 影响表。

---

## 实验 1：Baseline

**目的**：建立性能基准

**步骤**：

1. 打开 `src/config.py`，修改以下内容：

```python
# --- 1. 🔬 Ablation Switch (消融实验核心开关) ---
use_hybrid_attention = False  # HA: OFF
use_dldl_v2 = False           # DLDL: OFF
use_multi_scale = False       # MSFF: OFF
use_spp = False               # SPP: OFF
use_mv_loss = False           # MV: OFF
```

2. 运行训练：

```bash
# Seed 42
.\.venv\Scripts\python.exe src/train.py --seed 42 --split 72-8-20 --fresh

# Seed 3407
.\.venv\Scripts\python.exe src/train.py --seed 3407 --split 72-8-20 --fresh

# Seed 2026
.\.venv\Scripts\python.exe src/train.py --seed 2026 --split 72-8-20 --fresh
```

3. 记录结果（最终 `Selected_Test_MAE`）：

| Seed | Selected_Test_MAE | MAE_raw | MAE_flip | MAE_multi |
|:----:|:-----------------:|:-------:|:--------:|:---------:|
| 42 | ___.___ | ___.___ | ___.___ | ___.___ |
| 3407 | ___.___ | ___.___ | ___.___ | ___.___ |
| 2026 | ___.___ | ___.___ | ___.___ | ___.___ |
| **Mean±Std** | ___.___ ± ___.___ | - | - | - |

---

## 实验 2：+HA (Coordinate Attention)

**目的**：验证 Hybrid Attention 的贡献

**步骤**：

1. 修改 `src/config.py`：

```python
use_hybrid_attention = True   # HA: ON ✅
use_dldl_v2 = False           # DLDL: OFF
use_multi_scale = False       # MSFF: OFF
use_spp = False               # SPP: OFF
use_mv_loss = False           # MV: OFF
```

2. 运行训练：

```bash
.\.venv\Scripts\python.exe src/train.py --seed 42 --split 72-8-20 --fresh
.\.venv\Scripts\python.exe src/train.py --seed 3407 --split 72-8-20 --fresh
.\.venv\Scripts\python.exe src/train.py --seed 2026 --split 72-8-20 --fresh
```

3. 记录结果：

| Seed | Selected_Test_MAE | Δ vs Baseline |
|:----:|:-----------------:|:-------------:|
| 42 | ___.___ | ___.___ |
| 3407 | ___.___ | ___.___ |
| 2026 | ___.___ | ___.___ |
| **Mean±Std** | ___.___ ± ___.___ | ___.___ |

---

## 实验 3：+DLDL-v2

**目的**：验证自适应标签分布学习的贡献

**步骤**：

1. 修改 `src/config.py`：

```python
use_hybrid_attention = False  # HA: OFF
use_dldl_v2 = True            # DLDL: ON ✅
use_multi_scale = False       # MSFF: OFF
use_spp = False               # SPP: OFF
use_mv_loss = False           # MV: OFF
```

2. 运行训练：

```bash
.\.venv\Scripts\python.exe src/train.py --seed 42 --split 72-8-20 --fresh
.\.venv\Scripts\python.exe src/train.py --seed 3407 --split 72-8-20 --fresh
.\.venv\Scripts\python.exe src/train.py --seed 2026 --split 72-8-20 --fresh
```

3. 记录结果：

| Seed | Selected_Test_MAE | Δ vs Baseline |
|:----:|:-----------------:|:-------------:|
| 42 | ___.___ | ___.___ |
| 3407 | ___.___ | ___.___ |
| 2026 | ___.___ | ___.___ |
| **Mean±Std** | ___.___ ± ___.___ | ___.___ |

---

## 实验 4：+MSFF (Multi-Scale Feature Fusion)

**目的**：验证双流特征融合的贡献

**步骤**：

1. 修改 `src/config.py`：

```python
use_hybrid_attention = False  # HA: OFF
use_dldl_v2 = False           # DLDL: OFF
use_multi_scale = True        # MSFF: ON ✅
use_spp = False               # SPP: OFF
use_mv_loss = False           # MV: OFF
```

2. 运行训练：

```bash
.\.venv\Scripts\python.exe src/train.py --seed 42 --split 72-8-20 --fresh
.\.venv\Scripts\python.exe src/train.py --seed 3407 --split 72-8-20 --fresh
.\.venv\Scripts\python.exe src/train.py --seed 2026 --split 72-8-20 --fresh
```

3. 记录结果：

| Seed | Selected_Test_MAE | Δ vs Baseline |
|:----:|:-----------------:|:-------------:|
| 42 | ___.___ | ___.___ |
| 3407 | ___.___ | ___.___ |
| 2026 | ___.___ | ___.___ |
| **Mean±Std** | ___.___ ± ___.___ | ___.___ |

---

## 实验 5：+SPP (Spatial Pyramid Pooling)

**目的**：验证 SPP 模块的贡献

**步骤**：

1. 修改 `src/config.py`：

```python
use_hybrid_attention = False  # HA: OFF
use_dldl_v2 = False           # DLDL: OFF
use_multi_scale = False       # MSFF: OFF
use_spp = True                # SPP: ON ✅
use_mv_loss = False           # MV: OFF
```

2. 运行训练：

```bash
.\.venv\Scripts\python.exe src/train.py --seed 42 --split 72-8-20 --fresh
.\.venv\Scripts\python.exe src/train.py --seed 3407 --split 72-8-20 --fresh
.\.venv\Scripts\python.exe src/train.py --seed 2026 --split 72-8-20 --fresh
```

3. 记录结果：

| Seed | Selected_Test_MAE | Δ vs Baseline |
|:----:|:-----------------:|:-------------:|
| 42 | ___.___ | ___.___ |
| 3407 | ___.___ | ___.___ |
| 2026 | ___.___ | ___.___ |
| **Mean±Std** | ___.___ ± ___.___ | ___.___ |

---

## 实验 6：Full Model (默认完整配置)

**目的**：展示完整模型性能

**步骤**：

1. 修改 `src/config.py`：

```python
use_hybrid_attention = True   # HA: ON ✅
use_dldl_v2 = True            # DLDL: ON ✅
use_multi_scale = True        # MSFF: ON ✅
use_spp = True                # SPP: ON ✅
use_mv_loss = True            # MV: ON ✅
```

2. 运行训练：

```bash
.\.venv\Scripts\python.exe src/train.py --seed 42 --split 72-8-20 --fresh
.\.venv\Scripts\python.exe src/train.py --seed 3407 --split 72-8-20 --fresh
.\.venv\Scripts\python.exe src/train.py --seed 2026 --split 72-8-20 --fresh
```

3. 记录结果：

| Seed | Selected_Test_MAE | Δ vs Baseline |
|:----:|:-----------------:|:-------------:|
| 42 | ___.___ | ___.___ |
| 3407 | ___.___ | ___.___ |
| 2026 | ___.___ | ___.___ |
| **Mean±Std** | ___.___ ± ___.___ | ___.___ |

---

## 可选：方案 B 补充实验

如果时间允许，可以追加以下实验以展示模块间的协同效应。

### 实验 7：HA + DLDL

```python
use_hybrid_attention = True   # ✅
use_dldl_v2 = True            # ✅
use_multi_scale = False       # ❌
use_spp = False               # ❌
use_mv_loss = False           # ❌
```

### 实验 8：HA + DLDL + MSFF

```python
use_hybrid_attention = True   # ✅
use_dldl_v2 = True            # ✅
use_multi_scale = True        # ✅
use_spp = False               # ❌
use_mv_loss = False           # ❌
```

### 实验 9：HA + DLDL + MSFF + SPP

```python
use_hybrid_attention = True   # ✅
use_dldl_v2 = True            # ✅
use_multi_scale = True        # ✅
use_spp = True                # ✅
use_mv_loss = False           # ❌
```

### 实验 10：Full - MV (验证 MV 的边际贡献)

```python
use_hybrid_attention = True   # ✅
use_dldl_v2 = True            # ✅
use_multi_scale = True        # ✅
use_spp = True                # ✅
use_mv_loss = False           # ❌ (移除 MV)
```

---

## 最终表格模板

完成所有实验后，整理为以下表格用于论文：

### 表 1：模块消融实验结果

| Method | HA | DLDL | MSFF | SPP | MV | Selected_Test_MAE ↓ | MAE_raw | MAE_flip | MAE_multi | Params |
|:-------|:--:|:----:|:----:|:---:|:--:|:-------------------:|:-------:|:--------:|:---------:|:------:|
| Baseline | OFF | OFF | OFF | OFF | OFF | __.__ ± __.__ | __.__ | __.__ | __.__ | 4.84M |
| +HA | ON | OFF | OFF | OFF | OFF | __.__ ± __.__ | __.__ | __.__ | __.__ | 4.84M |
| +DLDL | OFF | ON | OFF | OFF | OFF | __.__ ± __.__ | __.__ | __.__ | __.__ | 4.84M |
| +MSFF | OFF | OFF | ON | OFF | OFF | __.__ ± __.__ | __.__ | __.__ | __.__ | 4.84M |
| +SPP | OFF | OFF | OFF | ON | OFF | __.__ ± __.__ | __.__ | __.__ | __.__ | 4.84M |
| **FADE-Net** | ON | ON | ON | ON | ON | **__.__ ± __.__** | __.__ | __.__ | __.__ | 4.84M |

> 注：`Selected_Test_MAE` 为 3 个 seed 的平均值和标准差；`MAE_raw`、`MAE_flip`、`MAE_multi` 用于说明 TTA 差异，最终论文中需明确选择口径。

### 表 2：轻量骨干筛选结果

| Backbone | Seed | Selected_Test_MAE | MAE_raw | MAE_flip | MAE_multi | Params | FLOPs | GPU latency |
|:---------|:---:|:-----------------:|:-------:|:--------:|:---------:|:------:|:-----:|:-----------:|
| `torchvision/mobilenet_v3_large` | 42 | TBD | TBD | TBD | TBD | 4.84M | 1.05G | TBD |
| `timm/mobilenetv4_conv_small` | 42 | TBD | TBD | TBD | TBD | 3.98M | 292.39M | 11.80 ms |
| `timm/mobilenetv4_conv_small_050` | 42 | TBD | TBD | TBD | TBD | 2.08M | 119.76M | 13.45 ms |
| `timm/repvit_m0_9` | 42 | TBD | TBD | TBD | TBD | 6.27M | 894.76M | 21.35 ms |

---

## 时间估算

| 配置 | 每个 Seed 耗时 | 3 Seeds 总耗时 |
|------|---------------|----------------|
| 单个配置 | ~2-3 小时 (GPU) | ~6-9 小时 |
| 方案 A (6 组) | - | ~36-54 小时 |
| 方案 B (10 组) | - | ~60-90 小时 |

---

## 注意事项

1. **不要删除 split 文件**：`dataset_split_AFAD_72_8_20.json` 保证所有实验使用相同的数据划分
2. **保存所有 checkpoint**：便于后续分析或复现
3. **记录异常**：如果某个 seed 表现异常，记录并分析原因
4. **GPU 内存**：Baseline 配置显存占用最小，Full Model 最大，注意 OOM

---

## 快速恢复

如果训练中断，可以恢复：

```bash
# 只有确认 metadata 匹配时才恢复
.\.venv\Scripts\python.exe src/train.py --seed 42 --split 72-8-20 --resume
```
