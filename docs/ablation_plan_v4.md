# FADE-Net 消融实验计划（MobileNetV4-Small）

## 背景

主 backbone 已切换为 `timm/mobilenetv4_conv_small`（2024 架构，内置高效注意力）。
HA（CoordAtt 注入）不适用于 timm backbone（无 SE 块可替换），消融聚焦于 MSFF、SPP、DLDL、MV Loss 四个模块。

## 协议

- **数据集**：AFAD-Full，72-8-20 分层划分
- **Split tag**：`formal_v1`（带 dataset_fingerprint）
- **Seeds**：42, 3407, 2026（3 seeds，用于 mean/std）
- **Epochs**：20
- **Pretrained**：是（ImageNet 预训练权重）
- **审计**：所有结果必须通过 `scripts/audit_paper_results.py --split_file_tag formal_v1` 判定 `paper-ready`

## 消融配置

| 编号 | 配置名 | DLDL | MV | MSFF | SPP | 说明 |
|------|--------|------|----|------|-----|------|
| A0 | Baseline | Y | Y | - | - | V4-small + FC head + DLDL + MV |
| A1 | +MSFF | Y | Y | Y | - | 加多尺度特征融合 |
| A2 | +SPP | Y | Y | - | Y | 加空间金字塔池化 |
| A3 | Full | Y | Y | Y | Y | 完整模型 |

> **为什么 Baseline 保留 DLDL + MV**：这两项是年龄估计任务的核心 loss 设计（标签分布学习 + 均值方差约束），不是结构模块。消融目的是验证结构模块（MSFF、SPP）的增量贡献。

## 训练命令

### 公共参数

```
AFAD_DIR="F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full"
SPLIT_TAG="formal_v1"
EPOCHS=20
```

### A0: Baseline（V4-small + DLDL + MV，无 MSFF，无 SPP）

```powershell
# Seed 42
.\.venv\Scripts\python.exe -u src\train.py --seed 42 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-msff --no-spp

# Seed 3407
.\.venv\Scripts\python.exe -u src\train.py --seed 3407 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-msff --no-spp

# Seed 2026
.\.venv\Scripts\python.exe -u src\train.py --seed 2026 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-msff --no-spp
```

### A1: +MSFF（加多尺度特征融合，无 SPP）

```powershell
# Seed 42
.\.venv\Scripts\python.exe -u src\train.py --seed 42 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-spp

# Seed 3407
.\.venv\Scripts\python.exe -u src\train.py --seed 3407 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-spp

# Seed 2026
.\.venv\Scripts\python.exe -u src\train.py --seed 2026 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-spp
```

### A2: +SPP（加空间金字塔池化，无 MSFF）

```powershell
# Seed 42
.\.venv\Scripts\python.exe -u src\train.py --seed 42 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-msff

# Seed 3407
.\.venv\Scripts\python.exe -u src\train.py --seed 3407 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-msff

# Seed 2026
.\.venv\Scripts\python.exe -u src\train.py --seed 2026 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small --no-msff
```

### A3: Full（MSFF + SPP + DLDL + MV，全部启用）

> seed42 已有 formal_v1 结果（MAE=3.6130），只需补 seed 3407 和 2026。

```powershell
# Seed 3407
.\.venv\Scripts\python.exe -u src\train.py --seed 3407 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small

# Seed 2026
.\.venv\Scripts\python.exe -u src\train.py --seed 2026 --epochs 20 --split 72-8-20 --fresh --split_file_tag formal_v1 --afad_dir "F:/QQFiles/Study/shit/tarball/tarball-master/AFAD-Full.tar/AFAD-Full~/AFAD-Full" --backbone_source timm --backbone_name mobilenetv4_conv_small
```

## 训练后步骤

### 1. 审计所有结果

```powershell
.\.venv\Scripts\python.exe -B scripts/audit_paper_results.py --split_file_tag formal_v1 --candidates mobilenetv4_conv_small --seeds 42,3407,2026 --output docs/paper_result_audit_full.csv
```

### 2. 生成主表汇总

```powershell
.\.venv\Scripts\python.exe -B scripts/summarize_paper_results.py --audit_csv docs/paper_result_audit_full.csv --seeds 42,3407,2026 --output docs/paper_result_summary.md
```

### 3. 效率表

```powershell
.\.venv\Scripts\python.exe -B scripts/benchmark_speed.py --backbone_source timm --backbone_name mobilenetv4_conv_small
.\.venv\Scripts\python.exe -B scripts/calc_params.py --backbone_source timm --backbone_name mobilenetv4_conv_small
```

## 预期论文主表结构

| 配置 | Params | FLOPs | MAE (mean +/- std) | TTA |
|------|--------|-------|---------------------|-----|
| A0: Baseline | ? | ? | ? | multi |
| A1: +MSFF | ? | ? | ? | multi |
| A2: +SPP | ? | ? | ? | multi |
| A3: Full | ? | ? | ? | multi |

## 当前进度

- [x] A3 Full seed42: MAE=3.6130 (paper-ready)
- [ ] A3 Full seed3407
- [ ] A3 Full seed2026
- [ ] A0 Baseline seed42
- [ ] A0 Baseline seed3407
- [ ] A0 Baseline seed2026
- [ ] A1 +MSFF seed42
- [ ] A1 +MSFF seed3407
- [ ] A1 +MSFF seed2026
- [ ] A2 +SPP seed42
- [ ] A2 +SPP seed3407
- [ ] A2 +SPP seed2026
- [ ] 效率表
- [ ] 审计汇总
